import http from 'http';
import https from 'https';
import fs from 'fs';

const config = JSON.parse(fs.readFileSync(new URL('config.json', import.meta.url), 'utf8'));
const GO_API_KEY = process.env.OPENCODE_GO_KEY || config.api_key;
const PORT = parseInt(process.env.PORT || config.port || '8080');

const MODEL_MAP = { ...config.model_mapping, default: config.default_model || 'deepseek-v4-flash' };
const ANTHROPIC_MODELS = Object.keys(MODEL_MAP).filter(k => k !== 'default').map(id => ({
  id, created: 1745880000, owned_by: 'opencode-go',
}));

function cleanModel(name) {
  return name.replace(/\[\w+\]/g, '');
}

function mapModel(name) {
  return MODEL_MAP[cleanModel(name)] || MODEL_MAP['default'];
}

function goRequest(body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = https.request({
      hostname: 'opencode.ai',
      path: '/zen/go/v1/chat/completions',
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + GO_API_KEY,
        'Content-Length': Buffer.byteLength(data),
      },
    }, (res) => {
      let chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const raw = Buffer.concat(chunks).toString();
        try { resolve(JSON.parse(raw)); }
        catch { reject(new Error(raw)); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function toAnthropic(goResp, anthropicModel) {
  const c = (goResp.choices || [])[0] || {};
  const m = c.message || {};
  const content = m.content || '';
  const reasoning = m.reasoning_content || null;
  const toolCalls = m.tool_calls || [];
  const usage = goResp.usage || {};

  const anthroContent = [];

  if (reasoning) {
    anthroContent.push({ type: 'thinking', thinking: reasoning, signature: '' });
  }

  // If finish_reason==tool_calls and content is JSON, parse as tool_use
  let textContent = content;
  if (c.finish_reason === 'tool_calls' && content.trim().startsWith('{"type":"tool_use"')) {
    try {
      const parsed = JSON.parse(content.trim());
      const arr = Array.isArray(parsed) ? parsed : [parsed];
      for (const tu of arr) {
        anthroContent.push({ type: 'tool_use', id: tu.id, name: tu.name, input: tu.input || {} });
      }
      textContent = '';
    } catch {}
  }

  if (textContent) {
    anthroContent.push({ type: 'text', text: textContent });
  }

  for (const tc of toolCalls) {
    let input = {};
    try { input = JSON.parse(tc.function.arguments || '{}'); } catch {}
    anthroContent.push({ type: 'tool_use', id: tc.id, name: tc.function.name, input });
  }

  let stopReason = 'end_turn';
  if (c.finish_reason === 'tool_calls') stopReason = 'tool_use';
  else if (c.finish_reason === 'length') stopReason = 'max_tokens';

  return {
    id: 'msg_' + (goResp.id?.replace(/chatcmpl[_-]/, '') || Date.now()),
    type: 'message',
    role: 'assistant',
    content: anthroContent,
    model: anthropicModel,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: usage.prompt_tokens || 0,
      output_tokens: usage.completion_tokens || 0,
      cache_read_input_tokens: usage.prompt_cache_hit_tokens || usage.prompt_tokens_details?.cached_tokens || 0,
    },
  };
}

function streamAnthropic(anthroResp, res) {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'anthropic-version': '2023-06-01',
  });

  const sse = (event, data) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  const msgId = anthroResp.id;
  const blocks = anthroResp.content;

  sse('message_start', {
    type: 'message_start',
    message: { id: msgId, type: 'message', role: 'assistant', content: [], model: anthroResp.model, stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
  });

  blocks.forEach((block, i) => {
    sse('content_block_start', { type: 'content_block_start', index: i, content_block: block });
    if (block.type === 'thinking') {
      if (block.thinking) {
        sse('content_block_delta', { type: 'content_block_delta', index: i, delta: { type: 'thinking_delta', thinking: block.thinking } });
      }
    } else if (block.type === 'text') {
      if (block.text) {
        sse('content_block_delta', { type: 'content_block_delta', index: i, delta: { type: 'text_delta', text: block.text } });
      }
    }
    sse('content_block_stop', { type: 'content_block_stop', index: i });
  });

  sse('message_delta', {
    type: 'message_delta',
    delta: { stop_reason: anthroResp.stop_reason, stop_sequence: null },
    usage: { output_tokens: anthroResp.usage.output_tokens },
  });

  sse('message_stop', { type: 'message_stop' });
  res.end();
}

function convertToOpenAI(anthropicReq) {
  const model = mapModel(anthropicReq.model);
  const messages = [];

  if (anthropicReq.system) {
    const text = typeof anthropicReq.system === 'string'
      ? anthropicReq.system
      : anthropicReq.system.map(s => s.text || s).join('\n');
    messages.push({ role: 'system', content: text });
  }

  for (const msg of (anthropicReq.messages || [])) {
    let content;
    if (typeof msg.content === 'string') {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      const texts = msg.content.filter(c => c.type === 'text').map(c => c.text);
      content = texts.join('\n');
    }
    messages.push({ role: msg.role, content: content || '' });
  }

  const req = {
    model,
    messages,
    max_tokens: anthropicReq.max_tokens || 8192,
    stream: false,
    temperature: anthropicReq.temperature,
    top_p: anthropicReq.top_p,
  };

  if (anthropicReq.stop_sequences) req.stop = anthropicReq.stop_sequences;

  if (anthropicReq.tools) {
    req.tools = anthropicReq.tools.map(t => ({
      type: 'function',
      function: { name: t.name, description: t.description || '', parameters: t.input_schema || {} },
    }));
  }

  if (anthropicReq.tool_choice) {
    if (anthropicReq.tool_choice.type === 'tool') {
      req.tool_choice = { type: 'function', function: { name: anthropicReq.tool_choice.name } };
    } else if (anthropicReq.tool_choice.type === 'any') {
      req.tool_choice = 'required';
    } else {
      req.tool_choice = anthropicReq.tool_choice.type;
    }
  }

  if (anthropicReq.metadata) req.user = anthropicReq.metadata.user_id;
  return req;
}

const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const method = req.method;

  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-api-key, anthropic-version');

  if (method === 'OPTIONS') { res.writeHead(204); res.end(); return; }
  if (method === 'GET' && url.pathname === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ data: ANTHROPIC_MODELS }));
    return;
  }

  if (method === 'POST' && url.pathname === '/v1/messages') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      try {
        const anthroReq = JSON.parse(body);
        const openAIReq = convertToOpenAI(anthroReq);

        console.log(`[${new Date().toISOString().slice(11,19)}] model=${openAIReq.model} messages=${openAIReq.messages.length} tools=${openAIReq.tools?.length || 0} tool_choice=${JSON.stringify(openAIReq.tool_choice)}`);

        const goResp = await goRequest(openAIReq);
        const anthroResp = toAnthropic(goResp, anthroReq.model);

        if (anthroReq.stream) {
          streamAnthropic(anthroResp, res);
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json', 'anthropic-version': '2023-06-01' });
          res.end(JSON.stringify(anthroResp));
        }
      } catch (err) {
        console.log(`ERROR: ${err.message}`);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: err.message, type: 'api_error' } }));
      }
    });
    return;
  }

  if (method === 'GET' && (url.pathname === '/' || url.pathname === '/health')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', provider: 'opencode-go' }));
    return;
  }

  res.writeHead(404); res.end();
});

server.listen(PORT, () => {
  console.log(`OpenCode Go Proxy running on http://localhost:${PORT}`);
  console.log(`Models: ${Object.keys(MODEL_MAP).filter(k=>k!=='default').join(', ')}`);
});
