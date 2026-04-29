import http from 'http';
import https from 'https';
import fs from 'fs';

const config = JSON.parse(fs.readFileSync(new URL('config.json', import.meta.url), 'utf8'));

const GO_API_KEY = process.env.OPENCODE_GO_KEY || config.api_key;
const GO_BASE = 'opencode.ai';
const GO_PATH = '/zen/go/v1';
const PORT = parseInt(process.env.PORT || config.port || '8080');

const MODEL_MAP = { ...config.model_mapping, default: config.default_model || 'deepseek-v4-flash' };

const ANTHROPIC_MODELS = Object.keys(MODEL_MAP).filter(k => k !== 'default').map(id => ({
  id, created: 1745880000, owned_by: 'opencode-go',
}));

function cleanModel(name) {
  return name.replace(/\[\w+\]/g, '');
}

function mapModel(anthropicModel) {
  return MODEL_MAP[cleanModel(anthropicModel)] || MODEL_MAP['default'];
}

function anthropicToOpenAI(body) {
  const model = mapModel(body.model);
  const messages = [];

  if (body.system) {
    const text = typeof body.system === 'string'
      ? body.system
      : body.system.map(s => s.text || s).join('\n');
    messages.push({ role: 'system', content: text });
  }

  for (const msg of (body.messages || [])) {
    let content = '';
    const parts = [];
    if (typeof msg.content === 'string') {
      content = msg.content;
    } else if (Array.isArray(msg.content)) {
      for (const c of msg.content) {
        if (c.type === 'text') parts.push(c.text);
        else if (c.type === 'tool_use') parts.push(JSON.stringify(c));
        else if (c.type === 'tool_result') parts.push(c.content);
      }
      content = parts.join('\n');
    }
    messages.push({ role: msg.role, content });
  }

  const req = {
    model,
    messages,
    max_tokens: body.max_tokens || 8192,
    stream: body.stream || false,
    temperature: body.temperature,
    top_p: body.top_p,
  };

  if (body.stop_sequences) req.stop = body.stop_sequences;

  if (body.tools) {
    req.tools = body.tools.map(t => ({
      type: 'function',
      function: {
        name: t.name,
        description: t.description || '',
        parameters: t.input_schema || {},
      }
    }));
  }

  if (body.tool_choice) {
    if (body.tool_choice.type === 'tool') {
      req.tool_choice = { type: 'function', function: { name: body.tool_choice.name } };
    } else if (body.tool_choice.type === 'any') {
      req.tool_choice = 'required';
    } else {
      req.tool_choice = body.tool_choice.type;
    }
  }

  if (body.metadata) req.user = body.metadata.user_id;

  return req;
}

function openAIToAnthropic(resp, originalModel) {
  const choice = (resp.choices || [])[0] || {};
  const msg = choice.message || {};
  const content = msg.content || '';
  const toolCalls = msg.tool_calls || [];
  const reasoning = msg.reasoning_content || msg.reasoning || null;

  const anthroContent = [];

  if (reasoning) {
    anthroContent.push({ type: 'thinking', thinking: reasoning, signature: '' });
  }

  if (content) {
    anthroContent.push({ type: 'text', text: content });
  }

  for (const tc of toolCalls) {
    let input = {};
    try { input = JSON.parse(tc.function.arguments || '{}'); } catch {}
    anthroContent.push({
      type: 'tool_use',
      id: tc.id,
      name: tc.function.name,
      input,
    });
  }

  let stopReason = 'end_turn';
  if (choice.finish_reason === 'tool_calls') stopReason = 'tool_use';
  else if (choice.finish_reason === 'length') stopReason = 'max_tokens';
  else if (choice.finish_reason === 'stop') stopReason = 'end_turn';

  return {
    id: 'msg_' + resp.id?.replace('chatcmpl-', '').replace('chatcmpl_', '') || Date.now(),
    type: 'message',
    role: 'assistant',
    content: anthroContent,
    model: originalModel,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: {
      input_tokens: resp.usage?.prompt_tokens || 0,
      output_tokens: resp.usage?.completion_tokens || 0,
      cache_creation_input_tokens: resp.usage?.prompt_tokens_details?.cached_tokens || 0,
      cache_read_input_tokens: resp.usage?.prompt_cache_hit_tokens || 0,
    },
  };
}

function requestGo(options, body) {
  return new Promise((resolve, reject) => {
    const data = JSON.stringify(body);
    const req = https.request({
      hostname: GO_BASE,
      path: GO_PATH + options.path,
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + GO_API_KEY,
        'Content-Length': Buffer.byteLength(data),
        ...(options.headers || {}),
      },
    }, (res) => {
      let chunks = [];
      res.on('data', c => chunks.push(c));
      res.on('end', () => {
        const raw = Buffer.concat(chunks).toString();
        try { resolve(JSON.parse(raw)); }
        catch { resolve(raw); }
      });
    });
    req.on('error', reject);
    req.write(data);
    req.end();
  });
}

function streamGo(body, res, originalModelName) {
  const data = JSON.stringify({ ...body, stream: true });
  const goReq = https.request({
    hostname: GO_BASE,
    path: GO_PATH + '/chat/completions',
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer ' + GO_API_KEY,
      'Content-Length': Buffer.byteLength(data),
      'Accept': 'text/event-stream',
    },
  });

  const originalModel = originalModelName || body.model;
  let buffer = '';
  let messageId = 'msg_' + Date.now();
  let hasContent = false;
  let hasThinking = false;
  let contentIndex = 0;
  let currentToolCall = null;

  function sendSSE(event, data) {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  }

  goReq.on('response', (goRes) => {
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'anthropic-version': '2023-06-01',
    });

    // Send message_start
    sendSSE('message_start', {
      type: 'message_start',
      message: {
        id: messageId,
        type: 'message',
        role: 'assistant',
        content: [],
        model: originalModel,
        stop_reason: null,
        stop_sequence: null,
        usage: { input_tokens: 0, output_tokens: 0 },
      },
    });

    let pingInterval = setInterval(() => {
      res.write('event: ping\ndata: {}\n\n');
    }, 5000);

    goRes.on('data', (chunk) => {
      buffer += chunk.toString();
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (jsonStr === '[DONE]') continue;

        try {
          const parsed = JSON.parse(jsonStr);
          const delta = parsed.choices?.[0]?.delta || {};
          const finishReason = parsed.choices?.[0]?.finish_reason;

          // Reasoning content
          if (delta.reasoning_content) {
            if (!hasThinking) {
              hasThinking = true;
              sendSSE('content_block_start', {
                type: 'content_block_start',
                index: contentIndex,
                content_block: { type: 'thinking', thinking: '', signature: '' },
              });
            }
            sendSSE('content_block_delta', {
              type: 'content_block_delta',
              index: contentIndex,
              delta: { type: 'thinking_delta', thinking: delta.reasoning_content },
            });
          }

          // Text content
          if (delta.content) {
            if (!hasContent && !hasThinking) {
              hasContent = true;
              sendSSE('content_block_start', {
                type: 'content_block_start',
                index: contentIndex,
                content_block: { type: 'text', text: '' },
              });
            } else if (!hasContent && hasThinking) {
              // Close thinking block, start text
              sendSSE('content_block_stop', {
                type: 'content_block_stop',
                index: contentIndex,
              });
              contentIndex++;
              hasContent = true;
              hasThinking = false;
              sendSSE('content_block_start', {
                type: 'content_block_start',
                index: contentIndex,
                content_block: { type: 'text', text: '' },
              });
            }
            if (hasContent) {
              sendSSE('content_block_delta', {
                type: 'content_block_delta',
                index: contentIndex,
                delta: { type: 'text_delta', text: delta.content },
              });
            }
          }

          // Tool calls
          if (delta.tool_calls) {
            for (const tc of delta.tool_calls) {
              if (tc.index !== undefined && (!currentToolCall || currentToolCall.index !== tc.index)) {
                if (currentToolCall) {
                  sendSSE('content_block_stop', {
                    type: 'content_block_stop',
                    index: contentIndex,
                  });
                  contentIndex++;
                }
                currentToolCall = { index: tc.index, id: tc.id || '', name: tc.function?.name || '', args: '' };
                if (hasContent || hasThinking) contentIndex++;
                hasContent = false; hasThinking = false;
                sendSSE('content_block_start', {
                  type: 'content_block_start',
                  index: contentIndex,
                  content_block: { type: 'tool_use', id: currentToolCall.id, name: currentToolCall.name, input: {} },
                });
              }
              if (tc.function?.arguments) {
                currentToolCall.args += tc.function.arguments;
                sendSSE('content_block_delta', {
                  type: 'content_block_delta',
                  index: contentIndex,
                  delta: { type: 'input_json_delta', partial_json: tc.function.arguments },
                });
              }
              if (tc.id) currentToolCall.id = tc.id;
              if (tc.function?.name) currentToolCall.name = tc.function.name;
            }
          }

          // Finish
          if (finishReason) {
            if (currentToolCall) {
              sendSSE('content_block_stop', { type: 'content_block_stop', index: contentIndex });
            }
            if (hasContent) {
              sendSSE('content_block_stop', { type: 'content_block_stop', index: contentIndex });
            }
            if (hasThinking) {
              sendSSE('content_block_stop', { type: 'content_block_stop', index: contentIndex });
            }

            let stopReason = 'end_turn';
            if (finishReason === 'tool_calls') stopReason = 'tool_use';
            else if (finishReason === 'length') stopReason = 'max_tokens';

            sendSSE('message_delta', {
              type: 'message_delta',
              delta: { stop_reason: stopReason, stop_sequence: null },
              usage: {
                output_tokens: parsed.usage?.completion_tokens || 0,
              },
            });

            sendSSE('message_stop', { type: 'message_stop' });
          }
        } catch {}
      }
    });

    goRes.on('end', () => {
      clearInterval(pingInterval);
      res.end();
    });

    goRes.on('error', () => {
      clearInterval(pingInterval);
      res.end();
    });
  });

  goReq.on('error', (err) => {
    res.writeHead(502, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: { message: err.message, type: 'proxy_error' } }));
  });

  goReq.write(data);
  goReq.end();
}

function log(req, msg) {
  const ts = new Date().toISOString().slice(11, 19);
  const id = req?._id || (req._id = Math.random().toString(36).slice(2, 6));
  console.log(`[${ts}][${id}] ${msg}`);
}

// --- Server ---
const server = http.createServer((req, res) => {
  const url = new URL(req.url, `http://localhost:${PORT}`);
  const method = req.method;
  log(req, `${method} ${url.pathname}`);

  // CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, x-api-key, anthropic-version');

  if (method === 'OPTIONS') {
    res.writeHead(204);
    res.end();
    return;
  }

  // GET /v1/models
  if (method === 'GET' && url.pathname === '/v1/models') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ data: ANTHROPIC_MODELS }));
    return;
  }

  // POST /v1/messages
  if (method === 'POST' && url.pathname === '/v1/messages') {
    let body = '';
    req.on('data', c => body += c);
    req.on('end', async () => {
      try {
        const anthropicReq = JSON.parse(body);
        log(req, `model=${anthropicReq.model} stream=${anthropicReq.stream} messages=${anthropicReq.messages?.length} tools=${anthropicReq.tools?.length || 0}`);
        const openAIReq = anthropicToOpenAI(anthropicReq);

        if (anthropicReq.stream) {
          streamGo(openAIReq, res, anthropicReq.model);
          return;
        }

        log(req, `→ OpenAI chat completions`);
        const goResp = await requestGo({ path: '/chat/completions' }, openAIReq);
        const anthroResp = openAIToAnthropic(goResp, anthropicReq.model);
        log(req, `← ${anthroResp.content.length} content blocks, stop_reason=${anthroResp.stop_reason}`);
        res.writeHead(200, {
          'Content-Type': 'application/json',
          'anthropic-version': '2023-06-01',
        });
        res.end(JSON.stringify(anthroResp));
      } catch (err) {
        log(req, `ERROR: ${err.message}`);
        res.writeHead(400, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: { message: err.message, type: 'invalid_request' } }));
      }
    });
    return;
  }

  // Health
  if (method === 'GET' && (url.pathname === '/' || url.pathname === '/health')) {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'ok', provider: 'opencode-go' }));
    return;
  }

  res.writeHead(404);
  res.end();
});

server.listen(PORT, () => {
  console.log(`OpenCode Go Proxy (Anthropic API) running on http://localhost:${PORT}`);
  console.log(`Using OpenCode Go API key: ${GO_API_KEY.slice(0, 12)}...`);
  console.log('---');
  console.log('To use with Claude Code:');
  console.log(`  export ANTHROPIC_BASE_URL=http://localhost:${PORT}`);
  console.log(`  export ANTHROPIC_API_KEY=any-value`);
  console.log('  claude');
  console.log('---');
  console.log(`Models: ${Object.keys(MODEL_MAP).join(', ')}`);
  console.log(`Mapped to Go models: ${Object.values(MODEL_MAP).filter((v,i,a)=>a.indexOf(v)===i).join(', ')}`);
});
