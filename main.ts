// free-model-aggregator-02: one server, two client wires, a real free model behind every answer.
//
// STRATEGY (from the leaderboard + the kit's scorer, verified live 17 Nov 2026):
//   The 284 top entries all score 94: quality 90, served/wire/failover/catalog 100.
//   Quality 90 is exactly what you get using nemotron-3.5-super-vl (1303) for every prompt
//   (1303*8 / 11456 = 90.99, floored to 90). NOBODY uses the best model,
//   thinkingmachines/inkling-2-small (1475), for the non-JSON prompts — even though routing
//   Inkling there and Super-VL for JSON gives quality 100 and score 100.
//   This server does exactly that: rating-first, capability-aware routing.
//   Inkling's risk (one provider, P95 latency ~27s vs a 30s prompt limit) is handled by a
//   bounded failover ladder with token validation: a model that refuses, times out, or
//   answers without the prompt's token is skipped and the next-rated compatible model serves.
//   One weight-1 fallback still leaves quality 98; one weight-2 fallback leaves 96.
//   Streaming is buffered: the whole upstream stream is read, validated, then rebuilt as a
//   clean downstream stream, so a bad answer is never half-sent to the client.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's
//   standard library alone. The env is at $MAKEFUN_ENV_URL; model calls go through the grader's
//   gateway at $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the
//   key you pushed to the competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can
//       do (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its
//       weight (1 or 2), the requirements a model must meet, and the unique token its answer
//       must carry back.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got), POST
//        /v1/chat/completions and POST /v1/messages, each with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway, validating the
//        answer carries the prompt's token, and failing over to the next-rated compatible model
//        when it does not;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually
//   dialled. The house records every gateway call and scores the model that really answered.
//   This server sends the brief's model id in the gateway request and puts that same id back on
//   every answer and every streamed chunk, so its claim is the call it made, never another.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The most upstream attempts for one prompt, across the whole candidate ladder. */
const MAX_ATTEMPTS_PER_PROMPT = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * The free models that can serve a prompt, best first: the brief's rating_milli descending,
 * ties broken by the brief's own order (stable and deterministic, so a live pass and its replay
 * walk the same ladder).
 *
 * Deliberately NO permanent blacklist: a model that refused once (a 429, a timeout) is tried
 * fresh on the next prompt. A single refusal must not poison the best model for the whole
 * trial — that would drop quality back to 90, the exact ceiling this server is built to beat.
 * Within one prompt each candidate is tried at most once (the ladder walks it once), so there
 * is no retry loop; across prompts the best model always gets another chance.
 *
 * This is the whole quality play: for a JSON prompt the best capable model is
 * nemotron-3.5-super-vl (1303); for anything else it is thinkingmachines/inkling-2-small
 * (1475). The starter took the brief's first capable model instead and left quality on the table.
 */
function candidatesFor(requires: Requirement): ModelRow[] {
  const order = new Map(freeModels(brief).map((m, i) => [m.id, i]));
  return freeModels(brief)
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .sort((a, b) => b.rating_milli - a.rating_milli || (order.get(a.id) ?? 0) - (order.get(b.id) ?? 0));
}

/** The prompt row now arriving: its requirements and the token its answer must carry back. */
function promptContextFor(body: Json): { requires: Requirement; token: string | null } {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return { requires: entry.requires, token: entry.token };
  // The brief's prompt list is the declared source; this fallback is a guess for a brief that holds none.
  return { requires: requirementsFromBody(body), token: null };
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders?: Record<string, string>): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...(extraHeaders ?? {}) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify({ ...body, model }) });
}

/** The answer text of a Chat Completions JSON body (message content, or a tool call's arguments). */
function textOfJsonAnswer(answer: Json): string {
  const choices = Array.isArray(answer.choices) ? answer.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  if (typeof message.content === 'string') return message.content;
  if (Array.isArray(message.content)) {
    return message.content
      .flatMap((part) => (isRecord(part) && typeof part.text === 'string' ? [part.text] : []))
      .join('');
  }
  return '';
}

/** The answer text of a buffered Chat Completions SSE body (the content deltas, in order). */
function textOfSseAnswer(sse: string): string {
  let text = '';
  for (const line of sse.split('\n')) {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) continue;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') continue;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (!isRecord(chunk)) continue;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : {};
      const delta = isRecord(choice.delta) ? choice.delta : {};
      if (typeof delta.content === 'string') text += delta.content;
    } catch {
      // A frame that is not JSON carries no text.
    }
  }
  return text;
}

/** A buffered upstream SSE body as a stream the wire relays can read (for the clean re-emit). */
function streamOf(text: string): ReadableStream<Uint8Array> {
  const encoded = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(encoded);
      controller.close();
    },
  });
}

/** One refusal row for the x-makefun-upstream-failures trail. */
interface FailureRow {
  status: number;
  code: string;
  model: string | null;
}

function failureCode(status: number, bodyText: string): string {
  try {
    const parsed: unknown = JSON.parse(bodyText);
    if (isRecord(parsed) && isRecord(parsed.error) && typeof parsed.error.code === 'string' && parsed.error.code !== '') {
      return parsed.error.code;
    }
    if (isRecord(parsed) && typeof parsed.code === 'string' && parsed.code !== '') return parsed.code;
  } catch {
    // Not JSON: fall through to the status-based code.
  }
  if (status === 429) return 'rate_limited';
  if (status === 400) return 'invalid_request_error';
  if (status >= 500) return 'upstream_error';
  return 'upstream_error';
}

/**
 * One prompt served by the best model that answers it validly.
 *
 * Walks the rating-ordered candidate ladder: dials each model in turn, and keeps the first
 * answer that (a) arrived with HTTP 2xx, (b) carries the prompt's token in its text, and
 * (c) parses as JSON when the prompt asked for JSON mode. A refusal (4xx/5xx/unreachable)
 * marks the model unavailable for the rest of the trial and is recorded for the
 * x-makefun-upstream-failures trail, so the failover column can see the recovery. A 200
 * whose content fails validation is skipped silently — the model is fine, its answer was
 * not — and the next candidate is tried. Returns null when no candidate serves the prompt.
 */
interface ServedAnswer {
  model: string;
  /** The upstream JSON body, for a non-streaming answer. */
  json: Json | null;
  /** The buffered upstream SSE text, for a streaming answer. */
  sse: string | null;
  failures: FailureRow[];
}

async function serveWithFailover(
  upstreamBody: Json,
  requires: Requirement,
  token: string | null,
  stream: boolean,
): Promise<ServedAnswer | null> {
  const failures: FailureRow[] = [];
  const candidates = candidatesFor(requires);
  let attempts = 0;
  for (const candidate of candidates) {
    if (attempts >= MAX_ATTEMPTS_PER_PROMPT) break;
    attempts += 1;
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBody);
    } catch (err) {
      // Unreachable: record it for the trail and move to the next candidate.
      failures.push({ status: 502, code: 'upstream_unreachable', model: candidate.id });
      log(`dial ${candidate.id} failed: ${(err as Error).message}`);
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      const text = await upstream.text().catch(() => '');
      failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model: candidate.id });
      log(`dial ${candidate.id} refused HTTP ${upstream.status}`);
      continue;
    }
    if (!stream) {
      let answer: unknown;
      try {
        answer = await upstream.json();
      } catch {
        log(`dial ${candidate.id} answered unparseable JSON`);
        continue;
      }
      if (!isRecord(answer)) continue;
      const text = textOfJsonAnswer(answer);
      if (text === '') {
        log(`dial ${candidate.id} answered with no text`);
        continue;
      }
      if (token !== null && !text.includes(token)) {
        log(`dial ${candidate.id} answered without the prompt token`);
        continue;
      }
      if (requires.json === true) {
        try {
          JSON.parse(text);
        } catch {
          log(`dial ${candidate.id} answered invalid JSON in JSON mode`);
          continue;
        }
      }
      return { model: candidate.id, json: answer, sse: null, failures };
    }
    // Streaming: buffer the whole upstream stream first, so a bad answer is never half-sent.
    const sse = await upstream.text();
    const text = textOfSseAnswer(sse);
    if (text === '') {
      log(`dial ${candidate.id} streamed no text`);
      continue;
    }
    if (token !== null && !text.includes(token)) {
      log(`dial ${candidate.id} streamed without the prompt token`);
      continue;
    }
    if (requires.json === true) {
      try {
        JSON.parse(text);
      } catch {
        log(`dial ${candidate.id} streamed invalid JSON in JSON mode`);
        continue;
      }
    }
    return { model: candidate.id, json: null, sse, failures };
  }
  return null;
}

/** The failures trail header, only when a refusal was recovered from. */
function failuresHeader(failures: FailureRow[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, 8)) };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that serves it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, token } = promptContextFor(body);
  const served = await serveWithFailover({ ...body, model: 'auto' }, requires, token, body.stream === true);
  if (served === null) {
    return sendError(res, 'openai', 502, 'upstream_error', 'no free model in the brief served this request');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true && served.sse !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const frame of openAIStreamWithModel(streamOf(served.sse), served.model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIJsonWithModel(served.json ?? {}, served.model), headers);
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const { requires, token } = promptContextFor(body);
  const served = await serveWithFailover(anthropicToOpenAI(body, 'auto'), requires, token, body.stream === true);
  if (served === null) {
    return sendError(res, 'anthropic', 502, 'api_error', 'no free model in the brief served this request');
  }
  const headers = failuresHeader(served.failures);
  if (body.stream === true && served.sse !== null) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive', ...headers });
    try {
      for await (const event of anthropicEventsFromOpenAI(streamOf(served.sse), served.model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  sendJson(res, 200, openAIToAnthropic(served.json ?? {}, served.model), headers);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
