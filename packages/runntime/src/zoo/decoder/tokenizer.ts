/** A self-contained reader of Hugging Face tokenizer.json BPE tokenizers, in
 *  the two flavours the decoder models use.
 *
 *  SentencePiece-style (Gemma 3):
 *    1. added tokens (`<start_of_turn>`, `<unused0>`, html tags, ...) split
 *       the text first and map straight to their ids;
 *    2. every other run has ' ' replaced by '▁' (no prefix space);
 *    3. BPE over code points, lowest merge rank first (ties: leftmost); a code
 *       point missing from the vocab falls back to its UTF-8 bytes `<0xNN>`.
 *    Decoding reverses it: '▁' → ' ', byte tokens are re-assembled as UTF-8.
 *
 *  Byte-level (Qwen 3, GPT-2 lineage):
 *    1. added tokens (`<|im_start|>`, `<think>`, ...) split the text first;
 *    2. every other run is NFC-normalized and cut by the pre-tokenizer regex;
 *    3. each piece's UTF-8 bytes are mapped to printable code points
 *       (GPT-2's bytes_to_unicode) and merged by the same BPE.
 *    Decoding maps the code points back to bytes.
 *
 *  Each tokenizer carries its model's chat template. */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'model' | 'tool' | (string & {});
  content: string;
}

export interface ChatTemplateOptions {
  /** End with an open assistant turn. Default true. */
  addGenerationPrompt?: boolean;
  /** Qwen 3: let the model think first. Default false, which pre-fills an
   *  empty `<think>\n\n</think>\n\n` block after the generation prompt (the
   *  HF template's enable_thinking=False). Ignored by Gemma. */
  enableThinking?: boolean;
  /** Qwen 3: function schemas listed in the system turn, rendered as the HF
   *  template's `tools=` does. Ignored by Gemma. */
  tools?: readonly unknown[];
}

/** Which chat template a tokenizer applies. */
export type ChatFormat = 'gemma' | 'qwen3';

export interface TextTokenizer {
  readonly chatFormat: ChatFormat;
  readonly vocabSize: number;
  /** Prepended by `encode(..., { addBos: true })`; undefined where the model
   *  has none (Qwen). */
  readonly bosId: number | undefined;
  readonly eosId: number;
  /** Closes a chat turn (`<end_of_turn>`, `<|im_end|>`). */
  readonly endOfTurnId: number;
  /** Ids that end generation: end of turn, end of text. */
  readonly stopIds: readonly number[];
  /** Text → ids. `addBos` prepends the bos token where the model has one
   *  (default false). Added tokens (`<start_of_turn>`, `<|im_start|>`, ...)
   *  in the text are parsed as their single ids unless `parseSpecial: false`. */
  encode(text: string, opts?: { addBos?: boolean; parseSpecial?: boolean }): number[];
  /** Ids → text. Special tokens (`<bos>`, `<end_of_turn>`, `<|im_end|>`, ...)
   *  are dropped unless `skipSpecial: false`. */
  decode(ids: readonly number[], opts?: { skipSpecial?: boolean }): string;
  /** Incremental decoder: push ids one at a time, get back the newly complete
   *  text (multi-byte characters split over byte tokens are held back). */
  createStreamDecoder(opts?: { skipSpecial?: boolean }): {
    push(id: number): string;
    flush(): string;
  };
  tokenToId(token: string): number | undefined;
  idToToken(id: number): string | undefined;
  /** True for added tokens flagged special (`<bos>`, `<eos>`, turn markers). */
  isSpecial(id: number): boolean;
  /** The model's chat template, ending in an open assistant turn.
   *  Gemma: `<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n`
   *  (no system role: a system message is prepended to the first user turn).
   *  Qwen 3: ChatML, `<|im_start|>role\n...<|im_end|>\n`. */
  applyChatTemplate(messages: readonly ChatMessage[], opts?: ChatTemplateOptions): string;
}

/** The Gemma 3 tokenizer: a TextTokenizer with the Gemma template. */
export type Gemma3Tokenizer = TextTokenizer;

interface HfAddedToken {
  id: number;
  content: string;
  special: boolean;
}

interface HfPreTokenizer {
  type: string;
  pretokenizers?: HfPreTokenizer[];
  pattern?: { Regex?: string; String?: string };
  behavior?: string;
  use_regex?: boolean;
}

interface HfTokenizerJson {
  added_tokens?: HfAddedToken[];
  normalizer?: { type: string; normalizers?: { type: string }[] } | null;
  pre_tokenizer?: HfPreTokenizer | null;
  model: {
    type: string;
    vocab: Record<string, number>;
    merges: (string | [string, string])[];
    byte_fallback?: boolean;
    ignore_merges?: boolean;
    unk_token?: string | null;
  };
}

const BYTE_TOKEN = /^<0x([0-9A-Fa-f]{2})>$/;
const SPACE = '▁';
/** The GPT-2 split, for a ByteLevel pre-tokenizer with use_regex and no
 *  Split of its own. */
const GPT2_SPLIT =
  "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";

/** GPT-2's bytes_to_unicode: byte → printable code point. */
function byteToUnicode(): string[] {
  const out = new Array<string>(256);
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
    out[b] = String.fromCodePoint(printable ? b : 256 + n++);
  }
  return out;
}

/** An HF split pattern (Rust regex syntax) as a JS RegExp. Inline
 *  case-insensitive groups `(?i:...)`, which older JS engines reject, are
 *  rewritten with ASCII letter classes. */
function splitRegex(pattern: string): RegExp {
  const src = pattern.replace(/\(\?i:((?:[^()\\]|\\.)*)\)/g, (_, body: string) => {
    const caseless = body.replace(/[A-Za-z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`);
    return `(?:${caseless})`;
  });
  return new RegExp(src, 'gu');
}

/** The split regex of a byte-level pre-tokenizer, or undefined when the
 *  tokenizer is not byte-level. */
function byteLevelSplit(pre: HfPreTokenizer | null | undefined): RegExp | undefined {
  if (!pre) return undefined;
  const all = pre.type === 'Sequence' ? (pre.pretokenizers ?? []) : [pre];
  const byteLevel = all.find((p) => p.type === 'ByteLevel');
  if (!byteLevel) return undefined;
  const split = all.find((p) => p.type === 'Split' && p.pattern?.Regex !== undefined);
  if (split) {
    if (split.behavior !== 'Isolated') {
      throw new Error(`tokenizer: Split behavior '${split.behavior}' unsupported`);
    }
    return splitRegex(split.pattern!.Regex!);
  }
  return splitRegex(byteLevel.use_regex === false ? '[\\s\\S]+' : GPT2_SPLIT);
}

/** Binary min-heap of merge candidates keyed by (rank, position). */
class MergeHeap {
  private rank: number[] = [];
  private pos: number[] = [];
  get size(): number {
    return this.rank.length;
  }
  private less(i: number, j: number): boolean {
    const a = this.rank[i]!;
    const b = this.rank[j]!;
    return a < b || (a === b && this.pos[i]! < this.pos[j]!);
  }
  private swap(i: number, j: number): void {
    [this.rank[i], this.rank[j]] = [this.rank[j]!, this.rank[i]!];
    [this.pos[i], this.pos[j]] = [this.pos[j]!, this.pos[i]!];
  }
  push(rank: number, pos: number): void {
    this.rank.push(rank);
    this.pos.push(pos);
    let i = this.rank.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!this.less(i, p)) break;
      this.swap(i, p);
      i = p;
    }
  }
  pop(): [number, number] {
    const top: [number, number] = [this.rank[0]!, this.pos[0]!];
    const lastR = this.rank.pop()!;
    const lastP = this.pos.pop()!;
    if (this.rank.length > 0) {
      this.rank[0] = lastR;
      this.pos[0] = lastP;
      let i = 0;
      const n = this.rank.length;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < n && this.less(l, m)) m = l;
        if (r < n && this.less(r, m)) m = r;
        if (m === i) break;
        this.swap(i, m);
        i = m;
      }
    }
    return top;
  }
}

/** Builds the Gemma 3 tokenizer from a parsed tokenizer.json. */
export function createGemma3Tokenizer(json: unknown): Gemma3Tokenizer {
  return createTextTokenizer(json, { chatFormat: 'gemma' });
}

/** Builds a tokenizer from a parsed Hugging Face tokenizer.json. The chat
 *  template follows from the vocab (`<start_of_turn>` → gemma,
 *  `<|im_start|>` → qwen3) unless `chatFormat` names it. */
export function createTextTokenizer(
  json: unknown,
  opts: { chatFormat?: ChatFormat } = {},
): TextTokenizer {
  const bad = (why: string) => new Error(`tokenizer: ${why}`);
  const hf = json as Partial<HfTokenizerJson>;
  if (!hf || typeof hf !== 'object' || hf.model?.type !== 'BPE') {
    throw bad('expected a BPE tokenizer.json');
  }
  const model = hf.model;
  const vocab = new Map<string, number>(Object.entries(model.vocab));
  const added = hf.added_tokens ?? [];
  for (const t of added) vocab.set(t.content, t.id);
  let size = 0;
  for (const id of vocab.values()) size = Math.max(size, id + 1);
  const idToTok: string[] = new Array<string>(size).fill('');
  for (const [tok, id] of vocab) idToTok[id] = tok;

  const special = new Uint8Array(size);
  const isAdded = new Uint8Array(size);
  for (const t of added) {
    isAdded[t.id] = 1;
    if (t.special) special[t.id] = 1;
  }

  const split = byteLevelSplit(hf.pre_tokenizer);
  const norm = hf.normalizer;
  const nfc =
    norm?.type === 'NFC' ||
    (norm?.type === 'Sequence' && !!norm.normalizers?.some((n) => n.type === 'NFC'));
  const b2u = byteToUnicode();
  const u2b = new Map<string, number>(b2u.map((c, b) => [c, b]));

  // Byte tokens: id → byte value, and byte → id for the fallback.
  const byteOf = new Int16Array(size).fill(-1);
  const byteId = new Int32Array(256).fill(-1);
  if (!split) {
    for (let b = 0; b < 256; b++) {
      const tok = `<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`;
      const id = vocab.get(tok);
      if (id !== undefined) {
        byteId[b] = id;
        byteOf[id] = b;
      }
    }
    for (let id = 0; id < size; id++) {
      if (byteOf[id] === -1) {
        const m = BYTE_TOKEN.exec(idToTok[id]!);
        if (m) byteOf[id] = parseInt(m[1]!, 16);
      }
    }
  }
  const unkId = model.unk_token ? vocab.get(model.unk_token) : undefined;

  // Merge table keyed by the (left id, right id) pair. ids < 2^19, so the
  // key stays a safe integer.
  const pairKey = (a: number, b: number) => a * 524288 + b;
  const mergeRank = new Map<number, number>();
  const mergeOut = new Int32Array(model.merges.length);
  model.merges.forEach((m, rank) => {
    let a: string;
    let b: string;
    if (Array.isArray(m)) [a, b] = m;
    else {
      const cut = m.indexOf(' ', 1);
      a = m.slice(0, cut);
      b = m.slice(cut + 1);
    }
    const ia = vocab.get(a);
    const ib = vocab.get(b);
    const out = vocab.get(a + b);
    if (ia === undefined || ib === undefined || out === undefined) return;
    const key = pairKey(ia, ib);
    if (!mergeRank.has(key)) {
      mergeRank.set(key, rank);
      mergeOut[rank] = out;
    }
  });

  // Added tokens split the input before BPE, longest match first.
  const addedByFirst = new Map<string, string[]>();
  for (const t of added) {
    const first = t.content[0];
    if (first === undefined) continue;
    let list = addedByFirst.get(first);
    if (!list) addedByFirst.set(first, (list = []));
    list.push(t.content);
  }
  for (const list of addedByFirst.values()) list.sort((x, y) => y.length - x.length);

  const utf8 = new TextEncoder();

  const bpe = (text: string, out: number[]): void => {
    const ids: number[] = [];
    for (const ch of text) {
      const id = vocab.get(ch);
      if (id !== undefined) {
        ids.push(id);
        continue;
      }
      if (model.byte_fallback) {
        for (const byte of utf8.encode(ch)) {
          const bid = byteId[byte]!;
          if (bid < 0) throw bad(`no byte token for 0x${byte.toString(16)}`);
          ids.push(bid);
        }
      } else if (unkId !== undefined) {
        ids.push(unkId);
      } else {
        throw bad(`character '${ch}' not in vocab`);
      }
    }
    const n = ids.length;
    if (n < 2) {
      out.push(...ids);
      return;
    }
    const prev = new Int32Array(n);
    const next = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      prev[i] = i - 1;
      next[i] = i + 1 < n ? i + 1 : -1;
    }
    const alive = new Uint8Array(n).fill(1);
    const heap = new MergeHeap();
    // The heap stores (rank, left position); a stale entry is detected by
    // re-reading the pair's current rank.
    const rankAt = (i: number): number => {
      const j = next[i]!;
      if (j < 0) return -1;
      return mergeRank.get(pairKey(ids[i]!, ids[j]!)) ?? -1;
    };
    for (let i = 0; i + 1 < n; i++) {
      const r = rankAt(i);
      if (r >= 0) heap.push(r, i);
    }
    while (heap.size > 0) {
      const [r, i] = heap.pop();
      if (!alive[i] || rankAt(i) !== r) continue;
      const j = next[i]!;
      ids[i] = mergeOut[r]!;
      alive[j] = 0;
      const k = next[j]!;
      next[i] = k;
      if (k >= 0) prev[k] = i;
      const p = prev[i]!;
      if (p >= 0) {
        const rp = rankAt(p);
        if (rp >= 0) heap.push(rp, p);
      }
      const ri = rankAt(i);
      if (ri >= 0) heap.push(ri, i);
    }
    for (let i = 0; i >= 0; i = next[i]!) out.push(ids[i]!);
  };

  // Byte-level pieces repeat a lot (words, indentation), so their BPE is
  // memoized.
  const pieceCache = new Map<string, number[]>();
  const bytePiece = (piece: string, out: number[]): void => {
    let ids = pieceCache.get(piece);
    if (!ids) {
      let mapped = '';
      for (const byte of utf8.encode(piece)) mapped += b2u[byte]!;
      const whole = model.ignore_merges ? vocab.get(mapped) : undefined;
      if (whole !== undefined) ids = [whole];
      else bpe(mapped, (ids = []));
      if (pieceCache.size < 65536) pieceCache.set(piece, ids);
    }
    for (const id of ids) out.push(id);
  };
  /** Encodes a run of text that holds no added token. */
  const encodeRun = (run: string, out: number[]): void => {
    if (!split) {
      bpe(run.replaceAll(' ', SPACE), out);
      return;
    }
    if (nfc) run = run.normalize('NFC');
    let last = 0;
    for (const m of run.matchAll(split)) {
      if (m.index > last) bytePiece(run.slice(last, m.index), out);
      if (m[0].length > 0) bytePiece(m[0], out);
      last = m.index + m[0].length;
    }
    if (last < run.length) bytePiece(run.slice(last), out);
  };

  const encode = (text: string, opts: { addBos?: boolean; parseSpecial?: boolean } = {}) => {
    const out: number[] = [];
    if (opts.addBos && bosId !== undefined) out.push(bosId);
    const parseSpecial = opts.parseSpecial ?? true;
    let start = 0;
    const flush = (end: number) => {
      if (end > start) encodeRun(text.slice(start, end), out);
    };
    if (parseSpecial) {
      for (let i = 0; i < text.length;) {
        const cands = addedByFirst.get(text[i]!);
        let hit: string | undefined;
        if (cands)
          for (const c of cands)
            if (text.startsWith(c, i)) {
              hit = c;
              break;
            }
        if (hit === undefined) {
          i++;
          continue;
        }
        flush(i);
        out.push(vocab.get(hit)!);
        i += hit.length;
        start = i;
      }
    }
    flush(text.length);
    return out;
  };

  const tokenBytes = (id: number): Uint8Array => {
    const tok = idToTok[id];
    if (tok === undefined) throw bad(`id ${id} outside vocab`);
    if (split) {
      if (isAdded[id]) return utf8.encode(tok);
      const bytes = new Uint8Array(tok.length);
      let n = 0;
      for (const ch of tok) {
        const b = u2b.get(ch);
        if (b === undefined) return utf8.encode(tok);
        bytes[n++] = b;
      }
      return bytes.subarray(0, n);
    }
    const b = byteOf[id]!;
    if (b >= 0) return Uint8Array.of(b);
    return utf8.encode(tok.replaceAll(SPACE, ' '));
  };

  const createStreamDecoder = (opts: { skipSpecial?: boolean } = {}) => {
    const skip = opts.skipSpecial ?? true;
    const dec = new TextDecoder('utf-8');
    return {
      push(id: number): string {
        if (skip && special[id]) return '';
        return dec.decode(tokenBytes(id), { stream: true });
      },
      flush(): string {
        return dec.decode();
      },
    };
  };

  const decode = (ids: readonly number[], opts: { skipSpecial?: boolean } = {}) => {
    const s = createStreamDecoder(opts);
    let text = '';
    for (const id of ids) text += s.push(id);
    return text + s.flush();
  };

  const need = (tok: string): number => {
    const id = vocab.get(tok);
    if (id === undefined) throw bad(`vocab has no ${tok}`);
    return id;
  };
  let chatFormat = opts.chatFormat;
  if (!chatFormat) {
    if (vocab.has('<start_of_turn>')) chatFormat = 'gemma';
    else if (vocab.has('<|im_start|>')) chatFormat = 'qwen3';
    else throw bad('unknown chat format (no <start_of_turn> or <|im_start|>)');
  }
  const gemma = chatFormat === 'gemma';
  const bosId = gemma ? need('<bos>') : undefined;
  const eosId = need(gemma ? '<eos>' : '<|endoftext|>');
  const endOfTurnId = need(gemma ? '<end_of_turn>' : '<|im_end|>');
  const format = gemma ? formatGemmaChat : formatQwen3Chat;

  return {
    chatFormat,
    vocabSize: size,
    bosId,
    eosId,
    endOfTurnId,
    stopIds: Object.freeze([endOfTurnId, eosId]),
    encode,
    decode,
    createStreamDecoder,
    tokenToId: (t) => vocab.get(t),
    idToToken: (id) => idToTok[id],
    isSpecial: (id) => special[id] === 1,
    applyChatTemplate: (messages, o) => format(messages, o),
  };
}

/** The Gemma 3 chat template (HF tokenizer_config.json), with lenient role
 *  handling: a system message becomes the first user turn's prefix,
 *  consecutive same-role turns are merged, and any role other than
 *  assistant/model is treated as user. */
export function formatGemmaChat(
  messages: readonly ChatMessage[],
  opts: ChatTemplateOptions = {},
): string {
  const system: string[] = [];
  const turns: { role: 'user' | 'model'; content: string }[] = [];
  for (const m of messages) {
    if (m.role === 'system') {
      if (turns.length === 0) system.push(m.content.trim());
      else turns.push({ role: 'user', content: m.content.trim() });
      continue;
    }
    const role = m.role === 'assistant' || m.role === 'model' ? 'model' : 'user';
    const last = turns[turns.length - 1];
    if (last && last.role === role) last.content += `\n\n${m.content.trim()}`;
    else turns.push({ role, content: m.content.trim() });
  }
  if (turns.length === 0 || turns[0]!.role !== 'user') turns.unshift({ role: 'user', content: '' });
  if (system.length > 0) {
    const first = turns[0]!;
    first.content = `${system.join('\n\n')}\n\n${first.content}`;
  }
  let out = '<bos>';
  for (const t of turns) out += `<start_of_turn>${t.role}\n${t.content}<end_of_turn>\n`;
  if (opts.addGenerationPrompt ?? true) out += '<start_of_turn>model\n';
  return out;
}

/** Python's json.dumps defaults (', ' and ': ' separators, non-ASCII kept),
 *  which the HF template's `tojson` produces. */
function pyJson(v: unknown): string {
  if (v === null || v === undefined) return 'null';
  if (Array.isArray(v)) return `[${v.map(pyJson).join(', ')}]`;
  if (typeof v === 'object') {
    const entries = Object.entries(v as Record<string, unknown>).filter(([, x]) => x !== undefined);
    return `{${entries.map(([k, x]) => `${JSON.stringify(k)}: ${pyJson(x)}`).join(', ')}}`;
  }
  return JSON.stringify(v);
}

const isToolResponse = (s: string) =>
  s.startsWith('<tool_response>') && s.endsWith('</tool_response>');

/** The Qwen 3 chat template (HF tokenizer_config.json) for string contents:
 *  ChatML turns; a leading system message, with the tool list when `tools`
 *  is given; assistant turns before the last user query stripped of their
 *  `<think>` part; consecutive `tool` messages grouped into one user turn of
 *  `<tool_response>` blocks; and, unless `enableThinking`, an empty think
 *  block after the generation prompt. Role `model` is read as assistant. */
export function formatQwen3Chat(
  messages: readonly ChatMessage[],
  opts: ChatTemplateOptions = {},
): string {
  const msgs = messages.map((m) => (m.role === 'model' ? { ...m, role: 'assistant' } : m));
  const first = msgs[0];
  const leadingSystem = first?.role === 'system';
  let out = '';
  if (opts.tools && opts.tools.length > 0) {
    out += '<|im_start|>system\n';
    if (leadingSystem) out += `${first.content}\n\n`;
    out +=
      '# Tools\n\nYou may call one or more functions to assist with the user query.\n\n' +
      'You are provided with function signatures within <tools></tools> XML tags:\n<tools>';
    for (const t of opts.tools) out += `\n${pyJson(t)}`;
    out +=
      '\n</tools>\n\nFor each function call, return a json object with function name and ' +
      'arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n' +
      '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n';
  } else if (leadingSystem) {
    out += `<|im_start|>system\n${first.content}<|im_end|>\n`;
  }
  let lastQuery = msgs.length - 1;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const m = msgs[i]!;
    if (m.role === 'user' && !isToolResponse(m.content)) {
      lastQuery = i;
      break;
    }
  }
  msgs.forEach((m, i) => {
    let content = m.content;
    if (m.role === 'user' || (m.role === 'system' && i > 0)) {
      out += `<|im_start|>${m.role}\n${content}<|im_end|>\n`;
    } else if (m.role === 'assistant') {
      let reasoning = '';
      if (content.includes('</think>')) {
        const parts = content.split('</think>');
        reasoning = parts[0]!.replace(/\n+$/, '').split('<think>').pop()!.replace(/^\n+/, '');
        content = parts[parts.length - 1]!.replace(/^\n+/, '');
      }
      if (i > lastQuery && (i === msgs.length - 1 || reasoning)) {
        const r = reasoning.replace(/^\n+|\n+$/g, '');
        out += `<|im_start|>assistant\n<think>\n${r}\n</think>\n\n${content.replace(/^\n+/, '')}`;
      } else {
        out += `<|im_start|>assistant\n${content}`;
      }
      out += '<|im_end|>\n';
    } else if (m.role === 'tool') {
      if (i === 0 || msgs[i - 1]!.role !== 'tool') out += '<|im_start|>user';
      out += `\n<tool_response>\n${content}\n</tool_response>`;
      if (i === msgs.length - 1 || msgs[i + 1]!.role !== 'tool') out += '<|im_end|>\n';
    }
  });
  if (opts.addGenerationPrompt ?? true) {
    out += '<|im_start|>assistant\n';
    if (!opts.enableThinking) out += '<think>\n\n</think>\n\n';
  }
  return out;
}
