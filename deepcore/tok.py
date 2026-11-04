"""Byte-level BPE tokenizer straight from GGUF vocab (Qwen2/3 pre, ChatML template).

pip install regex (for \\p{L}/\\p{N} classes). Falls back to a coarse splitter if absent.
"""
import struct

try:
    import regex as _rx

    def _compile(pat):
        return _rx.compile(pat)
except Exception:  # pragma: no cover - coarse fallback
    _rx = None

    def _compile(pat):
        import re
        return re.compile(pat.replace(r"\p{L}", r"a-zA-Z").replace(r"\p{N}", r"0-9"))

QWEN2_PRE = r"(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs])), {chr(c): b for b, c in zip(bs, cs)}


_THINK = None


def clean_reply(text):
    """Hide Qwen3 thinking blocks from streamed/final output."""
    import re
    global _THINK
    if _THINK is None:
        _THINK = re.compile(r"<think>.*?(?:</think>|$)", re.S)
    return _THINK.sub("", text or "").strip()


def apply_no_think(messages, model_key=None):
    """Qwen3-1.7B is a thinking model; /no_think keeps demo replies short."""
    if not model_key or "1.7b" not in str(model_key):
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            m["content"] = (m.get("content") or "") + " /no_think"
            break
    return out


class BPETokenizer:
    def __init__(s, tokens, token_types, merges, pre=QWEN2_PRE):
        s.vocab = list(tokens)
        s.ty = list(token_types)
        s.str2id = {t: i for i, t in enumerate(s.vocab)}
        s.b2u, s.u2b = _bytes_to_unicode()
        s.ranks = {}
        for r, m in enumerate(merges):
            parts = m.split(" ")
            if len(parts) == 2:
                s.ranks[(parts[0], parts[1])] = r
        s.rx = _compile(pre)
        s.special = {t: i for i, (t, ty) in enumerate(zip(s.vocab, s.ty)) if ty in (3, 4) and t.startswith("<|")}
        # some GGUFs type ChatML controls as NORMAL; bind known names explicitly
        for n in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
            if n in s.str2id:
                s.special.setdefault(n, s.str2id[n])

    @classmethod
    def from_meta(s, meta):
        toks = meta["tokenizer.ggml.tokens"]
        tys = meta["tokenizer.ggml.token_type"]
        merg = meta.get("tokenizer.ggml.merges", [])
        if not isinstance(merg, list):
            merg = []
        return s(toks, tys, merg, QWEN2_PRE)

    def _bpe(s, piece):
        u = "".join(s.b2u[b] for b in piece.encode("utf-8"))
        word = list(u)
        while len(word) > 1:
            best, bi = None, None
            for i in range(len(word) - 1):
                r = s.ranks.get((word[i], word[i + 1]))
                if r is not None and (best is None or r < best):
                    best, bi = r, i
            if bi is None:
                break
            word[bi:bi + 2] = [word[bi] + word[bi + 1]]
        return [s.str2id[w] for w in word if w in s.str2id]

    def encode(s, text, add_specials=True):
        if not add_specials:
            return s.encode_plain(text)
        ids = []
        i = 0
        n = len(text)
        # find special boundaries, then pre-tokenize each plain run as one piece
        while i < n:
            j, hit = n, None
            for sp in s.special:
                p = text.find(sp, i)
                if p != -1 and p < j:
                    j, hit = p, sp
            run = text[i:j]
            if run:
                ids.extend(s.encode_plain(run))
            if hit:
                ids.append(s.special[hit])
                i = j + len(hit)
            else:
                i = j
        return ids

    def encode_plain(s, text):
        if s.rx is None:
            pieces = [text]
        else:
            pieces = [m.group(0) for m in s.rx.finditer(text)]
        out = []
        for p in pieces:
            out.extend(s._bpe(p))
        return out

    def decode(s, ids):
        buf = []
        for i in ids:
            i = int(i)
            if 0 <= i < len(s.vocab):
                buf.append(s.vocab[i])
        u = "".join(buf)
        return bytes(s.u2b[ch] for ch in u if ch in s.u2b).decode("utf-8", "replace")

    # ChatML (Qwen3 instruct-2507, non-thinking)
    IM_START, IM_END = "<|im_start|>", "<|im_end|>"

    def chat_ids(s, messages, add_gen=True):
        out = []
        out += s.encode(s.IM_START + "system\n")
        sysmsg = [m["content"] for m in messages if m["role"] == "system"]
        body = [m for m in messages if m["role"] != "system"]
        out += s.encode((sysmsg[0] if sysmsg else "You are a helpful assistant.") + s.IM_END + "\n")
        for m in body:
            out += s.encode(s.IM_START + m["role"] + "\n" + m["content"] + s.IM_END + "\n")
        if add_gen:
            out += s.encode(s.IM_START + "assistant\n")
        return out