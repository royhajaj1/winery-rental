# -*- coding: utf-8 -*-
import argparse, os, json, numpy as np, faiss, re

# ==== NEW: token helpers ======================================================
try:
    import tiktoken
    _enc = tiktoken.get_encoding("o200k_base")
except Exception:
    _enc = None

def _token_len(s: str) -> int:
    if not s: return 0
    if _enc:
        try:
            return len(_enc.encode(s))
        except Exception:
            pass
    # fallback heuristic (~4 chars per token works OK for He/En mix)
    return max(1, len(s) // 4)

def shrink_long_chunk(t: str, per_chunk_cap_tokens: int = 400) -> str:
    """Keep salient sentences if a single chunk is way too long."""
    if _token_len(t) <= per_chunk_cap_tokens:
        return t
    sents = re.split(r'(?<=[.!?])\s+', t)
    keep = []
    i, j = 0, len(sents) - 1
    while i <= j and _token_len(" ".join(keep)) < per_chunk_cap_tokens:
        if i == 0:
            keep.append(sents[i]); i += 1
        elif j == len(sents) - 1:
            keep.append(sents[j]); j -= 1
        else:
            keep.append(sents[i]); i += 1
            if i <= j: keep.append(sents[j]); j -= 1
    return " ".join(keep)

def pack_context(texts, max_tokens, sep="\n\n---\n\n"):
    """Greedily pack chunk texts into a token budget."""
    out, used = [], 0
    for t in texts:
        t = (t or "").strip()
        if not t: continue
        cost = _token_len(t) + (_token_len(sep) if out else 0)
        if used + cost > max_tokens: break
        out.append(t); used += cost
    return sep.join(out), used
# ==============================================================================

def embed_query(text, provider='openai', model='text-embedding-3-large'):
    if provider == 'openai':
        from openai import OpenAI
        from dotenv import load_dotenv; load_dotenv()
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        resp = client.embeddings.create(model=model, input=[text])
        vec = np.array(resp.data[0].embedding, dtype=np.float32)
    else:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model)
        vec = st.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-12)
    return vec

# ==== CHANGED: return dict keyed by 'id' ======================================
def load_meta(meta_path):
    """
    Returns:
      metas_by_id: dict[int, obj]   # only for numeric ids
      metas_by_pos: list[obj]       # positional fallback (line order)
    """
    metas_by_id = {}
    metas_by_pos = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for pos, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            metas_by_pos.append(obj)
            _id = obj.get("id", None)
            # Only store in metas_by_id when it's safely numeric
            if _id is not None:
                try:
                    metas_by_id[int(_id)] = obj
                except (TypeError, ValueError):
                    # non-numeric id (e.g., a string key) -> skip numeric map
                    pass
    return metas_by_id, metas_by_pos

# ==== NEW: build a doc key safely =============================================
def _doc_key(m):
    return m.get('doc_id') or m.get('source') or m.get('title') or "unknown"

# ==== CHANGED: token-aware context in OpenAI call =============================
def call_llm_openai(context_items, request_text, model='gpt-4o'):
    from openai import OpenAI
    from dotenv import load_dotenv; load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_mvp_he.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    # assemble raw texts (optionally shrink long chunks)
    raw_texts = []
    ctx_lines = []
    for i, m in enumerate(context_items, 1):
        ctx_lines.append(f'-- מסמך עזר #{i} --')
        raw_texts.append(m.get('text', ''))

    # budgets (env overridable)
    CTX_BUDGET    = int(os.getenv("CTX_BUDGET", "12000"))  # tokens for context packing
    ANSWER_BUDGET = int(os.getenv("ANSWER_BUDGET", "800")) # leave room for the model to answer
    SAFETY_BUDGET = int(os.getenv("SAFETY_BUDGET", "300")) # buffer for headers/formatting

    # Pack only the chunk texts; we’ll prepend headings to each piece after packing
    shrunk = [shrink_long_chunk(t) for t in raw_texts]
    # Subtract system + question + some buffer from the context allowance
    budget = max(512, CTX_BUDGET - ANSWER_BUDGET - SAFETY_BUDGET - _token_len(system_prompt) - _token_len(request_text))
    packed_body, used = pack_context(shrunk, budget)

    # Reconstruct with numbered headings aligned to what was kept
    # (Simplest: just provide one combined section of packed text)
    # If you want headings per chunk, keep indices alongside during packing.
    context_block = packed_body

    user_input = (
        f"בקשה חדשה:\n{request_text}\n\n"
        f"מסמכי עזר (חופשיים):\n{context_block}\n"
    )

    try:
        resp = client.responses.create(
            model=model,
            input=[
                {"role":"system","content":system_prompt},
                {"role":"user","content":user_input}
            ]
        )
        content = getattr(resp, 'output_text', None)
        if content: return content
    except Exception:
        pass
    cc = client.chat.completions.create(
        model=model,
        messages=[
            {"role":"system","content":system_prompt},
            {"role":"user","content":user_input}
        ]
    )
    return cc.choices[0].message.content

# ==== CHANGED: token-aware context in Ollama call =============================
def call_llm_ollama(context_items, request_text, model='llama3.1:8b-instruct'):
    import requests
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_mvp_he.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    raw_texts = [m.get('text', '') for m in context_items]
    shrunk = [shrink_long_chunk(t) for t in raw_texts]
    CTX_BUDGET    = int(os.getenv("CTX_BUDGET", "6000"))
    ANSWER_BUDGET = int(os.getenv("ANSWER_BUDGET", "600"))
    SAFETY_BUDGET = int(os.getenv("SAFETY_BUDGET", "200"))
    budget = max(512, CTX_BUDGET - ANSWER_BUDGET - SAFETY_BUDGET - _token_len(system_prompt) - _token_len(request_text))
    context_block, _ = pack_context(shrunk, budget)

    user_input = (
        f"בקשה חדשה:\n{request_text}\n\n"
        f"מסמכי עזר (חופשיים):\n{context_block}\n"
    )

    payload = {
        "model": model,
        "prompt": f"{system_prompt}\n\n{user_input}",
        "stream": False
    }
    r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response","")

def write_out(text, path):
    with open(path, 'a+', encoding='utf-8-sig') as f:  # UTF-8 with BOM
        f.write(text if isinstance(text, str) else str(text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', required=True)
    ap.add_argument('--meta', required=True)
    ap.add_argument('--request', required=True, help='Free-text Hebrew request for the new quotation')
    ap.add_argument('--from_file', action='store_true', help='Read request value from new_request.txt')
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--embed_provider', default='openai')
    ap.add_argument('--embed_model', default='text-embedding-3-large')
    ap.add_argument('--gen_provider', default='openai', choices=['openai','ollama'])
    ap.add_argument('--gen_model', default='gpt-4o')
    ap.add_argument('--print_context', action='store_true', help='Print retrieved (token-packed) snippets and exit')
    ap.add_argument('--out_file', help='Write output to this UTF-8 file')
    args = ap.parse_args()

    request_value = args.request
    if args.from_file:
        parent_dir = os.path.dirname(os.path.dirname(__file__))
        req_path = os.path.join(parent_dir, 'new_request.txt')
        if os.path.exists(req_path):
            with open(req_path, 'r', encoding='utf-8') as f:
                request_value = f.read().strip()
        else:
            print(f"new_request.txt not found at {req_path}")
            return

    qvec = embed_query(request_value, provider=args.embed_provider, model=args.embed_model)

    index = faiss.read_index(args.index)
    D, I = index.search(qvec.reshape(1,-1).astype(np.float32), max(args.topk, 12))

    metas_by_id, metas_list = load_meta(args.meta)

    # Build hits with correct meta lookup:
    hits = []
    for d, idx_id in zip(D[0], I[0]):
        if idx_id == -1: continue
        m = metas_by_id.get(int(idx_id))
        if m is None:
            # fallback: if id mapping is missing, try list index
            if 0 <= idx_id < len(metas_list):
                m = metas_list[int(idx_id)]
            else:
                continue
        hits.append((float(d), m))
    hits.sort(key=lambda x: x[0], reverse=True)

    # Simple doc diversity (tolerate missing keys)
    chosen, used_docs = [], set()
    for _, m in hits:
        key = _doc_key(m)
        if key not in used_docs or len(chosen) < 4:
            chosen.append(m)
            used_docs.add(key)
        if len(chosen) >= args.topk:
            break

    # ==== CHANGED: print_context uses token-packed text, not char slice =======
    if args.print_context:
        # pack just like in the LLM calls, so you preview exactly what’s sent
        CTX_BUDGET    = int(os.getenv("CTX_BUDGET", "12000"))
        ANSWER_BUDGET = int(os.getenv("ANSWER_BUDGET", "800"))
        SAFETY_BUDGET = int(os.getenv("SAFETY_BUDGET", "300"))
        budget = max(512, CTX_BUDGET - ANSWER_BUDGET - SAFETY_BUDGET - _token_len(request_value))
        body, used = pack_context([shrink_long_chunk(m.get('text','')) for m in chosen], budget)
        print("=== Retrieved Context (token-packed) ===")
        print(body)
        if args.out_file:
            write_out(body, args.out_file)
        return

    if args.gen_provider == 'ollama':
        out = call_llm_ollama(chosen, request_value, model=args.gen_model)
    else:
        out = call_llm_openai(chosen, request_value, model=args.gen_model)

    print(out)

if __name__ == '__main__':
    main()
