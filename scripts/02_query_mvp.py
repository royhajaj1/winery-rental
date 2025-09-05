# -*- coding: utf-8 -*-
import argparse, os, json, numpy as np, faiss

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

def load_meta(meta_path):
    metas = []
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            metas.append(json.loads(line))
    return metas

def call_llm_openai(context_items, request_text, model='gpt-4o'):
    from openai import OpenAI
    from dotenv import load_dotenv; load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_mvp_he.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    ctx = []
    for i, m in enumerate(context_items, 1):
        ctx.append(f'-- מסמך עזר #{i} --')
        ctx.append('טקסט: ' + m.get('text','')[:1800])
        user_input = f"""בקשה חדשה:
        {request_text}

        מסמכי עזר (חופשיים):
        {os.linesep.join(ctx)}
        """
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

def call_llm_ollama(context_items, request_text, model='llama3.1:8b-instruct'):
    import requests
    prompt_path = os.path.join(os.path.dirname(__file__), 'prompt_mvp_he.txt')
    with open(prompt_path, 'r', encoding='utf-8') as f:
        system_prompt = f.read()
    ctx = []
    for i, m in enumerate(context_items, 1):
        ctx.append(f'-- מסמך עזר #{i} --')
        ctx.append('טקסט: ' + m.get('text','')[:1800])
        user_input = f"""בקשה חדשה:
        {request_text}

        מסמכי עזר (חופשיים):
        {os.linesep.join(ctx)}
        """
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
    ap.add_argument('--print_context', action='store_true', help='Print retrieved snippets and exit')
    ap.add_argument('--out_file', help='Write output to this UTF-8 file')

    args = ap.parse_args()

    request_value = args.request
    if args.from_file:
        req_path = os.path.join(os.path.dirname(__file__), 'new_request.txt')
        if os.path.exists(req_path):
            with open(req_path, 'r', encoding='utf-8') as f:
                request_value = f.read().strip()
        else:
            print(f"new_request.txt not found at {req_path}")
            return

    qvec = embed_query(request_value, provider=args.embed_provider, model=args.embed_model)

    index = faiss.read_index(args.index)
    D, I = index.search(qvec.reshape(1,-1).astype(np.float32), max(args.topk, 12))
    metas = load_meta(args.meta)

    hits = [(float(d), metas[i]) for d, i in zip(D[0], I[0]) if i != -1]
    hits.sort(key=lambda x: x[0], reverse=True)

    chosen, used_docs = [], set()
    for _, m in hits:
        if m['doc_id'] not in used_docs or len(chosen) < 4:
            chosen.append(m)
            used_docs.add(m['doc_id'])
        if len(chosen) >= args.topk:
            break

    if args.print_context:
        print("=== Retrieved Context ===")
        for i, m in enumerate(chosen, 1):
            print_str = f"\n--- #{i} from {m.get('doc_id')} ---\n"
            print_str += m.get('text','')[:1500]
            print(print_str)
            if args.out_file:
                write_out(print_str, args.out_file)
        return

    if args.gen_provider == 'ollama':
        out = call_llm_ollama(chosen, request_value, model=args.gen_model)
    else:
        out = call_llm_openai(chosen, request_value, model=args.gen_model)

    print(out)

if __name__ == '__main__':
    main()
