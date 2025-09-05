# -*- coding: utf-8 -*-
import argparse, os, json, numpy as np
import faiss

def embed_openai(texts, model):
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    out = []
    B = 256
    for i in range(0, len(texts), B):
        batch = texts[i:i+B]
        resp = client.embeddings.create(model=model, input=batch)
        for d in resp.data:
            out.append(np.array(d.embedding, dtype=np.float32))
    return np.vstack(out)

def embed_local(texts, model):
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(model)
    emb = st.encode(texts, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    return emb.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--provider', default='openai', choices=['openai','local'])
    ap.add_argument('--model', default='text-embedding-3-large')
    args = ap.parse_args()

    texts, metas = [], []
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec['text'])
            metas.append(rec)

    if args.provider == 'openai':
        X = embed_openai(texts, args.model)
    else:
        X = embed_local(texts, args.model)

    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    index = faiss.IndexFlatIP(Xn.shape[1])
    index.add(Xn)

    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, 'index.faiss'))
    with open(os.path.join(args.out, 'meta.jsonl'), 'w', encoding='utf-8') as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')
    print('Saved index.faiss and meta.jsonl')

if __name__ == '__main__':
    main()
