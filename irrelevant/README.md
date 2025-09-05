# Winery RAG — MVP (Semantic Only, Hebrew)
A minimal Retrieval‑Augmented Generation setup for your winery quotations in Hebrew. **No feature extraction** — just chunk+embed+retrieve and let the LLM draft the quotation.

## Layout
```
winery-rag-mvp/
  data/
    raw_docx/           # put your .docx quotations here
    processed/
      chunks.jsonl      # text chunks for retrieval
      index.faiss       # FAISS vector index
      meta.jsonl        # per-chunk metadata
  scripts/
    00_ingest_docx_mvp.py
    01_build_embeddings_mvp.py
    02_query_mvp.py
    prompt_mvp_he.txt
  requirements.txt
  .env.example
```
## Quickstart
1) Create venv & install:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
2) Drop your `.docx` quotations into `data/raw_docx/`  
3) Ingest → chunks:
```bash
python scripts/00_ingest_docx_mvp.py --input data/raw_docx --out data/processed
```
4) Build embeddings (choose one):
```bash
# OpenAI (recommended, Hebrew-capable)
export OPENAI_API_KEY=...   # Windows: set OPENAI_API_KEY=...
python scripts/01_build_embeddings_mvp.py --jsonl data/processed/chunks.jsonl --out data/processed --provider openai --model text-embedding-3-large

# OR local (offline)
python scripts/01_build_embeddings_mvp.py --jsonl data/processed/chunks.jsonl --out data/processed --provider local --model sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```
5) Query & draft a quote:
```bash
python scripts/02_query_mvp.py   --index data/processed/index.faiss --meta data/processed/meta.jsonl   --request "אירוח עבור 50 משתתפים באביב, פרגולה + אולם, משך 5 שעות, בופה, יין לפי בקבוק"   --topk 8 --gen_model gpt-4o
```
The script retrieves the most similar chunks and calls the LLM with a compact Hebrew **quotation-writer** prompt.

### Notes
- This MVP ignores structure. If you later want accurate pricing logic (per-person, minimums, VAT), we can layer in light features without changing the index format.
- `chunks.jsonl` stores one JSON object per chunk with: `id`, `doc_id`, `text`, and `source_path`.
