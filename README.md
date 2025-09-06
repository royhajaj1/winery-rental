Process new request:
python scripts/02_query_mvp2.py --index data/processed/index.faiss --meta data/processed/meta.jsonl --request "placeholder" --embed_provider local --embed_model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --print_context --out_file .\context.txt --from_file

Fill template after receiving answer.json:
python scripts/inject_json_to_docx.py answer1.json template.docx 