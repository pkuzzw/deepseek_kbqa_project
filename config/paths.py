from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATHS = {
    "documents": BASE_DIR / "data/documents.jsonl",
    "train": BASE_DIR / "data/train.jsonl",
    "val": BASE_DIR / "data/val.jsonl",
    "val_result_bm25": BASE_DIR / "data/val_predict_bm25_full.jsonl",
    "val_result_glove": BASE_DIR / "data/val_predict_glove_full.jsonl",
    "test": BASE_DIR / "data/test.jsonl"
}

MODEL_PATHS = {
    "glove": "glove.6B.300d.txt",
    "dpr_ctx": "facebook/dpr-ctx_encoder-single-nq-base",
    "dpr_q": "facebook/dpr-question_encoder-single-nq-base"
}

API_CONFIG = {
    "qwen_endpoint": "https://api.siliconflow.com/v1/chat/completions",
    "api_key": "your-api-key-here"
}