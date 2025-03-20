# retrieval/document_store.py
import json
from typing import Dict

class DocumentStore:
    def __init__(self, document_path: str):
        self.documents: Dict[int, str] = {}
        with open(document_path, "r", encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                assert "document_id" in doc and "document_text" in doc, "Invalid document format"
                self.documents[doc["document_id"]] = doc["document_text"]
    
    def get_document(self, doc_id: int) -> str:
        return self.documents.get(doc_id, "")