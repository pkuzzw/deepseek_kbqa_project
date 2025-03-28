# retrieval/document_store.py
import json
from typing import Dict

class DocumentStore:
    """
    A class to represent a document store that loads documents from a file
    and allows retrieval of documents by their ID.
    """
    def __init__(self, document_path: str):
        """
        Initializes the DocumentStore by loading documents from the specified file.

        Args:
            document_path (str): Path to the file containing documents in JSONL format.
                                 Each line should be a JSON object with "document_id" and "document_text".
        """
        self.documents: Dict[int, str] = {}
        with open(document_path, "r", encoding='utf-8') as f:
            for line in f:
                doc = json.loads(line)
                # Ensure the document has the required fields
                assert "document_id" in doc and "document_text" in doc, "Invalid document format"
                self.documents[doc["document_id"]] = doc["document_text"]
    
    def get_document(self, doc_id: int) -> str:
        """
        Retrieves the document text for the given document ID.

        Args:
            doc_id (int): The ID of the document to retrieve.

        Returns:
            str: The text of the document, or an empty string if the document ID is not found.
        """
        return self.documents.get(doc_id, "")