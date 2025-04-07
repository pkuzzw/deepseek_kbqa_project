# retrieval/document_store.py
import json
from typing import Dict, List
import re

class DocRetrievalResponse:
    def __init__(self, topk_doc_id: List[int], topk_doc_chunk: List['DocumentChunk']):
        """
        Initializes the DocRetrievalResponse with the given document id and text.

        Args:
            topk_doc_id (List[int]): List of document IDs.
            topk_doc_chunk (List['DocumentChunk']): List of DocumentChunk objects.
        """
        self.topk_doc_id = topk_doc_id
        self.topk_doc_chunk = topk_doc_chunk

class DocumentChunk:
    def __init__(self, chunk_id: int, doc_id: int, chunk_text: str):
        """
        Initializes the DocumentChunk with the given id, doc id and text.

        Args:
            chunk_id (int): The ID of the chunk.
            doc_id (int): The ID of the document.
            chunk_text (str): The text of the chunk.
        """
        self.chunk_id = chunk_id
        self.doc_id = doc_id
        self.chunk_text = chunk_text

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


        # Initialize chunks
        self.document_chunks: Dict[int, DocumentChunk] = {}
        chunks = self.chunk_documents()
        for chunk in chunks:
            self.document_chunks[chunk.chunk_id] = chunk

    def chunk_documents(self, chunk_size: int = 400) -> list:
        """
        Splits the documents into chunks of specified size.

        Args:
            chunk_size (int): The size of each chunk.
        """
        chunks = []
        id_count = 0
        for doc_id, text in self.documents.items():
            # Remove newlines and extra spaces
            text = re.sub(r"\s+", " ", text).strip()
            # Remove HTML tags and special characters
            text = re.sub(r"<[^>]+>", " ", text)

            for i in range(0, len(text), chunk_size):
                id_count += 1

                # if the remaining text is less than chunk_size + chunk_size/5, group it tgt
                if len(text[i:]) < chunk_size + chunk_size / 5:
                    chunk = text[i:]
                    chunks.append(
                        DocumentChunk(
                            chunk_id=id_count,
                            doc_id=doc_id,
                            chunk_text=chunk
                    ))
                    break

                chunk = text[i:i + chunk_size]
                chunks.append(
                    DocumentChunk(
                        chunk_id=id_count,
                        doc_id=doc_id,
                        chunk_text=chunk
                ))
        return chunks
    
    def get_document(self, doc_id: int) -> str:
        """
        Retrieves the document text for the given document ID.

        Args:
            doc_id (int): The ID of the document to retrieve.

        Returns:
            str: The text of the document, or an empty string if the document ID is not found.
        """
        return self.documents.get(doc_id, "")

    def get_chunk(self, chunk_id: int) -> DocumentChunk:
        """
        Retrieves the DocumentChunk object for the given chunk ID.

        Args:
            chunk_id (int): The ID of the chunk to retrieve.

        Returns:
            DocumentChunk: The DocumentChunk object, or None if the chunk ID is not found.
        """
        return self.document_chunks.get(chunk_id, None)
    
    def get_document_by_chunk_id(self, chunk_id: int) -> str:
        """
        Retrieves the document text for the given chunk ID.

        Args:
            chunk_id (int): The ID of the chunk to retrieve.

        Returns:
            str: The text of the chunk, or an empty string if the chunk ID is not found.
        """
        chunk = self.document_chunks.get(chunk_id)
        if chunk:
            return self.documents.get(chunk.doc_id, "")
        else:
            return ""
