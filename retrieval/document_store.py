# retrieval/document_store.py
import json
from typing import Dict, List
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

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

    def chunk_documents(self, chunk_size: int = 600) -> list:
        """
        Splits the documents into chunks of specified size.

        Args:
            chunk_size (int): The max size of each chunk.
        """
        chunks = []
        id_count = 0
        for doc_id, text in self.documents.items():
            # Preprocess the text
            text = re.sub(r"\s+", " ", text).strip()  # Remove newlines and extra spaces
            text = re.sub(r"\t+", " ", text).strip()  # Remove tabs
            text = re.sub(r"<[^>]+>", " ", text).strip()  # Remove HTML tags and special characters

            # Tokenize the text into sentences
            sentences = sent_tokenize(text)

            # Group sentences into chunks
            current_chunk = ""
            for sentence in sentences:
                sentence = sentence.strip()
                # If adding the sentence exceeds the chunk size, finalize the current chunk
                if len(current_chunk) + len(sentence) > chunk_size:
                    id_count += 1
                    chunks.append(
                        DocumentChunk(
                            chunk_id=id_count,
                            doc_id=doc_id,
                            chunk_text=current_chunk
                        )
                    )
                    current_chunk = ""  # Start a new chunk

                current_chunk += sentence + " "  # Add the sentence to the current chunk
                current_chunk = current_chunk.strip()  # Remove leading and trailing whitespace
 
            # Add the last chunk if it contains any text
            if current_chunk.strip():
                id_count += 1
                chunks.append(
                    DocumentChunk(
                        chunk_id=id_count,
                        doc_id=doc_id,
                        chunk_text=current_chunk.strip()
                    )
                )

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

    def get_doc_retrieval_response(self, sorted_chunk_ids: List[int], top_k: int) -> DocRetrievalResponse:
        doc_ids = []
        doc_chunks = []

        for i in sorted_chunk_ids:
            doc_chunk: DocumentChunk = self.get_chunk(i)

            # doc_chunks
            if len(doc_chunks) < top_k:
                doc_chunks.append(doc_chunk)

            # doc_ids
            if doc_chunk.doc_id not in doc_ids:
                # 如果文档 ID 不在 doc_ids 中，则添加
                doc_ids.append(doc_chunk.doc_id)
                if len(doc_ids) >= top_k:
                    break

        return DocRetrievalResponse(doc_ids, doc_chunks)
