import os
import asyncio
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document as LangchainDocument
from src.core.config import settings
import uuid


async def ingest_file(file_path: str):
    """Ingest a single file for a specific company."""
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return
    loader = DirectoryLoader(
        path=file_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )

    documents = loader.load()

    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", " ", ""]
    )
    chunks = test_splitter.split_documents(documents)

    embeddings = OllamaEmbeddings(
        model=settings.ollama_embedding_model, base_url=settings.ollama_base_url
    )
    vector_store = Chroma(
        persist_directory="db/chroma",
        embedding_function=embeddings,
        collection_name="documents",
    )

    doc_id = str(uuid.uuid4())
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        metadata = {
            "doc_id": doc_id,
            "source": file_path,
            "chunk_index": i,
        }
        processed_chunks.append(
            LangchainDocument(page_content=chunk.page_content, metadata=metadata)
        )

    vector_store.add_documents(processed_chunks)
    print(f"successfully ingested {len(processed_chunks)} chunks into vector store")


async def main():
    # Example usage
    doc_path = "src/documents"
    if os.path.exists(doc_path):
        await ingest_file(doc_path)
    else:
        print(f"Default document not found at {doc_path}")


if __name__ == "__main__":
    asyncio.run(main())
