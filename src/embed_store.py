import chromadb
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any
import os
from tqdm import tqdm

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHROMA_PATH = os.path.join(os.getcwd(), "data", "chroma_db")


def initialize_embedding_model():
    """Инициализация модели эмбеддингов."""
    return SentenceTransformer(EMBEDDING_MODEL)


def initialize_chroma_client():
    """Инициализация ChromaDB клиента."""
    os.makedirs(CHROMA_PATH, exist_ok=True)
    print(f"Chroma path: {os.path.abspath(CHROMA_PATH)}")
    return chromadb.Client(
        settings=chromadb.Settings(is_persistent=True, persist_directory=CHROMA_PATH)
    )


def create_embeddings(
    texts: List[str], model: SentenceTransformer, batch_size: int = 100
) -> List[List[float]]:
    """Создание эмбеддингов для списка текстов."""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, normalize_embeddings=True)
        embeddings.extend(batch_embeddings.tolist())
    return embeddings


def store_chunks_in_chroma(
    client: chromadb.Client,
    collection_name: str,
    chunks_with_metadata: List[Dict[str, Any]],
    model: SentenceTransformer,
):
    """Сохранение чанков с метаданными в ChromaDB."""
    collection = client.get_or_create_collection(name=collection_name)

    # Комбинированный текст для поиска: text + keywords + questions
    texts = []
    for chunk in chunks_with_metadata:
        text = chunk["text"]
        keywords = chunk["metadata"].get("keywords", "")
        questions = chunk["metadata"].get("questions", "")
        combined = f"{text} {keywords} {questions}"
        texts.append(combined)

    metadatas = [chunk["metadata"] for chunk in chunks_with_metadata]
    ids = [f"chunk_{i}" for i in range(len(texts))]

    embeddings = create_embeddings(texts, model)

    collection.add(embeddings=embeddings, documents=texts, metadatas=metadatas, ids=ids)


def process_and_store_documents(enriched_docs: Dict[str, List[Dict[str, Any]]]):
    """Обработка и сохранение всех документов."""
    model = initialize_embedding_model()
    client = initialize_chroma_client()

    total_docs = len(enriched_docs)
    with tqdm(total=total_docs, desc="Сохранение в БД") as pbar:
        for doc_name, chunks in enriched_docs.items():
            collection_name = doc_name.replace(".txt", "").replace(" ", "_")
            store_chunks_in_chroma(client, collection_name, chunks, model)
            print(f"Сохранено {len(chunks)} чанков для документа {doc_name}")
            pbar.update(1)

    print("Сохранение завершено, клиент закрыт")


if __name__ == "__main__":
    # Пример использования
    model = initialize_embedding_model()
    client = initialize_chroma_client()
    print("Модель и БД инициализированы")
