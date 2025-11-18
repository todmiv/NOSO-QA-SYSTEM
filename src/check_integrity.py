import chromadb
import os
from typing import Dict, List, Any

CHROMA_PATH = "../data/chroma_db"


def check_chroma_integrity() -> Dict[str, Any]:
    """Проверка целостности ChromaDB."""
    if not os.path.exists(CHROMA_PATH):
        return {"status": "error", "message": "База данных не найдена"}

    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()

        integrity_report = {"status": "ok", "collections": [], "total_chunks": 0}

        for collection in collections:
            count = collection.count()
            integrity_report["collections"].append(
                {"name": collection.name, "chunk_count": count}
            )
            integrity_report["total_chunks"] += count

        return integrity_report

    except Exception as e:
        return {"status": "error", "message": str(e)}


def list_documents() -> List[str]:
    """Список доступных документов (коллекций)."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collections = client.list_collections()
        return [col.name for col in collections]
    except Exception:
        return []


def get_document_chunks(collection_name: str) -> List[Dict[str, Any]]:
    """Получение всех чанков документа."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name=collection_name)
        results = collection.get(include=["documents", "metadatas"])
        chunks = []
        for i in range(len(results["documents"])):
            chunks.append(
                {
                    "id": results["ids"][i],
                    "text": results["documents"][i],
                    "metadata": results["metadatas"][i],
                }
            )
        return chunks
    except Exception as e:
        print(f"Ошибка получения чанков: {e}")
        return []


if __name__ == "__main__":
    report = check_chroma_integrity()
    print(report)
