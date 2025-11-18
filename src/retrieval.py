import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import os

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHROMA_PATH = os.path.join(os.getcwd(), "data", "chroma_db")


def initialize_embedding_model():
    """Инициализация модели эмбеддингов."""
    return SentenceTransformer(EMBEDDING_MODEL)


def initialize_chroma_client():
    """Инициализация ChromaDB клиента."""
    return chromadb.Client(
        settings=chromadb.Settings(is_persistent=True, persist_directory=CHROMA_PATH)
    )


def retrieve_relevant_chunks(
    query: str,
    collection_name: str,
    top_k: int = 5,
    model: SentenceTransformer = None,
    client: chromadb.Client = None,
) -> List[Dict[str, Any]]:
    """Поиск релевантных чанков по запросу в указанной коллекции."""
    if model is None:
        model = initialize_embedding_model()
    if client is None:
        client = initialize_chroma_client()

    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        return []

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)

    relevant_chunks = []
    for i in range(len(results["documents"][0])):
        chunk = {
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
            "collection": collection_name,  # Добавляем имя коллекции
        }
        relevant_chunks.append(chunk)

    return relevant_chunks


def retrieve_from_all_collections(
    query: str,
    top_k: int = 5,
    model: SentenceTransformer = None,
    client: chromadb.Client = None,
) -> List[Dict[str, Any]]:
    """Поиск релевантных чанков по запросу во всех коллекциях."""
    if model is None:
        model = initialize_embedding_model()
    if client is None:
        client = initialize_chroma_client()

    query_embedding = model.encode([query], normalize_embeddings=True).tolist()[0]

    all_chunks = []
    collections = client.list_collections()
    for collection in collections:
        try:
            results = collection.query(
                query_embeddings=[query_embedding], n_results=top_k
            )
            for i in range(len(results["documents"][0])):
                chunk = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                    "collection": collection.name,  # Добавляем имя коллекции
                }
                all_chunks.append(chunk)
        except Exception:
            continue

    # Сортируем по distance (меньше - лучше) и берем топ top_k
    all_chunks.sort(key=lambda x: x["distance"])
    return all_chunks[:top_k]


def generate_answer_with_deepseek(
    query: str, context_chunks: List[Dict[str, Any]], api_key: str
) -> str:
    """Генерация ответа с помощью DeepSeek."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # Формируем контекст с указанием названия документа для каждого чанка
    context_parts = []
    for chunk in context_chunks:
        doc_title = chunk.get("metadata", {}).get(
            "document_title", "Неизвестный документ"
        )
        context_parts.append(f"Документ: {doc_title}\n{chunk['text']}")
    context = "\n\n".join(context_parts)

    prompt = f"""
На основе следующего контекста ответь на вопрос пользователя. Указывай сокращенные названия документов: СРО НОСО для "САМОРЕГУЛИРУЕМОЙ ОРГАНИЗАЦИИ АССОЦИАЦИИ «НИЖЕГОРОДСКОЕ ОБЪЕДИНЕНИЕ СТРОИТЕЛЬНЫХ ОРГАНИЗАЦИЙ»". Указывай номера пунктов/разделов при ссылках.

Если ссылок на один документ много, группируйте их в конце ответа по документам.

Если контекст не содержит информации для ответа, скажи "Информация не найдена в документах".

Контекст:
{context}

Вопрос: {query}

Ответ:
"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Ошибка генерации ответа: {e}"


if __name__ == "__main__":
    # Пример использования
    model = initialize_embedding_model()
    client = initialize_chroma_client()
    query = "Что такое СМР?"
    chunks = retrieve_relevant_chunks(
        query, "Standart_NOSO", model=model, client=client
    )
    print(f"Найдено {len(chunks)} релевантных чанков")
