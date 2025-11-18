import json
from src.embed_store import process_and_store_documents

# Загрузить progress_metadata.json
with open("progress_metadata.json", "r", encoding="utf-8") as f:
    enriched_docs = json.load(f)

# Взять только один документ для теста
test_doc = {}
for doc_name, chunks in list(enriched_docs.items())[:1]:  # Первый документ
    test_doc[doc_name] = chunks[:1]  # Только первый чанк

print(f"Тестируем сохранение документа {doc_name} с 1 чанком")
print(f"Метаданные чанка: {test_doc[doc_name][0]['metadata']}")

try:
    process_and_store_documents(test_doc)
    print("Тест прошел успешно: чанк сохранен в ChromaDB")

    # Проверка retrieval
    from src.retrieval import retrieve_relevant_chunks

    collection_name = doc_name.replace(".txt", "").replace(" ", "_")
    chunks = retrieve_relevant_chunks("руководители", collection_name)
    print(f"Найдено {len(chunks)} чанков при поиске")
    if chunks:
        print("БД работает корректно")
    else:
        print("БД не сохранила данные")
except Exception as e:
    print(f"Ошибка: {e}")
