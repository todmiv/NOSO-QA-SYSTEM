import json
import os
from src.add_metadata import extract_document_title


def load_document_names():
    """Загрузка соответствий имен файлов и названий документов."""
    names_file = "Names_of_ documents.txt"
    doc_names = {}
    if os.path.exists(names_file):
        with open(names_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        file_name = parts[0].strip()
                        doc_title = parts[1].strip()
                        doc_names[file_name] = doc_title
    return doc_names


def fix_existing_metadata():
    """Добавление document_title в существующие метаданные."""
    progress_file = "progress_metadata.json"

    if not os.path.exists(progress_file):
        print("Файл progress_metadata.json не найден")
        return

    doc_names = load_document_names()
    print(f"Загружено {len(doc_names)} соответствий названий документов")

    with open(progress_file, "r", encoding="utf-8") as f:
        enriched_docs = json.load(f)

    updated = False
    for doc_name, chunks in enriched_docs.items():
        if chunks and "metadata" in chunks[0]:
            # Получаем название из файла или извлекаем
            file_key = doc_name.replace(".txt", "")
            document_title = doc_names.get(file_key)
            if not document_title:
                # Fallback к извлечению из текста
                full_text = "".join([chunk["text"] for chunk in chunks])
                document_title = extract_document_title(doc_name, full_text)

            if (
                "document_title" not in chunks[0]["metadata"]
                or chunks[0]["metadata"]["document_title"] != document_title
            ):
                print(f"Обновляем название '{document_title}' для документа {doc_name}")

                # Добавляем в каждый чанк
                for chunk in chunks:
                    chunk["metadata"]["document_title"] = document_title
                updated = True

    if updated:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(enriched_docs, f, ensure_ascii=False, indent=2)
        print("Метаданные обновлены")
    else:
        print("Метаданные уже актуальны")


if __name__ == "__main__":
    fix_existing_metadata()
