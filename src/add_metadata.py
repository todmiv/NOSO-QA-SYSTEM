import requests
import json
import os
import time
from typing import Dict, List, Any
from tqdm import tqdm
from .chunking import HierarchicalChunk

OLLAMA_URL = "http://localhost:11434/api/generate"


def load_document_names_mapping() -> Dict[str, str]:
    """Загрузка маппинга имен файлов к названиям документов из Names_of_ documents.txt."""
    mapping = {}
    try:
        with open("Names_of_ documents.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and "\t" in line:
                    parts = line.split("\t", 1)
                    if len(parts) == 2:
                        filename = parts[0].strip()
                        title = parts[1].strip()
                        mapping[filename] = title
    except FileNotFoundError:
        print("Файл Names_of_ documents.txt не найден, используется резервный метод")
    return mapping


def generate_metadata_with_llm(chunk: str, model: str = "qwen2.5:3b") -> Dict[str, Any]:
    """Генерация метаданных для чанка с помощью локальной LLM."""
    prompt = f"""
Для следующего текста сгенерируй метаданные в формате JSON:

Текст: {chunk}

Формат ответа (только JSON, без дополнительного текста):
{{
  "summary": "Краткое резюме 1-2 предложений",
  "keywords": ["ключевое_слово1", "ключевое_слово2", ...],
  "category": "Категория контента (например, технический, финансовый, юридический)",
  "questions": ["Вопрос1?", "Вопрос2?", ...]
}}
"""

    payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        generated_text = result.get("response", "")

        # Парсинг JSON из ответа
        metadata = json.loads(generated_text.strip())

        # Преобразование lists в strings для совместимости с ChromaDB
        if "keywords" in metadata and isinstance(metadata["keywords"], list):
            metadata["keywords"] = ", ".join(metadata["keywords"])
        if "questions" in metadata and isinstance(metadata["questions"], list):
            metadata["questions"] = "; ".join(metadata["questions"])

        return metadata
    except Exception as e:
        print(f" Ошибка генерации метаданных: {e}")
        return {
            "summary": "Резюме недоступно",
            "keywords": "",
            "category": "Неизвестно",
            "questions": "",
        }


def extract_document_title(doc_name: str, text: str) -> str:
    """Извлечение названия документа из файла маппинга или текста."""
    # Сначала пытаемся найти в файле маппинга
    mapping = load_document_names_mapping()
    base_name = doc_name.replace(".txt", "")

    if base_name in mapping:
        return mapping[base_name]

    # Резервный метод: извлечение из текста
    lines = text.split("\n")[:10]  # Первые 10 строк
    for line in lines:
        line = line.strip()
        if (
            line
            and len(line) > 10
            and not line.startswith("УТВЕРЖДЕНО")
            and not line.startswith("г.")
        ):
            return line
    return doc_name.replace("_", " ").replace(".txt", "")


def add_metadata_to_hierarchical_chunks(
    chunked_docs: Dict[str, List[HierarchicalChunk]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Добавление метаданных к иерархическим чанкам с сохранением прогресса."""
    progress_file = "progress_metadata_hierarchical.json"
    enriched_docs = {}

    # Загрузка существующего прогресса
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            enriched_docs = json.load(f)
        print(f"Загружен прогресс из {progress_file}")

    total_chunks = sum(len(chunks) for chunks in chunked_docs.values())
    processed_chunks = sum(
        len(enriched_docs.get(doc_name, [])) for doc_name in chunked_docs.keys()
    )
    remaining_chunks = total_chunks - processed_chunks

    with tqdm(
        total=remaining_chunks,
        desc="Генерация метаданных для иерархических чанков",
        initial=0,
    ) as pbar:
        processing_times = []  # Для расчета ETA

        for doc_name, chunks in chunked_docs.items():
            if doc_name in enriched_docs:
                print(f"Документ {doc_name} уже обработан, пропускаем")
                continue

            # Извлечение названия документа из первого чанка
            document_title = extract_document_title(
                doc_name, chunks[0].text if chunks else ""
            )

            print(f"Обработка документа: {doc_name} ({len(chunks)} чанков)")
            enriched_chunks = []
            doc_start_time = time.time()

            for i, chunk in enumerate(chunks):
                chunk_start_time = time.time()

                # Генерируем дополнительные метаданные с помощью LLM
                # Используем только контент чанка (без заголовка и ключевых слов)
                content_lines = chunk.text.split("\n\n")
                content = content_lines[-1] if len(content_lines) > 2 else chunk.text

                metadata = generate_metadata_with_llm(content)
                metadata["document_title"] = document_title
                metadata["section_path"] = chunk.section_path
                metadata["hierarchy_level"] = chunk.hierarchy_level
                metadata["section_title"] = chunk.section_title
                metadata["keywords"] = ", ".join(chunk.keywords)  # преобразуем в строку

                # Добавляем информацию о перекрытии если есть
                if chunk.overlap_info:
                    metadata["overlap_size"] = chunk.overlap_info.get("overlap_size", 0)

                enriched_chunk = {"text": chunk.text, "metadata": metadata}
                enriched_chunks.append(enriched_chunk)

                # Расчет времени обработки чанка
                chunk_time = time.time() - chunk_start_time
                processing_times.append(chunk_time)

                # Расчет ETA для текущего документа
                if len(processing_times) >= 3:  # После первых 3 чанков для стабильности
                    avg_time_per_chunk = sum(processing_times[-10:]) / min(
                        10, len(processing_times)
                    )  # Среднее по последним 10
                    remaining_chunks_in_doc = len(chunks) - (i + 1)
                    eta_seconds = avg_time_per_chunk * remaining_chunks_in_doc
                    eta_minutes = eta_seconds / 60

                    if eta_minutes > 1:
                        pbar.set_description(
                            f"Обработка {doc_name}: ~{eta_minutes:.1f} мин до завершения"
                        )
                    else:
                        pbar.set_description(
                            f"Обработка {doc_name}: ~{eta_seconds:.0f} сек до завершения"
                        )

                pbar.update(1)

            doc_time = time.time() - doc_start_time
            print(
                f"Документ {doc_name} обработан за {doc_time:.1f} сек ({len(chunks)} чанков)"
            )

            enriched_docs[doc_name] = enriched_chunks

            # Сохранение прогресса после каждого документа
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(enriched_docs, f, ensure_ascii=False, indent=2)
            print(f"Прогресс сохранен для документа {doc_name}")
    return enriched_docs


def add_metadata_to_chunks(
    chunked_docs: Dict[str, List[str]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Добавление метаданных ко всем чанкам с сохранением прогресса (устаревший метод)."""
    progress_file = "progress_metadata.json"
    enriched_docs = {}

    # Загрузка существующего прогресса
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            enriched_docs = json.load(f)
        print(f"Загружен прогресс из {progress_file}")

    total_chunks = sum(len(chunks) for chunks in chunked_docs.values())
    processed_chunks = sum(
        len(enriched_docs.get(doc_name, [])) for doc_name in chunked_docs.keys()
    )
    remaining_chunks = total_chunks - processed_chunks

    with tqdm(total=remaining_chunks, desc="Генерация метаданных", initial=0) as pbar:
        for doc_name, chunks in chunked_docs.items():
            if doc_name in enriched_docs:
                print(f"Документ {doc_name} уже обработан, пропускаем")
                continue
            # Извлечение названия документа
            full_text = "".join(chunks)  # Собираем полный текст для извлечения названия
            document_title = extract_document_title(doc_name, full_text)
            enriched_chunks = []
            for chunk in chunks:
                metadata = generate_metadata_with_llm(chunk)
                metadata["document_title"] = (
                    document_title  # Добавляем название документа
                )
                enriched_chunk = {"text": chunk, "metadata": metadata}
                enriched_chunks.append(enriched_chunk)
                pbar.update(1)
            enriched_docs[doc_name] = enriched_chunks
            # Сохранение прогресса после каждого документа
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(enriched_docs, f, ensure_ascii=False, indent=2)
            print(f" Обработан документ {doc_name}, сохранен прогресс")
    return enriched_docs


if __name__ == "__main__":
    # Пример использования
    sample_chunk = "Это пример текста для тестирования."
    metadata = generate_metadata_with_llm(sample_chunk)
    print(metadata)
