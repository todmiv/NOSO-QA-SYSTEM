import json
import re


def fix_overlap_in_chunks():
    """Исправляет перекрытие чанков, добавляя недостающую часть предложения из предыдущего чанка."""
    progress_file = "progress_metadata.json"

    with open(progress_file, "r", encoding="utf-8") as f:
        enriched_docs = json.load(f)

    for doc_name, chunks in enriched_docs.items():
        if len(chunks) < 2:
            continue  # Нет перекрытия для одного чанка

        for i in range(1, len(chunks)):
            prev_chunk_text = chunks[i - 1]["text"].strip()
            current_chunk_text = chunks[i]["text"].strip()

            # Проверяем, начинается ли текущий чанк с маленькой буквы (продолжение предложения)
            first_char = current_chunk_text[0] if current_chunk_text else ""
            starts_with_lowercase = first_char.islower() and not first_char.isdigit()

            # Проверяем, заканчивается ли предыдущий чанк не точкой
            ends_without_period = not prev_chunk_text.endswith((".", "!", "?"))

            if starts_with_lowercase and ends_without_period:
                # Находим последнее предложение в предыдущем чанке
                sentences = re.split(r"(?<=[.!?])\s+", prev_chunk_text)
                if sentences:
                    last_sentence = sentences[-1].strip()
                    # Добавляем последнее предложение в начало текущего чанка
                    chunks[i]["text"] = last_sentence + " " + current_chunk_text
                    print(
                        f"Добавлено перекрытие для {doc_name}, чанк {i}: '{last_sentence}'"
                    )

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(enriched_docs, f, ensure_ascii=False, indent=2)

    print("Перекрытие чанков исправлено")


if __name__ == "__main__":
    fix_overlap_in_chunks()
