import re
from typing import List, Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class Section:
    """Структура для представления раздела документа."""

    level: int
    number: str
    title: str
    content: str
    parent_path: str = ""


@dataclass
class HierarchicalChunk:
    """Иерархический чанк с метаданными."""

    text: str
    section_path: str
    hierarchy_level: int
    section_title: str
    keywords: List[str]
    overlap_info: Optional[Dict[str, Any]] = None


def parse_sections(text: str) -> List[Section]:
    """Разбор текста на иерархические разделы."""
    # Паттерн для заголовков: 1., 1.1., 2., etc.
    # Заголовки обычно короткие (< 200 символов) и не содержат точек в конце
    header_pattern = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(.+?)(?:\n|$)")

    sections = []
    lines = text.split("\n")
    current_section = None
    current_content = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Пропустить пустые строки в начале
        if not line:
            i += 1
            continue

        match = header_pattern.match(line)
        if match:
            number = match.group(1)
            title = match.group(2).strip()

            # Проверка: заголовок должен быть разумной длины (< 200 символов)
            # и не содержать много точек (не быть предложением)
            if len(title) < 200 and title.count(".") <= 1:
                # Это заголовок! Сохранить предыдущий раздел
                if current_section:
                    current_section.content = "\n".join(current_content).strip()
                    sections.append(current_section)

                # Определить уровень иерархии
                level = len(number.split(".")) - 1 if "." in number else 1

                # Построить путь родительских разделов
                parent_path = ""
                if level > 1:
                    # Найти родительский раздел
                    parent_num = ".".join(number.split(".")[:-1]) + "."
                    for sec in reversed(sections):
                        if sec.number == parent_num:
                            parent_path = (
                                sec.parent_path + " > " + sec.title
                                if sec.parent_path
                                else sec.title
                            )
                            break

                current_section = Section(
                    level=level,
                    number=number,
                    title=title,
                    content="",
                    parent_path=parent_path,
                )
                current_content = []
            else:
                # Это не заголовок, а часть контента
                if current_section:
                    current_content.append(line)
        else:
            # Часть контента
            if current_section:
                current_content.append(line)

        i += 1

    # Сохранить последний раздел
    if current_section:
        current_section.content = "\n".join(current_content).strip()
        sections.append(current_section)

    return sections


def find_word_boundary(text: str, target_pos: int, direction: int = -1) -> int:
    """Найти границу слова близкую к target_pos."""
    if direction == -1:  # искать влево
        pos = target_pos
        while pos > 0 and text[pos - 1] not in " \t\n":
            pos -= 1
        return pos
    else:  # искать вправо
        pos = target_pos
        while pos < len(text) and text[pos] not in " \t\n":
            pos += 1
        return pos


def create_adaptive_chunks(
    content: str, section_title: str, keywords: List[str], base_chunk_size: int = 1250
) -> List[HierarchicalChunk]:
    """Создание чанков с адаптивным перекрытием на границах слов."""
    if not content.strip():
        return []

    # Если контент меньше размера чанка, вернуть один чанк
    if len(content) <= base_chunk_size:
        chunk_text = (
            f"{section_title}\n\nКлючевые слова: {', '.join(keywords)}\n\n{content}"
        )
        return [
            HierarchicalChunk(
                text=chunk_text,
                section_path=section_title,
                hierarchy_level=1,  # временно
                section_title=section_title,
                keywords=keywords,
            )
        ]

    chunks = []
    start = 0
    content_length = len(content)

    while start < content_length:
        # Рассчитать адаптивный размер перекрытия
        remaining = content_length - start
        overlap_size = min(200, int(remaining * 0.15))  # 15% от оставшегося

        # Определить конец чанка
        end = min(start + base_chunk_size, content_length)

        # Если не конец текста, найти границу слова
        if end < content_length:
            end = find_word_boundary(content, end)

        # Извлечь текст чанка
        chunk_content = content[start:end]

        # Добавить заголовок и ключевые слова
        full_chunk_text = f"{section_title}\n\nКлючевые слова: {', '.join(keywords)}\n\n{chunk_content}"

        # Информация о перекрытии
        overlap_info = None
        if start > 0:
            overlap_info = {
                "overlap_size": overlap_size,
                "previous_chunk_end": content[max(0, start - overlap_size) : start],
            }

        chunk = HierarchicalChunk(
            text=full_chunk_text,
            section_path=section_title,
            hierarchy_level=1,  # будет установлено позже
            section_title=section_title,
            keywords=keywords,
            overlap_info=overlap_info,
        )
        chunks.append(chunk)

        # Следующий старт с учетом перекрытия
        if end >= content_length:
            break
        start = max(start + base_chunk_size - overlap_size, end - overlap_size)
        start = find_word_boundary(content, start, 1)  # найти начало слова

    return chunks


def split_into_chunks(
    text: str, chunk_size: int = 1000, overlap: int = 200
) -> List[str]:
    """Разбиение текста на семантические чанки с перекрытием."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                # Перекрытие: взять последние overlap символов
                overlap_text = (
                    current_chunk[-overlap:]
                    if len(current_chunk) > overlap
                    else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                chunks.append(paragraph)
        else:
            current_chunk += "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def load_documents(directory: str) -> Dict[str, str]:
    """Загрузка документов из директории."""
    documents = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                documents[filename] = f.read()
    return documents


def hierarchical_chunk_documents(
    documents: Dict[str, str], base_chunk_size: int = 1250
) -> Dict[str, List[HierarchicalChunk]]:
    """Иерархическое чанкирование всех документов."""
    chunked_docs = {}

    for doc_name, text in documents.items():
        # Разбор на разделы
        sections = parse_sections(text)
        all_chunks = []

        for section in sections:
            # Создание ключевых слов из заголовка (простая эвристика)
            keywords = extract_keywords_from_title(section.title)

            # Создание чанков для раздела
            chunks = create_adaptive_chunks(
                section.content,
                f"{section.number} {section.title}",
                keywords,
                base_chunk_size,
            )

            # Установка правильного уровня иерархии
            for chunk in chunks:
                chunk.hierarchy_level = section.level
                # Построить полный путь
                chunk.section_path = (
                    section.parent_path + " > " + f"{section.number} {section.title}"
                    if section.parent_path
                    else f"{section.number} {section.title}"
                )

            all_chunks.extend(chunks)

        chunked_docs[doc_name] = all_chunks

    return chunked_docs


def extract_keywords_from_title(title: str) -> List[str]:
    """Извлечение ключевых слов из заголовка (простая эвристика)."""
    # Разделить по пробелам, убрать короткие слова
    words = title.lower().split()
    keywords = [word for word in words if len(word) > 3]
    return keywords[:5]  # максимум 5 ключевых слов


def chunk_documents(documents: Dict[str, str]) -> Dict[str, List[str]]:
    """Чанкирование всех документов (устаревший метод для совместимости)."""
    chunked_docs = {}
    for doc_name, text in documents.items():
        chunks = split_into_chunks(text)
        chunked_docs[doc_name] = chunks
    return chunked_docs


if __name__ == "__main__":
    # Тестирование иерархического чанкирования
    docs_dir = os.path.join(os.path.dirname(__file__), "..", "documents", "txts")
    docs = load_documents(docs_dir)

    # Тестировать на одном документе
    test_doc = "Standart_NOSO.txt"
    if test_doc in docs:
        print(f"Тестирование на документе: {test_doc}")

        # Разбор на разделы
        sections = parse_sections(docs[test_doc])
        print(f"Найдено {len(sections)} разделов:")

        for i, section in enumerate(sections[:5]):  # первые 5 разделов
            print(
                f"  {i + 1}. Уровень {section.level}: {section.number} {section.title}"
            )
            print(f"     Длина контента: {len(section.content)} символов")

        # Иерархическое чанкирование
        hierarchical_chunks = hierarchical_chunk_documents({test_doc: docs[test_doc]})
        chunks = hierarchical_chunks[test_doc]

        print(f"\nСоздано {len(chunks)} иерархических чанков:")
        for i, chunk in enumerate(chunks[:3]):  # первые 3 чанка
            print(f"  Чанк {i + 1}:")
            print(f"    Уровень: {chunk.hierarchy_level}")
            print(f"    Путь: {chunk.section_path}")
            print(f"    Ключевые слова: {chunk.keywords}")
            print(f"    Длина текста: {len(chunk.text)}")
            if chunk.overlap_info:
                print(f"    Перекрытие: {chunk.overlap_info['overlap_size']} символов")
            print()

    else:
        print(f"Документ {test_doc} не найден")
        chunked = chunk_documents(docs)
        print(f"Чанкировано {len(chunked)} документов (старым методом)")
