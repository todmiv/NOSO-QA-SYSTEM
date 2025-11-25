import gradio as gr
from src.chunking import load_documents, hierarchical_chunk_documents
from src.add_metadata import add_metadata_to_hierarchical_chunks
from src.embed_store import process_and_store_documents
from src.retrieval import (
    retrieve_relevant_chunks,
    retrieve_from_all_collections,
    generate_answer_with_deepseek,
)
from src.check_integrity import list_documents
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Загрузка переменных окружения из .env
load_dotenv()

# API ключ DeepSeek (placeholder, заменить на реальный)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your_deepseek_api_key")


def initialize_system():
    """Инициализация системы: загрузка и обработка документов с иерархическим чанкированием."""
    docs_dir = "documents/txts"
    if not os.path.exists("data/chroma_db"):
        print("Инициализация базы данных с иерархическим чанкированием...")
        with tqdm(total=4, desc="Общий прогресс инициализации") as pbar:
            pbar.set_description("Загрузка документов")
            docs = load_documents(docs_dir)
            pbar.update(1)

            pbar.set_description("Иерархическое чанкирование документов")
            chunked = hierarchical_chunk_documents(docs)
            pbar.update(1)

            pbar.set_description("Генерация метаданных (может занять время)")
            enriched = add_metadata_to_hierarchical_chunks(chunked)
            pbar.update(1)

            pbar.set_description("Сохранение в БД")
            process_and_store_documents(enriched)
            pbar.update(1)

        print("База данных инициализирована с иерархическим чанкированием.")
    else:
        print("База данных уже существует.")


def chat_with_ai(query: str, selected_doc: str) -> str:
    """Чат с AI: поиск и генерация ответа."""
    if not selected_doc:
        return "Выберите документ для поиска."

    if selected_doc == "Все документы":
        chunks = retrieve_from_all_collections(query, top_k=15)
        if not chunks:
            return "Информация не найдена в документах."
    else:
        chunks = retrieve_relevant_chunks(
            query, selected_doc.replace(".txt", "").replace(" ", "_"), top_k=15
        )
        if not chunks:
            return "Информация не найдена в выбранном документе."

    answer = generate_answer_with_deepseek(query, chunks, DEEPSEEK_API_KEY)
    return answer


def search_documents(query: str, selected_doc: str) -> str:
    """Поиск по документам."""
    if not selected_doc:
        return "Выберите документ."

    if selected_doc == "Все документы":
        chunks = retrieve_from_all_collections(query, top_k=5)
        if not chunks:
            return "Ничего не найдено."
        results = []
        for i, chunk in enumerate(chunks, 1):
            collection_info = f" (Документ: {chunk.get('collection', 'Неизвестно')})"
            results.append(
                f"{i}. {chunk['text'][:200]}... (Релевантность: {1 - chunk['distance']:.2f}){collection_info}"
            )
    else:
        chunks = retrieve_relevant_chunks(
            query, selected_doc.replace(".txt", "").replace(" ", "_"), top_k=5
        )
        if not chunks:
            return "Ничего не найдено."
        results = []
        for i, chunk in enumerate(chunks, 1):
            results.append(
                f"{i}. {chunk['text'][:200]}... (Релевантность: {1 - chunk['distance']:.2f})"
            )
    return "\n\n".join(results)


def analyze_chunks(query: str, selected_doc: str) -> str:
    """Анализ найденных чанков с векторным сходством."""
    if not selected_doc:
        return "Выберите документ."

    if selected_doc == "Все документы":
        chunks = retrieve_from_all_collections(
            query, top_k=15
        )  # Те же чанки, что для чата с AI
    else:
        chunks = retrieve_relevant_chunks(
            query, selected_doc.replace(".txt", "").replace(" ", "_"), top_k=15
        )

    if not chunks:
        return "Чанки не найдены."

    results = []
    for i, chunk in enumerate(chunks, 1):
        doc_title = chunk.get("metadata", {}).get(
            "document_title", "Неизвестный документ"
        )
        summary = chunk.get("metadata", {}).get("summary", "Резюме недоступно")
        relevance = 1 - chunk["distance"]
        results.append(
            f"Чанк {i}:\n"
            f"Текст: {chunk['text'][:300]}...\n"
            f"Документ: {doc_title}\n"
            f"Резюме: {summary}\n"
            f"Векторное сходство: {relevance:.4f}\n"
            f"---"
        )
    return "\n\n".join(results)


def get_available_documents() -> list:
    """Получение списка доступных документов."""
    docs = list_documents()
    return ["Все документы"] + docs


# Gradio интерфейс
with gr.Blocks(title="Q&A по документам СРО НОСО") as app:
    gr.Markdown("# Система Q&A по документам СРО НОСО")

    with gr.Row():
        doc_dropdown = gr.Dropdown(
            label="Выберите документ", choices=get_available_documents(), value=None
        )

    with gr.Tab("Чат с AI"):
        chat_history_state = gr.State([])
        query_input = gr.Textbox(label="Введите вопрос")
        chat_button = gr.Button("Задать вопрос")
        chat_output = gr.Textbox(label="Ответ AI", lines=10)
        show_analyze_checkbox = gr.Checkbox(
            label="Показать окно Анализ чанков"
        )
        analyze_output = gr.Textbox(label="Анализ чанков", lines=25, visible=False)
        chat_history_display = gr.Textbox(label="История диалогов", lines=20, interactive=False)

        def toggle_analyze_visibility(visible):
            return gr.update(visible=visible)

        show_analyze_checkbox.change(
            toggle_analyze_visibility,
            inputs=[show_analyze_checkbox],
            outputs=[analyze_output]
        )

        def chat_and_analyze(query, selected_doc, analyze_visible, history):
            answer = chat_with_ai(query, selected_doc)
            history.append((query, answer))
            history_str = "\n\n".join([f"Вопрос: {q}\nОтвет: {a}" for q, a in history])
            if analyze_visible:
                analyze_result = analyze_chunks(query, selected_doc)
            else:
                analyze_result = gr.update()
            return answer, analyze_result, history_str, history

        chat_button.click(
            chat_and_analyze,
            inputs=[query_input, doc_dropdown, show_analyze_checkbox, chat_history_state],
            outputs=[chat_output, analyze_output, chat_history_display, chat_history_state],
        )

        def clear_history():
            return "", gr.update(), "", []

        clear_button = gr.Button("Очистить историю")
        clear_button.click(
            clear_history,
            inputs=[],
            outputs=[chat_output, analyze_output, chat_history_display, chat_history_state],
        )

    with gr.Tab("Поиск по документам"):
        search_query = gr.Textbox(label="Введите запрос для поиска")
        search_output = gr.Textbox(label="Результаты поиска", lines=15)
        search_button = gr.Button("Поиск")
        search_button.click(
            search_documents, inputs=[search_query, doc_dropdown], outputs=search_output
        )

    with gr.Tab("Анализ чанков"):
        analyze_query = gr.Textbox(label="Введите вопрос для анализа чанков")
        analyze_tab_output = gr.Textbox(label="Анализ чанков", lines=25)
        analyze_button = gr.Button("Анализировать")
        analyze_button.click(
            analyze_chunks, inputs=[analyze_query, doc_dropdown], outputs=analyze_tab_output
        )

if __name__ == "__main__":
    initialize_system()
    app.launch()
