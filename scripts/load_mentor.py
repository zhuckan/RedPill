"""Загрузка статьи наставника в векторную базу знаний (Qdrant).

Скрипт читает Markdown-файл со статьей, очищает текст от артефактов Notion
(кнопки, невидимые символы, описания картинок), разбивает его на смысловые
чанки и загружает в коллекцию `knowledge_memory`.

Запуск:
    python -m scripts.load_mentor
"""
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import settings
from core.rag_memory import _embed, _init_collection


def clean_text(text: str) -> str:
    """Очищает текст от артефактов Notion.

    Args:
        text: Исходный текст статьи.

    Returns:
        Очищенный текст, готовый к чанкингу.
    """
    # Удаляем кнопки (Click) и ( Click )
    text = re.sub(r'\(\s*Click\s*\)', '', text, flags=re.IGNORECASE)

    # Удаляем невидимые пробелы (zero-width space)
    text = text.replace('\u200b', '')

    # Удаляем описания графиков (строки, начинающиеся с | и содержащие 'график' или 'визуализация')
    text = re.sub(
        r'\|\s*(На графике|График|Визуализация).*?\n',
        '',
        text,
        flags=re.IGNORECASE,
    )

    # Удаляем множественные пустые строки (оставляем максимум две подряд)
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def load_mentor_article() -> None:
    """Загружает статью наставника в Qdrant.

    Читает файл, очищает текст, режет на чанки (1500 символов с перекрытием 200)
    и загружает в коллекцию `knowledge_memory` с метаданными.
    """
    print("Инициализация коллекции...")
    _init_collection(settings.QDRANT_KNOWLEDGE_COLLECTION)
    client = QdrantClient(settings.QDRANT_HOST)

    if not os.path.exists(settings.MENTOR_ARTICLE_FILE):
        print(f"Ошибка: файл {settings.MENTOR_ARTICLE_FILE} не найден.")
        return

    with open(settings.MENTOR_ARTICLE_FILE, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("Очистка текста от артефактов Notion...")
    text = clean_text(raw_text)

    print("Нарезка текста на чанки...")
    # chunk_size=1500: статья плотная по смыслу, крупные чанки сохраняют контекст.
    # overlap=200: перекрытие между чанками, чтобы не терять связь на границах.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_text(text)
    print(f"Получено {len(chunks)} чанков.")

    points = []
    for i, chunk in enumerate(chunks):
        print(f"Эмбеддинг чанка {i + 1}/{len(chunks)}...")
        vector = _embed(chunk)

        points.append(
            PointStruct(
                id=i + 1,
                vector=vector,
                payload={
                    "source": "mentor_article_2026-10-28",
                    "text": chunk,
                    "date": "2026-10-28",
                    "type": "mentor_insight",
                },
            )
        )

    print("Загрузка в Qdrant...")
    client.upsert(
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
        points=points,
    )
    print("Статья наставника успешно загружена в базу знаний.")


if __name__ == "__main__":
    load_mentor_article()