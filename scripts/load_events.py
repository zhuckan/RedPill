"""Загрузка исторических рыночных событий в векторную базу знаний (Qdrant).

Скрипт читает JSON-файл с описанием исторических кейсов (манипуляции,
ликвидации, пампы), склеивает все поля каждого события в единый текст
и загружает в коллекцию `knowledge_memory`.

ID событий начинаются с 10000, чтобы не пересекаться с чанками
статьи наставника (у них ID от 1 до N).

Запуск:
    python -m scripts.load_events
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from config import settings
from core.rag_memory import _embed, _init_collection

# Путь к файлу с историческими событиями
EVENTS_FILE_PATH = "knowledge_base/events/manipulations.json"


def load_historical_events() -> None:
    """Загружает исторические события в Qdrant.

    Читает JSON, формирует из каждого события единый текст (для лучшего
    семантического поиска), создает эмбеддинги и загружает батчами по 10 штук.
    """
    print("Инициализация коллекции...")
    _init_collection(settings.QDRANT_KNOWLEDGE_COLLECTION)
    client = QdrantClient(settings.QDRANT_HOST)

    if not os.path.exists(EVENTS_FILE_PATH):
        print(f"Ошибка: файл {EVENTS_FILE_PATH} не найден.")
        return

    with open(EVENTS_FILE_PATH, "r", encoding="utf-8") as f:
        events = json.load(f)

    print(f"Найдено {len(events)} событий. Начинаем обработку...")
    points = []

    for i, event in enumerate(events):
        print(f"Эмбеддинг события {i + 1}/{len(events)}: {event.get('event_name')}")

        # Склеиваем все поля в один большой текст для лучшего поиска
        full_text = (
            f"Историческое событие: {event.get('event_name')}. "
            f"Дата: {event.get('date')}. Тип: {event.get('type')}. "
            f"Настроение до: {event.get('pre_sentiment')}. "
            f"Триггер: {event.get('trigger')}. "
            f"Метрики до: {event.get('metrics_before')}. "
            f"Реакция: {event.get('reaction')}. "
            f"Ликвидации: {event.get('liquidations')}. "
            f"Настроение после: {event.get('post_sentiment')}. "
            f"Траектория: {event.get('trajectory')}. "
            f"Урок: {event.get('lesson')}."
        )

        vector = _embed(full_text)

        points.append(
            PointStruct(
                id=10000 + i,  # Смещаем ID, чтобы не пересекаться со статьей наставника
                vector=vector,
                payload={
                    "source": "historical_events",
                    "event_name": event.get("event_name"),
                    "date": event.get("date"),
                    "type": event.get("type"),
                    "text": full_text,
                    "lesson": event.get("lesson"),
                },
            )
        )

    print("Загрузка в Qdrant...")
    # Загружаем батчами по 10 штук, чтобы не перегружать память
    for i in range(0, len(points), 10):
        client.upsert(
            collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
            points=points[i: i + 10],
        )

    print("Все исторические события успешно загружены в базу знаний.")


if __name__ == "__main__":
    load_historical_events()