"""Сброс накопленных данных перед чистым перезапуском проекта.

Удаляет:
- Все коллекции Qdrant (trading_memory, news_memory, knowledge_memory, portfolio_history)
- Локальные MD-файлы новостей
- QR-код для авторизации Telethon

Запуск:
    python -m scripts.reset_knowledge_base
"""
import os
import shutil

from qdrant_client import QdrantClient

from config import settings


def main() -> None:
    """Основная функция: очистка всех данных."""
    client = QdrantClient(settings.QDRANT_HOST)

    # Удаляем все коллекции (включая knowledge_memory и portfolio_history)
    collections_to_delete = [
        settings.QDRANT_TRADING_COLLECTION,
        settings.QDRANT_NEWS_COLLECTION,
        settings.QDRANT_KNOWLEDGE_COLLECTION,
        settings.QDRANT_PORTFOLIO_COLLECTION,
    ]

    for name in collections_to_delete:
        if client.collection_exists(collection_name=name):
            client.delete_collection(collection_name=name)
            print(f"Удалена коллекция: {name}")

    # Удаляем папку с локальными MD-файлами новостей
    if os.path.isdir(settings.NEWS_DIR):
        shutil.rmtree(settings.NEWS_DIR)
        print(f"Удалена папка: {settings.NEWS_DIR}")

    # Удаляем QR-код авторизации Telethon
    if os.path.exists(settings.QR_CODE_FILE):
        os.remove(settings.QR_CODE_FILE)
        print(f"Удалён файл: {settings.QR_CODE_FILE}")

    print("\nВсе данные успешно удалены. Можно загружать базу знаний заново.")


if __name__ == "__main__":
    main()