"""Сброс тестовой истории для старта реального использования бота.

Удаляет только runtime-данные, связанные с историей работы агента:
- коллекцию trading_memory (журнал прогнозов и вердиктов),
- коллекцию portfolio_history (снимки баланса для контекста),
- коллекцию chat_history_memory (семантическая память диалогов),
- локальный файл knowledge_base/chat_history.json.

НЕ трогает:
- knowledge_memory (статья учителя + исторические события),
- news_memory (новости из Telegram).

Запуск:
    python -m scripts.reset_runtime
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from config import settings

# Коллекции, которые нужно сбросить (история тестовых запусков)
RUNTIME_COLLECTIONS = [
    settings.QDRANT_TRADING_COLLECTION,
    settings.QDRANT_PORTFOLIO_COLLECTION,
    settings.QDRANT_CHAT_HISTORY_COLLECTION,
]

# Коллекции, которые НЕ трогаем (база знаний и новости)
PRESERVED_COLLECTIONS = [
    settings.QDRANT_KNOWLEDGE_COLLECTION,
    settings.QDRANT_NEWS_COLLECTION,
]


def _count_points(client: QdrantClient, collection_name: str) -> int:
    """Возвращает количество точек в коллекции или 0, если её нет."""
    try:
        if not client.collection_exists(collection_name=collection_name):
            return 0
        info = client.get_collection(collection_name=collection_name)
        return int(info.points_count or 0)
    except Exception:
        return 0


def main() -> None:
    """Очищает runtime-коллекции и локальную историю диалогов."""
    print("=" * 60)
    print("Сброс тестовой истории (knowledge_memory и news_memory сохраняются)")
    print("=" * 60)

    client = QdrantClient(settings.QDRANT_HOST)

    # 1. Удаляем runtime-коллекции
    print("\nУдаляю runtime-коллекции Qdrant:")
    deleted_count = 0
    for name in RUNTIME_COLLECTIONS:
        if client.collection_exists(collection_name=name):
            points_before = _count_points(client, name)
            client.delete_collection(collection_name=name)
            print(f"  - {name}: удалена ({points_before} точек)")
            deleted_count += 1
        else:
            print(f"  - {name}: не существует, пропуск")

    if deleted_count == 0:
        print("  (нечего удалять — все runtime-коллекции уже пустые)")

    # 2. Удаляем локальный файл истории диалогов
    print("\nУдаляю локальные файлы:")
    if os.path.exists(settings.CHAT_HISTORY_FILE):
        os.remove(settings.CHAT_HISTORY_FILE)
        print(f"  - {settings.CHAT_HISTORY_FILE}: удалён")
    else:
        print(f"  - {settings.CHAT_HISTORY_FILE}: не найден, пропуск")

    # 3. Проверяем, что база знаний не тронута
    print("\nПроверяю сохранность базы знаний:")
    for name in PRESERVED_COLLECTIONS:
        points = _count_points(client, name)
        if points > 0:
            print(f"  - {name}: сохранена ({points} точек)")
        else:
            print(f"  - {name}: ВНИМАНИЕ, коллекция пуста или отсутствует")

    print("\n" + "=" * 60)
    print("Готово. История тестовых запусков очищена.")
    print("База знаний (статья учителя, события, новости) не тронута.")
    print("Бот готов к реальной работе.")
    print("=" * 60)


if __name__ == "__main__":
    main()