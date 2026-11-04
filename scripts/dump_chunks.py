"""Выгрузка всех чанков из knowledge_memory для подготовки IR-датасета.

Создаёт два файла в knowledge_base/:
- chunks_dump.json  — полный дамп (id, text, source, date) для программной обработки
- chunks_dump.md    — человекочитаемый список для быстрого просмотра

Запуск:
    python -m scripts.dump_chunks
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qdrant_client import QdrantClient

from config import settings

# Папка для выгрузки
OUTPUT_DIR = "knowledge_base"
JSON_PATH = os.path.join(OUTPUT_DIR, "chunks_dump.json")
MD_PATH = os.path.join(OUTPUT_DIR, "chunks_dump.md")


def main() -> None:
    """Выгружает все чанки из Qdrant и сохраняет в JSON + Markdown."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = QdrantClient(settings.QDRANT_HOST)

    print("Загружаю все точки из knowledge_memory...")
    records, _ = client.scroll(
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
        limit=10000,
        with_payload=True,
    )

    if not records:
        print("Ошибка: коллекция knowledge_memory пуста.")
        return

    print(f"Получено {len(records)} чанков. Сортирую по id...")
    records.sort(key=lambda r: r.id)

    # JSON для программной обработки
    dump_data = []
    for rec in records:
        payload = rec.payload or {}
        dump_data.append(
            {
                "id": rec.id,
                "source": payload.get("source", ""),
                "date": payload.get("date", ""),
                "text": payload.get("text", ""),
            }
        )

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(dump_data, f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {JSON_PATH}")

    # Markdown для быстрого просмотра
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("# Дамп чанков из knowledge_memory\n\n")
        f.write(f"Всего чанков: **{len(dump_data)}**\n\n")
        f.write("---\n\n")

        for item in dump_data:
            f.write(f"## ID: `{item['id']}`\n")
            f.write(f"- **Source:** `{item['source']}`\n")
            f.write(f"- **Date:** {item['date']}\n")
            f.write(f"- **Preview:** {item['text'][:300]}...\n\n")
            f.write("---\n\n")

    print(f"Markdown сохранён: {MD_PATH}")
    print("Готово. Открой chunks_dump.md для просмотра.")


if __name__ == "__main__":
    main()