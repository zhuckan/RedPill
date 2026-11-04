"""Интерактивная разметка evaluation_dataset.json для IR-метрик.

Для каждого вопроса показывает:
1. Топ-5 чанков из Qdrant (что реально находит поисковик).
2. Все чанки, где встречаются expected_keywords (если топ-5 не покрыл).
Затем просит ввести ID правильных чанков через запятую.

Результат дописывается в evaluation_dataset.json полем "expected_chunk_ids".

Запуск:
    python -m scripts.tag_dataset
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.rag_memory import find_knowledge

DATASET_PATH = "knowledge_base/evaluation_dataset.json"
CHUNKS_JSON_PATH = "knowledge_base/chunks_dump.json"


def _load_chunks() -> list:
    """Загружает дамп всех чанков (для поиска по ключевым словам)."""
    if not os.path.exists(CHUNKS_JSON_PATH):
        print(f"Ошибка: файл {CHUNKS_JSON_PATH} не найден.")
        print("Сначала запусти: python -m scripts.dump_chunks")
        return []
    with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _preview(text: str, length: int = 250) -> str:
    """Возвращает превью текста в одну строку."""
    clean = " ".join(text.split())
    if len(clean) <= length:
        return clean
    return clean[:length] + "..."


def _find_by_keywords(chunks: list, keywords: list) -> list:
    """Ищет чанки, в которых встречаются expected_keywords."""
    if not keywords:
        return []
    found = []
    for ch in chunks:
        text_lower = ch.get("text", "").lower()
        if any(kw.lower() in text_lower for kw in keywords):
            found.append(ch)
    return found


def main() -> None:
    """Основная функция разметки."""
    if not os.path.exists(DATASET_PATH):
        print(f"Ошибка: файл {DATASET_PATH} не найден.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    chunks = _load_chunks()
    if not chunks:
        return

    print("=" * 60)
    print(f"Датасет: {len(dataset)} вопросов. Чанков в базе: {len(chunks)}.")
    print("=" * 60)

    for i, item in enumerate(dataset):
        question = item["question"]
        keywords = item.get("expected_keywords", [])
        existing_ids = item.get("expected_chunk_ids", [])

        print(f"\n[{i + 1}/{len(dataset)}] ВОПРОС: {question}")
        print("-" * 60)

        # 1. Топ-5 из семантического поиска
        print("ТОП-5 из find_knowledge (что находит поисковик):")
        top_docs = find_knowledge(question, limit=5)
        # find_knowledge возвращает payload без id, но можно сопоставить по тексту
        # Проще показать превью, а ID показать из dump_chunks.json
        for j, doc in enumerate(top_docs, 1):
            text = doc.get("text", "")
            # Ищем в chunks тот же текст, чтобы получить ID
            matched_id = next(
                (c["id"] for c in chunks if c.get("text") == text),
                "?",
            )
            print(f"  [{j}] id={matched_id} | {_preview(text)}")

        # 2. Чанки, где встречаются expected_keywords
        if keywords:
            kw_chunks = _find_by_keywords(chunks, keywords)
            kw_chunks = kw_chunks[:5]  # не больше 5
            if kw_chunks:
                print(f"\nПо ключевым словам {keywords}:")
                for ch in kw_chunks:
                    print(f"  id={ch['id']} | {_preview(ch.get('text', ''))}")

        # 3. Уже размеченные — показываем
        if existing_ids:
            print(f"\nТекущие expected_chunk_ids: {existing_ids}")

        # 4. Ввод
        print("\nВведи правильные ID через запятую (или Enter, чтобы пропустить):")
        user_input = input("ID: ").strip()

        if not user_input:
            print("Пропущено.")
            continue

        try:
            ids = [int(x.strip()) for x in user_input.split(",") if x.strip()]
            item["expected_chunk_ids"] = ids
            print(f"Сохранено: {ids}")
        except ValueError:
            print("Ошибка: нужно ввести числа через запятую. Пропущено.")

    # Сохраняем
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("\n" + "=" * 60)
    print(f"Готово. Файл {DATASET_PATH} обновлён.")
    print("=" * 60)


if __name__ == "__main__":
    main()