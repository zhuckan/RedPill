"""Оценка качества поиска в Qdrant по IR-метрикам.

Для каждого вопроса из knowledge_base/evaluation_dataset.json:
1. Ищет топ-K чанков через client.query_points.
2. Сравнивает найденные ID с expected_chunk_ids.
3. Считает Precision@K, Recall@K, MRR, nDCG@K.
4. Отправляет средние скоры в Langfuse.

Вопросы без expected_chunk_ids пропускаются (например, если нужный факт
отсутствует в базе знаний).

Запуск:
    python -m scripts.evaluate_retrieval
"""
import json
import math
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Set

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langfuse import Langfuse

from config import settings
from core.rag_memory import _embed, client

DATASET_PATH = "knowledge_base/evaluation_dataset.json"

# Значения K, для которых считаем метрики
K_VALUES = [1, 3, 5, 10]

# Инициализация клиента Langfuse
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)


def retrieve_ids(query: str, limit: int = 10) -> List[int]:
    """Возвращает список ID найденных чанков (в порядке ранжирования)."""
    result = client.query_points(
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
        query=_embed(query),
        limit=limit,
    )
    return [hit.id for hit in result.points]


def precision_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Precision@K: доля релевантных среди топ-K найденных."""
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for i in top_k if i in relevant)
    return hits / len(top_k)


def recall_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """Recall@K: доля релевантных, попавших в топ-K."""
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def mrr(retrieved: List[int], relevant: Set[int]) -> float:
    """MRR: 1 / позиция первого релевантного (1-based)."""
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg(relevances: List[int]) -> float:
    """Discounted Cumulative Gain (бинарная релевантность)."""
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(retrieved: List[int], relevant: Set[int], k: int) -> float:
    """nDCG@K: DCG / IDCG (нормализованный)."""
    top_k = retrieved[:k]
    relevances = [1 if doc_id in relevant else 0 for doc_id in top_k]

    actual_dcg = dcg(relevances)

    # Идеальное ранжирование: все релевантные в начале
    ideal_relevances = [1] * min(len(relevant), k) + [0] * max(0, k - len(relevant))
    ideal_dcg = dcg(ideal_relevances)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def main() -> None:
    """Основная функция: прогон датасета и подсчёт метрик."""
    if not os.path.exists(DATASET_PATH):
        print(f"Ошибка: файл {DATASET_PATH} не найден.")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 70)
    print(f"IR-метрики для {len(dataset)} вопросов.")
    print(f"K = {K_VALUES}")
    print("=" * 70)

    # Аккумуляторы
    sums: Dict[str, Dict[int, float]] = {
        "precision": {k: 0.0 for k in K_VALUES},
        "recall": {k: 0.0 for k in K_VALUES},
        "ndcg": {k: 0.0 for k in K_VALUES},
    }
    sum_mrr = 0.0
    evaluated = 0
    skipped = 0

    for i, item in enumerate(dataset, 1):
        question = item["question"]
        expected = item.get("expected_chunk_ids", [])

        if not expected:
            print(f"\n[{i}] ПРОПУСК: {question[:60]}... (нет expected_chunk_ids)")
            skipped += 1
            continue

        relevant = set(expected)
        print(f"\n[{i}] {question[:60]}...")
        print(f"    Эталонные ID: {sorted(relevant)}")

        retrieved = retrieve_ids(question, limit=max(K_VALUES))
        print(f"    Найдено (топ-10): {retrieved}")

        for k in K_VALUES:
            p = precision_at_k(retrieved, relevant, k)
            r = recall_at_k(retrieved, relevant, k)
            n = ndcg_at_k(retrieved, relevant, k)

            sums["precision"][k] += p
            sums["recall"][k] += r
            sums["ndcg"][k] += n

            print(f"    K={k:>2}: P={p:.2f}  R={r:.2f}  nDCG={n:.2f}")

        m = mrr(retrieved, relevant)
        sum_mrr += m
        print(f"    MRR: {m:.2f}")
        evaluated += 1

    if evaluated == 0:
        print("\nНет вопросов с разметкой. Нечего считать.")
        return

    # Средние значения
    print("\n" + "=" * 70)
    print(f"СРЕДНИЕ ЗНАЧЕНИЯ (по {evaluated} вопросам)")
    print("=" * 70)

    avg_precision = {k: sums["precision"][k] / evaluated for k in K_VALUES}
    avg_recall = {k: sums["recall"][k] / evaluated for k in K_VALUES}
    avg_ndcg = {k: sums["ndcg"][k] / evaluated for k in K_VALUES}
    avg_mrr = sum_mrr / evaluated

    print(f"\nPrecision@K: {avg_precision}")
    print(f"Recall@K:    {avg_recall}")
    print(f"nDCG@K:      {avg_ndcg}")
    print(f"MRR:         {avg_mrr:.4f}")

    # --- Отправка в Langfuse ---
    print("\n" + "=" * 70)
    print("Отправка скоров в Langfuse...")
    trace_id = f"retrieval_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    try:
        create_score = getattr(langfuse, "create_score", getattr(langfuse, "score", None))
        if create_score:
            for k in K_VALUES:
                create_score(trace_id=trace_id, name=f"Precision@{k}", value=avg_precision[k])
                create_score(trace_id=trace_id, name=f"Recall@{k}", value=avg_recall[k])
                create_score(trace_id=trace_id, name=f"nDCG@{k}", value=avg_ndcg[k])
            create_score(trace_id=trace_id, name="MRR", value=avg_mrr)
            langfuse.flush()
            print("Скоры отправлены в Langfuse (Scores).")
        else:
            print("В текущей версии Langfuse нет метода create_score.")
    except Exception as e:
        print(f"Ошибка отправки в Langfuse: {e}")


if __name__ == "__main__":
    main()