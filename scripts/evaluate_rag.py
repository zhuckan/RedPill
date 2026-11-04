"""Оценка качества RAG через LLM-as-a-judge + привязка скоров к реальному трейсу.

Скрипт прогоняет тестовый датасет, для каждого вопроса:
1. Находит контекст в Qdrant.
2. Генерирует ответ через Gemini.
3. Оценивает по трём метрикам (Faithfulness, Answer Relevancy, Context Precision).
4. Отправляет скоры в Langfuse НА ТОТ ЖЕ ТРЕЙС, где бот делал последний анализ
   (берётся через get_last_trace_id()).

Запуск:
    python -m scripts.evaluate_rag
"""
import asyncio
import json
import os
import re
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage
from langfuse import Langfuse

from core.ai_agent import _extract_text, analysis_llm
from core.rag_memory import find_knowledge, get_last_trace_id

# Инициализация Langfuse
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)


def judge_metric(
    metric_name: str,
    instruction: str,
    question: str,
    context: str,
    answer: str,
) -> float:
    """LLM-as-a-judge: оценивает ответ от 0.0 до 1.0."""
    prompt = f"""Ты — строгий эксперт по оценке качества RAG систем.
Твоя задача — оценить метрику {metric_name} по шкале от 0.0 до 1.0.

{instruction}

ВОПРОС: {question}
КОНТЕКСТ: {context}
ОТВЕТ: {answer}

ВЫВОД: Верни ТОЛЬКО одно число с плавающей точкой (например: 0.85 или 1.0). Без текста, без объяснений."""

    try:
        resp = analysis_llm.invoke([HumanMessage(content=prompt)])
        text = _extract_text(resp.content)
        match = re.search(r"0\.\d+|1\.0|0|1", text)
        if match:
            return float(match.group(0))
        return 0.0
    except Exception as e:
        print(f"Ошибка оценки {metric_name}: {e}")
        return 0.0


async def run_evaluation() -> None:
    """Основная функция оценки RAG."""
    print("Запуск оценки RAG (LLM-as-a-judge)...")

    dataset_path = "knowledge_base/evaluation_dataset.json"
    if not os.path.exists(dataset_path):
        print(f"Ошибка: файл {dataset_path} не найден.")
        return

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset_json = json.load(f)

    results = []
    total_faithfulness = 0.0
    total_relevancy = 0.0
    total_precision = 0.0

    for item in dataset_json:
        q = item["question"]
        print(f"\nВопрос: {q}")

        retrieved_docs = find_knowledge(q, limit=3)
        context_str = "\n\n".join([doc["text"] for doc in retrieved_docs])

        gen_prompt = (
            f"Ответь на вопрос, используя ТОЛЬКО контекст. "
            f"Если ответа нет — напиши 'Нет ответа'.\n\n"
            f"КОНТЕКСТ:\n{context_str}\n\n"
            f"ВОПРОС: {q}"
        )
        resp = analysis_llm.invoke([HumanMessage(content=gen_prompt)])
        answer = _extract_text(resp.content)

        print("  Оцениваю Faithfulness...")
        faithfulness = judge_metric(
            "Faithfulness",
            "Оцени, насколько ответ подтверждается контекстом. 1.0 - всё подтверждено, 0.0 - галлюцинация.",
            q, context_str, answer,
        )

        print("  Оцениваю Answer Relevancy...")
        relevancy = judge_metric(
            "Answer Relevancy",
            "Оцени, насколько ответ по смыслу соответствует вопросу. 1.0 - прямо отвечает, 0.0 - не по теме.",
            q, context_str, answer,
        )

        print("  Оцениваю Context Precision...")
        precision = judge_metric(
            "Context Precision",
            "Оцени, насколько найденный контекст полезен для ответа. 1.0 - идеален, 0.0 - мусор.",
            q, context_str, answer,
        )

        total_faithfulness += faithfulness
        total_relevancy += relevancy
        total_precision += precision

        results.append({
            "question": q,
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": precision,
        })

    n = len(results)
    avg_faithfulness = total_faithfulness / n
    avg_relevancy = total_relevancy / n
    avg_precision = total_precision / n

    print("\nРезультаты:")
    for r in results:
        print(f"- {r['question'][:50]}...")
        print(
            f"  Faithfulness: {r['faithfulness']:.2f} | "
            f"Relevancy: {r['answer_relevancy']:.2f} | "
            f"Precision: {r['context_precision']:.2f}"
        )

    print(f"\nСредние показатели:")
    print(f"Faithfulness: {avg_faithfulness:.2f}")
    print(f"Answer Relevancy: {avg_relevancy:.2f}")
    print(f"Context Precision: {avg_precision:.2f}")

    # --- Привязка к реальному trace из Qdrant ---
    print("\nИщу последний trace_id в Qdrant (trading_memory)...")
    real_trace_id = get_last_trace_id()

    if not real_trace_id:
        print("Не нашел trace_id в Qdrant (в прогнозах нет поля trace_id).")
        print("Использую фолбэк с фиктивным trace_id.")
        real_trace_id = f"eval_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print(f"Отправляю скоры в trace: {real_trace_id}")

    try:
        create_score = getattr(langfuse, "create_score", getattr(langfuse, "score", None))
        if create_score:
            create_score(trace_id=real_trace_id, name="RAG_Faithfulness", value=avg_faithfulness)
            create_score(trace_id=real_trace_id, name="RAG_Answer_Relevancy", value=avg_relevancy)
            create_score(trace_id=real_trace_id, name="RAG_Context_Precision", value=avg_precision)
            langfuse.flush()
            print("Скоры привязаны к реальному трейсу.")
            print(f"Открой Langfuse → Tracing → фильтр по trace_id: {real_trace_id}")
        else:
            print("В текущей версии Langfuse нет метода create_score.")
    except Exception as e:
        print(f"Ошибка отправки в Langfuse: {e}")


if __name__ == "__main__":
    asyncio.run(run_evaluation())