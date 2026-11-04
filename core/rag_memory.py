import hashlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from config import settings

# Инициализируем клиент Qdrant и модель для эмбеддингов при импорте модуля
client = QdrantClient(settings.QDRANT_HOST)
embedding_model = TextEmbedding(model_name="intfloat/multilingual-e5-large")

# Кеш дайджеста исторических событий (см. get_all_events_digest)
_EVENTS_DIGEST_CACHE: Optional[str] = None


def _init_collection(name: str) -> None:
    """
    Создает коллекцию в Qdrant, если она еще не существует.
    Размерность вектора берется из настроек.
    """
    if not client.collection_exists(collection_name=name):
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


def init_db() -> None:
    """Инициализирует все необходимые коллекции для работы агента."""
    _init_collection(settings.QDRANT_TRADING_COLLECTION)
    _init_collection(settings.QDRANT_KNOWLEDGE_COLLECTION)
    _init_collection(settings.QDRANT_PORTFOLIO_COLLECTION)
    _init_collection(settings.QDRANT_CHAT_HISTORY_COLLECTION)


def init_news_db() -> None:
    """Инициализирует коллекцию для новостей."""
    _init_collection(settings.QDRANT_NEWS_COLLECTION)


def _embed(text: str) -> List[float]:
    """
    Превращает текст в вектор (эмбеддинг) с помощью локальной модели.
    Возвращает список чисел с плавающей точкой.
    """
    return list(embedding_model.embed([text]))[0].tolist()


# --- Работа с прогнозами (Trading Journal) ---

def save_prediction(
    market_snapshot: str,
    prediction: str,
    reasoning: str,
    portfolio_state: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> int:
    """Сохраняет прогноз агента в историю.

    Args:
        market_snapshot: Снимок рынка (текст).
        prediction: Краткая суть прогноза (prediction_summary).
        reasoning: Развернутое обоснование.
        portfolio_state: Данные портфеля на момент прогноза.
        trace_id: ID трейса в Langfuse (для привязки RAG-скоров).
            Если None — поле в payload не добавляется.

    Returns:
        Уникальный ID сохраненного прогноза.
    """
    full_text = f"Рынок: {market_snapshot}. Прогноз: {prediction}. Обоснование: {reasoning}"
    vector = _embed(full_text)

    point_id = int(datetime.now().timestamp() * 1000)

    payload = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "market_snapshot": market_snapshot,
        "prediction": prediction,
        "reasoning": reasoning,
        "portfolio": portfolio_state,
        "verdict": "Ожидает проверки",
    }
    if trace_id:
        payload["trace_id"] = trace_id

    client.upsert(
        collection_name=settings.QDRANT_TRADING_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        ],
    )
    return point_id


def get_last_trace_id() -> Optional[str]:
    """Возвращает trace_id последнего прогноза (для привязки RAG-скоров).

    Используется в evaluate_rag.py: находит последний прогноз в Qdrant
    и достает из него trace_id, чтобы отправить RAG-скоры в тот же трейс.
    """
    try:
        records, _ = client.scroll(
            collection_name=settings.QDRANT_TRADING_COLLECTION,
            limit=10,
            with_payload=True,
        )
        if not records:
            return None

        records.sort(key=lambda r: r.payload.get("date", ""), reverse=True)

        for rec in records:
            trace_id = rec.payload.get("trace_id")
            if trace_id:
                return trace_id
        return None
    except Exception as e:
        print(f"Ошибка получения последнего trace_id: {e}")
        return None


def find_similar_situations(query: str, limit: int = 2) -> List[Dict[str, Any]]:
    """Ищет похожие рыночные ситуации в истории прогнозов."""
    result = client.query_points(
        collection_name=settings.QDRANT_TRADING_COLLECTION,
        query=_embed(query),
        limit=limit,
    )
    return [hit.payload for hit in result.points]


def update_prediction_outcome(point_id: int, actual_outcome: str, verdict: str) -> bool:
    """Обновляет прогноз фактическим исходом и вердиктом."""
    try:
        records = client.retrieve(
            collection_name=settings.QDRANT_TRADING_COLLECTION,
            ids=[point_id],
        )
        if not records:
            return False

        payload = records[0].payload
        payload["actual_outcome"] = actual_outcome
        payload["verdict"] = verdict
        payload["closed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")

        client.set_payload(
            collection_name=settings.QDRANT_TRADING_COLLECTION,
            payload=payload,
            points=[point_id],
        )
        return True
    except Exception as e:
        print(f"Ошибка обновления прогноза: {e}")
        return False


# --- Работа с базой знаний (Статьи, события) ---

def find_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """Ищет информацию в базе знаний (статья учителя, исторические кейсы)."""
    result = client.query_points(
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
        query=_embed(query),
        limit=limit,
    )
    return [hit.payload for hit in result.points]


def get_all_events_digest(lesson_max_chars: int = 260) -> str:
    """Возвращает компактный дайджест ВСЕХ исторических событий для промпта.

    В отличие от find_knowledge (топ-5 по семантике), эта функция вытаскивает
    весь набор кейсов (source=historical_events) и отдает агенту полный список.
    Агент сам выбирает 1-2 наиболее релевантных к текущей ситуации и объясняет,
    чем они похожи / отличаются.

    Args:
        lesson_max_chars: Максимальная длина обрезанного поля lesson.
            Обрезка по границе слова + '...', чтобы не рвать мысль.

    Returns:
        Многострочный текст вида:
            [1] 2020-03-12 · Black Thursday (COVID-19 Crash) · Crash / Liquidation Cascade
                Урок: ...
            [2] ...
        Если событий нет — возвращает 'Исторических событий в базе не найдено.'.
        Результат кешируется в памяти процесса до вызова invalidate_events_digest_cache().
    """
    global _EVENTS_DIGEST_CACHE
    if _EVENTS_DIGEST_CACHE is not None:
        return _EVENTS_DIGEST_CACHE

    try:
        records, _ = client.scroll(
            collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
            limit=10000,
            with_payload=True,
        )
    except Exception as e:
        print(f"Ошибка получения дайджеста событий: {e}")
        return "Исторических событий в базе не найдено."

    events: List[Dict[str, Any]] = []
    for rec in records:
        payload = rec.payload or {}
        if payload.get("source") != "historical_events":
            continue
        events.append(payload)

    if not events:
        _EVENTS_DIGEST_CACHE = "Исторических событий в базе не найдено."
        return _EVENTS_DIGEST_CACHE

    # Хронологический порядок — удобнее читать и ссылаться
    events.sort(key=lambda x: x.get("date", ""))

    lines: List[str] = []
    for idx, ev in enumerate(events, start=1):
        name = ev.get("event_name", "Без названия")
        date = ev.get("date", "")
        etype = ev.get("type", "")
        lesson = (ev.get("lesson") or "").strip()

        # Обрезаем lesson по границе слова, чтобы не рвать мысль
        if len(lesson) > lesson_max_chars:
            cut = lesson[:lesson_max_chars].rstrip()
            if " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            lesson = f"{cut}..."

        header = f"[{idx}] {date} · {name}"
        if etype:
            header += f" · {etype}"
        lines.append(header)
        if lesson:
            lines.append(f"    Урок: {lesson}")
        lines.append("")

    _EVENTS_DIGEST_CACHE = "\n".join(lines).rstrip()
    return _EVENTS_DIGEST_CACHE


def invalidate_events_digest_cache() -> None:
    """Сбрасывает кеш дайджеста исторических событий.

    Вызывать после scripts.load_events, если бот уже запущен
    (или просто перезапустить бота — кеш живёт в памяти процесса).
    """
    global _EVENTS_DIGEST_CACHE
    _EVENTS_DIGEST_CACHE = None


# --- Работа с новостями ---

def save_news_to_qdrant(date_str: str, text: str) -> None:
    """Сохраняет новость в базу."""
    full_text = f"Дата: {date_str}. Новость: {text}"
    stable_id = hashlib.md5(full_text.encode("utf-8")).hexdigest()

    client.upsert(
        collection_name=settings.QDRANT_NEWS_COLLECTION,
        points=[
            PointStruct(
                id=stable_id,
                vector=_embed(full_text),
                payload={"date": date_str, "text": text},
            )
        ],
    )


def search_relevant_news(query: str, limit: int = 7) -> List[Dict[str, Any]]:
    """Ищет релевантные новости по смыслу для передачи агенту."""
    result = client.query_points(
        collection_name=settings.QDRANT_NEWS_COLLECTION,
        query=_embed(query),
        limit=limit,
    )
    return [hit.payload for hit in result.points]


def cleanup_old_news(
    days: int = settings.NEWS_TTL_DAYS,
    max_points: int = settings.NEWS_MAX_POINTS,
) -> int:
    """Удаляет устаревшие новости."""
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

    try:
        records, _ = client.scroll(
            collection_name=settings.QDRANT_NEWS_COLLECTION,
            limit=50000,
            with_payload=True,
        )
        if not records:
            return 0

        to_delete = []
        kept_records = []

        for rec in records:
            date_str = (rec.payload.get("date", "") or "")[:10]
            if date_str and date_str < cutoff:
                to_delete.append(rec.id)
            else:
                kept_records.append(rec)

        if len(kept_records) > max_points:
            kept_records.sort(key=lambda r: (r.payload.get("date", "") or ""))
            overflow = kept_records[: len(kept_records) - max_points]
            to_delete.extend([r.id for r in overflow])

        if to_delete:
            client.delete(
                collection_name=settings.QDRANT_NEWS_COLLECTION,
                points_selector=to_delete,
            )
        return len(to_delete)
    except Exception as e:
        print(f"Ошибка очистки новостей: {e}")
        return 0


# --- Работа с историей портфеля ---

def save_portfolio_snapshot(portfolio_data: Dict[str, Any]) -> None:
    """Сохраняет текущий снимок портфеля в историю."""
    text = f"Портфель на {datetime.now().strftime('%Y-%m-%d')}: {json.dumps(portfolio_data, ensure_ascii=False)}"
    point_id = int(datetime.now().timestamp() * 1000)

    client.upsert(
        collection_name=settings.QDRANT_PORTFOLIO_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=_embed(text),
                payload={
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "portfolio_data": portfolio_data,
                    "total_balance": portfolio_data.get("total_balance", 0),
                    "total_pnl_pct": portfolio_data.get("total_pnl_pct", 0),
                },
            )
        ],
    )


def get_portfolio_history(limit: int = 5) -> List[Dict[str, Any]]:
    """Возвращает последние снимки портфеля (от новых к старым)."""
    try:
        records, _ = client.scroll(
            collection_name=settings.QDRANT_PORTFOLIO_COLLECTION,
            limit=limit * 3,
            with_payload=True,
        )
        payloads = [r.payload for r in records]
        payloads.sort(key=lambda x: x.get("date", ""), reverse=True)
        return payloads[:limit]
    except Exception as e:
        print(f"Ошибка получения истории портфеля: {e}")
        return []


# --- Работа с семантической памятью диалогов ---

def save_chat_exchange_to_qdrant(
    user_message: str,
    assistant_message: str,
    session_id: str = "default",
) -> None:
    """Сохраняет пару «вопрос пользователя + ответ агента» в семантическую память."""
    if not user_message.strip():
        return

    vector = _embed(user_message)
    exchange_id = str(uuid.uuid4())

    client.upsert(
        collection_name=settings.QDRANT_CHAT_HISTORY_COLLECTION,
        points=[
            PointStruct(
                id=exchange_id,
                vector=vector,
                payload={
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "session_id": session_id,
                    "user_message": user_message,
                    "assistant_message": assistant_message,
                },
            )
        ],
    )


def search_past_chat(query: str, limit: int = settings.CHAT_HISTORY_SEARCH_LIMIT) -> List[Dict[str, Any]]:
    """Ищет релевантные прошлые обсуждения в семантической памяти."""
    if not query.strip():
        return []

    try:
        result = client.query_points(
            collection_name=settings.QDRANT_CHAT_HISTORY_COLLECTION,
            query=_embed(query),
            limit=limit,
        )
        return [hit.payload for hit in result.points]
    except Exception as e:
        print(f"Ошибка поиска в истории диалогов: {e}")
        return []