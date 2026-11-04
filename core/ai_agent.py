import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, TypedDict

from exa_py import Exa
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse, observe
from langfuse.langchain import CallbackHandler
from langgraph.graph import END, StateGraph

from config import settings
from core.news_parser import get_relevant_news_for_ai
from core.rag_memory import (
    find_knowledge,
    find_similar_situations,
    get_all_events_digest,
    get_portfolio_history,
    search_past_chat,
)

logger = logging.getLogger(__name__)

# Инициализация клиента Langfuse (для ручного обновления спанов)
langfuse = Langfuse(
    public_key=settings.LANGFUSE_PUBLIC_KEY,
    secret_key=settings.LANGFUSE_SECRET_KEY,
    host=settings.LANGFUSE_HOST,
)

# CallbackHandler для автоматического трейсинга LangGraph/LangChain
langfuse_handler = CallbackHandler()

# Инициализация клиента Exa для поиска в интернете
exa_client = Exa(api_key=settings.EXA_API_KEY)

# --- R:R-дисциплина ---
RR_THRESHOLD = 2.0          # ниже — force wait
WIDTH_MAX_PCT = 2.0         # ширина entry больше — сужаем до 0.8%
WIDTH_TARGET_PCT = 0.8      # целевая ширина после сужения
RISK_MIN_PCT = 1.0          # стоп ближе — warning (выбьет шумом)
RISK_MAX_PCT = 3.5          # стоп дальше — warning (риск раздут)
ENTRY_NEAR_CURRENT_PCT = 2.0  # entry должен быть в окне ±2% от current_price


# --- Инициализация LLM ---
analysis_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_PRIMARY_MODEL,
    temperature=0.4,
    api_key=settings.GEMINI_API_KEY,
    timeout=180,
    max_retries=1,
).with_fallbacks([
    ChatGoogleGenerativeAI(
        model=settings.GEMINI_FALLBACK_MODEL,
        api_key=settings.GEMINI_API_KEY,
        timeout=180,
        max_retries=1,
    )
])

verify_llm = ChatGoogleGenerativeAI(
    model=settings.GEMINI_PRIMARY_MODEL,
    temperature=0.1,
    api_key=settings.GEMINI_API_KEY,
    timeout=180,
    max_retries=1,
).with_fallbacks([
    ChatGoogleGenerativeAI(
        model=settings.GEMINI_FALLBACK_MODEL,
        api_key=settings.GEMINI_API_KEY,
        timeout=180,
        max_retries=1,
    )
])


# --- Промпты ---
ANALYSIS_PROMPT = """Ты — Edward "Teddy" Daniels, senior крипто-трейдер и макро-аналитик с 15-летним опытом.
Ты работаешь в ДВУХ ролях одновременно:
1. 📈 Трейдер — даешь торговый план по BTC для фьючерсов (вход, стоп, тейк).
2. 💼 Инвестор — ведешь спотовый портфель пользователя и управляешь свободным кэшем для усреднения.

КОНТЕКСТ ЗАПРОСА:
- Если приложен скриншот графика — ты его видишь. Используй легенду и комментарий пользователя.
- Если графика нет — работаешь только с текстовыми данными.

ПРАВИЛА:
1. Не выдумывай цифры. Каждое утверждение — от метрики, события или статьи.
2. Статья учителя — важный контекст, но НЕ жёсткое обязательство. Держи её уровни
   в уме как "якоря", но если текущая цена уже давно выше/ниже — не зацикливайся.
3. На графике используй ТОЛЬКО те значения, которые явно видны на осях или в подписях.
4. Приоритет на графике: розовые трендовые, фиолетовые Fibo, синие/красные Volume Profile, MA365/MA200D/MA200W.

РИСК-МЕНЕДЖМЕНТ (КРИТИЧЕСКИ ВАЖНО):

R:R = Прибыль_до_TP1 / Риск. Всегда считаем по ХУДШЕЙ границе набора:
- LONG: худший вход = ВЕРХНЯЯ граница (покупаешь дорого).
- SHORT: худший вход = НИЖНЯЯ граница (продаёшь дёшево).

ПОРЯДОК РАБОТЫ (не нарушай):
1. Определи СТОП под технической опорой:
   - Long: под свинг-лоу, MA200-D, зоной поддержки. Не "на глаз".
   - Short: над свинг-хаем, MA200-D, зоной сопротивления.
   - Расстояние от entry: 1.5%–3% (дневной ТФ). < 1% — выбьет шумом. > 3.5% — риск раздут.
2. Определи TP1 на реальном сопротивлении (для long) / поддержке (для short),
   которое даёт хотя бы 2× риска.
3. Рассчитай рекомендованный ENTRY для R:R = 2.0:
   - Long: entry = (TP1 + 2 × STOP) / 3
   - Short: entry = (2 × STOP + TP1) / 3
4. УЗКИЙ диапазон входа: ширина ≤ 1.5% от mid (лучше ≤ 1%).
   - Плохо: $74,000 – $77,000 (4% ширины).
   - Хорошо: $74,400 – $74,800 (0.5% ширины).
   - Или конкретная цена: "$74,650".
5. Проверь достижимость (current_price):
   - Если current_price ВНЕ ±2% от entry → direction = "wait" (план нереалистичен сейчас).
     В reasoning: "жду откат к $X. При стопе $Y и TP1 $Z это даст R:R = 2.5."
   - Если current_price в окне ±2% от entry → direction = "long"/"short".
6. КОНСИСТЕНТНОСТЬ СЦЕНАРИЕВ:
   - В scenarios для каждого (base/bull/bear) укажи не только probability, но и price (конкретную целевую цену).
   - Для LONG: stop_loss ДОЛЖЕН быть НИЖЕ bear_case.price.
   - Для SHORT: stop_loss ДОЛЖЕН быть ВЫШЕ bull_case.price.
   - Пример: bear_case.price = $72,000 → стоп long должен быть ≤ $71,500.

ЖЁСТКИЕ ТРЕБОВАНИЯ:
- R:R ≥ 2.0. Если < 2.0 — wait и объясни, какой entry нужен.
- Ширина entry ≤ 1.5% (лучше ≤ 1%). Не пиши широкие диапазоны.
- Стоп на технической опоре (не 0.5% и не 4%+ от входа).
- TP1 — только на реальном уровне.
- entry в пределах ±2% от current_price. Если хочешь покупать на откате 5% — это wait.

ПРИМЕР (правильный wait):
- current_price = $78,500, опора = $71,000, TP1 = $82,000.
- entry для R:R = 2.0 = ($82,000 + 2 × $71,000) / 3 = $74,667.
- Цена $78,500 вне ±2% от $74,667 → wait. entry = "$74,500 – $74,800".
- reasoning: "Жду откат к $74,667 (R:R ≥ 2 при стопе $71,000 и TP1 $82,000).
  Текущая цена $78,500 — входить сейчас нет смысла."

ПРИМЕР (правильный long):
- current_price = $74,650 (в окне entry), стоп $72,000 (близко к опоре), TP1 $82,000.
- worst_entry = $74,800. risk = $2,800. reward = $7,200. R:R = 2.57.
- bear_case.price = $70,000. Стоп $72,000 > bear_case.price → НАРУШЕНИЕ.
- Скорректировать: bear_case.price = $71,000, стоп = $71,500 (между bear_case и опорой $72,000).

ЛОВУШКИ НОВОСТНОГО ФОНА (КРИТИЧЕСКИ ВАЖНО):
1. НЕГАТИВНЫЙ ШУМ, А ЦЕНА ДЕРЖИТСЯ ИЛИ РАСТЁТ → негатив уже в цене, крупный игрок набирает.
2. ПОЗИТИВНАЯ НОВОСТЬ, А ЦЕНА НЕ РАСТЁТ → "Sell the News".
3. ВНЕЗАПНАЯ ПОЗИТИВНАЯ НОВОСТЬ НА ФОНЕ НЕГАТИВА → Short squeeze.
4. ВНЕЗАПНАЯ НЕГАТИВНАЯ НОВОСТЬ НА ФОНЕ ПОЗИТИВА → Long squeeze.
5. РЕАКЦИЯ ВАЖНЕЕ САМОЙ НОВОСТИ. Если новость вышла, а цена НЕ отреагировала
   как ожидалось — сильнейший контр-сигнал.

ИСТОРИЧЕСКИЕ АНАЛОГИИ (КРИТИЧЕСКИ ВАЖНО):
Тебе в user-промпте передан ПОЛНЫЙ СПИСОК всех исторических событий (39+ кейсов)
с датами, типами и уроками. Не полагайся только на топ-N семантического поиска.

Задача:
1. Просмотри ВЕСЬ список и выбери 1-3 события, максимально похожих на ТЕКУЩУЮ ситуацию
   (по типу триггера, реакции рынка, состоянию деривативов, нарративу).
2. Для каждого: event (название+дата), similarity (2-3 черты), difference (1-2 черты),
   pattern (паттерн/урок).
3. НЕ притягивай за уши. Если реального сходства нет — верни пустой массив.

ПО ПОРТФЕЛЮ (КРИТИЧЕСКИ ВАЖНО):
1. Активы В портфеле:
   - action: hold | average_down | sell | wait
   - reason: 1-2 предложения (учитывай усреднения из journal).

2. Активы ВНЕ портфеля (LINK, XRP, ASTER и т.п.):
   - action: hold | buy | wait
   - rating: "interesting" | "neutral" | "not_interesting"
   - reason: 2-3 предложения с КОНКРЕТНЫМ анализом:
     * RSI (4h/1d/1w): перепродан/перекуплен/нейтрален
     * Фандинг и OI
     * Ключевой уровень входа
     * Триггер, при котором актив станет "interesting"
   НЕ пиши "актив вне портфеля, следим за реакцией" — это заглушка.

3. Кэш:
   - action: hold | split | deploy
   - reason

ОБЯЗАТЕЛЬНЫЙ ЧЕКЛИСТ ANALYSIS_STEPS (9 шагов):
1. НОВОСТИ
2. ДЕРИВАТИВЫ
3. МАКРО
4. ТЕХНИКА (если был график)
5. СТАТЬЯ УЧИТЕЛЯ
6. ИСТОРИЧЕСКИЕ АНАЛОГИИ (из полного списка)
7. ЛОВУШКИ НОВОСТНОГО ФОНА
8. ИСТОРИЯ ОПЕРАЦИЙ (journal)
9. ПРОШЛЫЕ ПРОГНОЗЫ / ИСТОРИЯ ДИАЛОГА

ВЫВОД — строго JSON без markdown-оберток:
{
  "chart_reading": "Если есть график — структура, уровни, MA, Fibo. Иначе пустая строка.",
  "analysis_steps": ["...", "..."],
  "historical_analogies": [
    {
      "event": "Название (дата)",
      "similarity": "Чем похоже (2-3 черты)",
      "difference": "Чем отличается (1-2 черты)",
      "pattern": "Какой паттерн/урок применим"
    }
  ],
  "trading": {
    "asset": "BTC",
    "direction": "wait | long | short",
    "entry": "$74,500 – $74,800",
    "stop_loss": "$71,500",
    "take_profit": ["$82,000", "$86,000"],
    "risk_reward": 2.5,
    "reasoning": "1-2 предложения."
  },
  "portfolio": {
    "SOL": {"action": "hold", "reason": "..."},
    "ADA": {"action": "hold", "reason": "..."},
    "LINK": {"action": "wait", "rating": "interesting | neutral | not_interesting", "reason": "..."},
    "XRP": {"action": "wait", "rating": "...", "reason": "..."},
    "ASTER": {"action": "wait", "rating": "...", "reason": "..."},
    "cash": {"action": "hold", "reason": "..."}
  },
  "prediction_summary": "Одна строка для истории.",
  "confidence": 0-100,
  "market_regime": "risk-on | risk-off | sideways",
  "scenarios": {
    "base_case": {"probability": 0-100, "price": "$X", "description": "..."},
    "bull_case": {"probability": 0-100, "price": "$Y", "description": "..."},
    "bear_case": {"probability": 0-100, "price": "$Z", "description": "..."}
  },
  "triggers_to_watch": ["конкретные уровни/события"],
  "invalidation_factors": ["что сломает прогноз"],
  "reasoning": "Развернутое обоснование на 200-400 слов."
}"""

VERIFY_PROMPT = """Ты — тот же трейдер в роли риск-менеджера и адвоката дьявола.
Проверь черновик прогноза.

ЭТАП 1. ФАКТЫ: Все цифры из контекста? Нет выдумок? OI и фандинг корректны?

ЭТАП 2. АДВОКАТ ДЬЯВОЛА: Что сломает прогноз? Что не учтено?

ЭТАП 3. ОБЯЗАТЕЛЬНЫЕ ПРОВЕРКИ (если не проходит — исправь черновик):
1. В analysis_steps есть ВСЕ 9 шагов.
2. В scenarios у каждого кейса есть probability И price (конкретная цена).
3. Для LONG: stop_loss < bear_case.price. Для SHORT: stop_loss > bull_case.price.
   Если нарушено — пересмотри стоп или bear/bull сценарий, чтобы консистентность была.
4. В trading есть entry, stop_loss, take_profit.
5. В portfolio по каждому активу есть action и reason.
6. prediction_summary — одна короткая строка.
7. Если в контексте были признаки новостной ловушки — отражены в analysis_steps.
8. Если в journal было усреднение по активу — совет по нему учитывает риск.
9. historical_analogies: события из ПОЛНОГО списка (не выдуманы), similarity/difference/pattern
   осмысленные. Если аналогий нет — пустой массив.
10. Внепортфельные активы: у каждого есть rating и конкретный анализ (RSI, funding/OI, уровень, триггер).
11. R:R-ГИГИЕНА:
    - R:R ≥ 2.0 по ХУДШЕЙ границе набора. Если меньше — direction = "wait".
    - Ширина диапазона entry ≤ 1.5%. Если шире — сузь до ≤1%.
    - Риск (худший entry − стоп) = 1.5%–3.5% от entry.
    - entry в пределах ±2% от текущей цены BTC (из контекста).
    - Стоп на технической опоре, укажи какую опору использовал.

ВНИМАНИЕ: финальный риск-менеджмент (сужение диапазона, force wait при R:R < 2,
проверка entry vs current_price, warnings) сделает Python-код после тебя.
Твоя задача — выдать корректный черновик.

ВЫВОД — тот же JSON."""


# --- Вспомогательные функции ---

def calculate_kv_cache(seq_len: int, batch_size: int = 1) -> float:
    """Расчет теоретического размера KV-кэша в МБ (FP16)."""
    num_layers = 32
    num_heads = 32
    head_dim = 128
    bytes_per_param = 2

    kv_cache_bytes = 2 * batch_size * seq_len * num_layers * num_heads * head_dim * bytes_per_param
    return round(kv_cache_bytes / (1024 * 1024), 2)


def _extract_prices(text: Any) -> List[float]:
    """Извлекает все числа из строки (игнорирует $, запятые, пробелы, тире)."""
    if not text:
        return []
    s = str(text).replace("$", "").replace(",", "").replace(" ", "")
    matches = re.findall(r"\d+(?:\.\d+)?", s)
    try:
        return [float(m) for m in matches]
    except ValueError:
        return []


def _extract_current_btc_price(market_snapshot: str) -> Optional[float]:
    """Парсит текущую цену BTC из market_snapshot (формат 'BTC: $78381, ...')."""
    if not market_snapshot:
        return None
    m = re.search(r"BTC:\s*\$?([\d,]+(?:\.\d+)?)", market_snapshot)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _extract_scenario_price(scenarios: Dict[str, Any], key: str) -> Optional[float]:
    """Извлекает price из scenarios[key].price (может быть строкой '$72000' или числом)."""
    sc = (scenarios or {}).get(key) or {}
    price_raw = sc.get("price")
    if price_raw is None:
        return None
    prices = _extract_prices(price_raw)
    return prices[0] if prices else None


def _fmt_entry_range(lo: float, hi: float) -> str:
    """Форматирует диапазон [lo, hi] в строку с подходящей точностью."""
    if lo >= 1000:
        return f"${lo:,.0f} – ${hi:,.0f}"
    elif lo >= 10:
        return f"${lo:,.1f} – ${hi:,.1f}"
    elif lo >= 1:
        return f"${lo:,.2f} – ${hi:,.2f}"
    else:
        return f"${lo:,.4f} – ${hi:,.4f}"


def _narrow_entry(entry_min: float, entry_max: float, target_width_pct: float) -> Tuple[float, float]:
    """Сужает диапазон до target_width_pct вокруг середины."""
    mid = (entry_min + entry_max) / 2
    half = mid * (target_width_pct / 2) / 100
    return mid - half, mid + half


def _calc_rr_from_prices(worst_entry: float, stop: float, tp1: float, direction: str) -> Optional[float]:
    """R:R по конкретным числам (worst_entry, stop, tp1)."""
    if direction == "long":
        risk = worst_entry - stop
        reward = tp1 - worst_entry
    elif direction == "short":
        risk = stop - worst_entry
        reward = worst_entry - tp1
    else:
        return None

    if risk <= 0 or reward <= 0:
        return None
    return round(reward / risk, 2)


def _calc_rr(trading: Dict[str, Any]) -> Optional[float]:
    """R:R по полям trading (для совместимости, если нужно)."""
    direction = (trading.get("direction") or "").lower()
    if direction not in ("long", "short"):
        return None
    entry_prices = _extract_prices(trading.get("entry"))
    stop_prices = _extract_prices(trading.get("stop_loss"))
    tp_list = trading.get("take_profit") or []
    tp1_prices = _extract_prices(tp_list[0] if tp_list else None)
    if not entry_prices or not stop_prices or not tp1_prices:
        return None
    worst_entry = max(entry_prices) if direction == "long" else min(entry_prices)
    return _calc_rr_from_prices(worst_entry, stop_prices[0], tp1_prices[0], direction)


def _calc_recommended_entry(
    stop: float, tp1: float, direction: str, target_rr: float = RR_THRESHOLD
) -> Optional[float]:
    """Цена входа для R:R = target_rr при фиксированных stop и tp1."""
    if direction == "long":
        return round((tp1 + target_rr * stop) / (1 + target_rr), 2)
    elif direction == "short":
        return round((target_rr * stop + tp1) / (1 + target_rr), 2)
    return None


def _fix_risk_reward(result: Dict[str, Any], market_snapshot: str = "") -> Dict[str, Any]:
    """R:R-guard: сужение диапазона, force wait при R:R < 2 или entry вне ±2% от current,
    проверка консистентности сценариев (стоп vs bear/bull).
    """
    trading = result.get("trading") or {}
    if not isinstance(trading, dict):
        return result

    direction = (trading.get("direction") or "").lower()
    if direction not in ("long", "short"):
        return result

    # Извлекаем исходные числа
    entry_prices = _extract_prices(trading.get("entry"))
    stop_prices = _extract_prices(trading.get("stop_loss"))
    tp_list = trading.get("take_profit") or []
    tp1_prices = _extract_prices(tp_list[0] if tp_list else None)

    if not entry_prices or not stop_prices or not tp1_prices:
        logger.warning("R:R-guard: не хватает entry/stop/tp1 — пропускаю.")
        return result

    entry_min = min(entry_prices)
    entry_max = max(entry_prices)
    stop = stop_prices[0]
    tp1 = tp1_prices[0]
    current_price = _extract_current_btc_price(market_snapshot)
    scenarios = result.get("scenarios") or {}

    # --- 1. Сужение диапазона, если ширина > WIDTH_MAX_PCT ---
    width_pct = ((entry_max - entry_min) / entry_max * 100) if entry_max > 0 else 0
    narrowed = False
    if width_pct > WIDTH_MAX_PCT:
        new_min, new_max = _narrow_entry(entry_min, entry_max, WIDTH_TARGET_PCT)
        entry_min, entry_max = new_min, new_max
        trading["entry"] = _fmt_entry_range(entry_min, entry_max)
        narrowed = True
        logger.info(
            f"Диапазон сужен: было {width_pct:.1f}% → стало {WIDTH_TARGET_PCT:.1f}%. "
            f"Новый entry: {trading['entry']}"
        )

    # Пересчитываем worst_entry / risk / R:R после сужения
    worst_entry = entry_max if direction == "long" else entry_min
    risk = abs(worst_entry - stop)
    risk_pct = (risk / worst_entry * 100) if worst_entry > 0 else 0
    calculated_rr = _calc_rr_from_prices(worst_entry, stop, tp1, direction) or 0.0

    # --- 2. Проверка консистентности сценариев (warning, не блокирует) ---
    scenario_warnings = []
    if direction == "long":
        bear_price = _extract_scenario_price(scenarios, "bear_case")
        if bear_price is not None and stop >= bear_price:
            scenario_warnings.append(
                f"⚠️ Стоп ${stop:,.0f} ≥ bear_case.price ${bear_price:,.0f}. "
                f"Для long стоп должен быть НИЖЕ bear_case, иначе сценарии противоречат."
            )
    elif direction == "short":
        bull_price = _extract_scenario_price(scenarios, "bull_case")
        if bull_price is not None and stop <= bull_price:
            scenario_warnings.append(
                f"⚠️ Стоп ${stop:,.0f} ≤ bull_case.price ${bull_price:,.0f}. "
                f"Для short стоп должен быть ВЫШЕ bull_case."
            )

    # --- 3. Проверка R:R < RR_THRESHOLD → force wait ---
    if calculated_rr < RR_THRESHOLD:
        _force_wait(result, trading, direction, worst_entry, stop, tp1,
                    calculated_rr, risk_pct, narrowed, scenario_warnings,
                    current_price, reason="R:R слишком низкий")
        return result

    # --- 4. Проверка entry vs current_price (окно ±ENTRY_NEAR_CURRENT_PCT) ---
    if current_price is not None and current_price > 0:
        lo = current_price * (1 - ENTRY_NEAR_CURRENT_PCT / 100)
        hi = current_price * (1 + ENTRY_NEAR_CURRENT_PCT / 100)
        if worst_entry < lo or worst_entry > hi:
            _force_wait(
                result, trading, direction, worst_entry, stop, tp1,
                calculated_rr, risk_pct, narrowed, scenario_warnings,
                current_price, reason="entry вне окна от текущей цены"
            )
            return result

    # --- 5. Всё ок: пересчёт R:R, warnings ---
    trading["risk_reward"] = calculated_rr
    result["trading"] = trading

    warnings = list(scenario_warnings)
    if narrowed:
        warnings.append(
            f"ℹ️ Диапазон был сужен до {WIDTH_TARGET_PCT:.1f}% (был слишком широкий)."
        )
    if risk_pct > RISK_MAX_PCT:
        warnings.append(f"⚠️ Стоп {risk_pct:.1f}% от входа — далеко (норма ≤ {RISK_MAX_PCT:.1f}%).")
    elif risk_pct < RISK_MIN_PCT:
        warnings.append(f"⚠️ Стоп {risk_pct:.1f}% от входа — близко (норма ≥ {RISK_MIN_PCT:.1f}%).")

    if warnings:
        old_reasoning = (trading.get("reasoning", "") or "").strip()
        trading["reasoning"] = "\n\n".join(filter(None, [" ".join(warnings), old_reasoning]))

    return result


def _force_wait(
    result: Dict[str, Any],
    trading: Dict[str, Any],
    direction: str,
    worst_entry: float,
    stop: float,
    tp1: float,
    calculated_rr: float,
    risk_pct: float,
    narrowed: bool,
    scenario_warnings: List[str],
    current_price: Optional[float],
    reason: str,
) -> None:
    """Принудительно ставит direction=wait и очищает план, добавляя _recommended_entry."""
    recommended_entry = _calc_recommended_entry(stop, tp1, direction)

    logger.warning(
        f"Force wait: {reason}. direction={direction}, entry={trading.get('entry')}, "
        f"stop={stop}, tp1={tp1}, R:R={calculated_rr:.2f}, recommended_entry={recommended_entry}"
    )

    old_direction = trading.get("direction", "")
    orig_entry = trading.get("entry", "—")
    orig_stop = trading.get("stop_loss", "—")

    trading["direction"] = "wait"
    trading["risk_reward"] = 0
    # Очищаем числовые поля плана — formatters покажет "ждём зону $X"
    trading["entry"] = None
    trading["stop_loss"] = f"${stop:,.0f}"
    trading["take_profit"] = [f"${tp1:,.0f}"] + [
        t for t in (trading.get("take_profit") or [])[1:]
    ]

    if recommended_entry is not None:
        trading["_recommended_entry"] = recommended_entry

    parts = []

    if reason == "R:R слишком низкий":
        parts.append(
            f"⚠️ R:R при худшем входе (${worst_entry:,.0f}) = {calculated_rr:.2f} "
            f"(< {RR_THRESHOLD:.1f}). План переведён в режим ожидания."
        )
        if recommended_entry is not None:
            cmp = "≤" if direction == "long" else "≥"
            parts.append(
                f"Для R:R ≥ {RR_THRESHOLD:.1f} при стопе ${stop:,.0f} и TP1 ${tp1:,.0f} "
                f"нужен вход {cmp} ${recommended_entry:,.0f}."
            )
    elif reason == "entry вне окна от текущей цены":
        parts.append(
            f"⚠️ Вход ${worst_entry:,.0f} вне ±{ENTRY_NEAR_CURRENT_PCT:.0f}% от текущей "
            f"цены BTC (${current_price:,.0f}). План нереалистичен сейчас — режим ожидания."
        )
        if recommended_entry is not None:
            parts.append(
                f"Ждём откат/подход к зоне ${recommended_entry:,.0f}."
            )

    if narrowed:
        parts.append(
            f"ℹ️ Изначальный диапазон входа был слишком широкий — сужен до "
            f"{WIDTH_TARGET_PCT:.1f}% для точности."
        )

    if risk_pct > RISK_MAX_PCT:
        parts.append(f"⚠️ Стоп был далеко ({risk_pct:.1f}% от входа > {RISK_MAX_PCT:.1f}%).")
    elif risk_pct < RISK_MIN_PCT:
        parts.append(f"⚠️ Стоп был слишком близко ({risk_pct:.1f}% от входа < {RISK_MIN_PCT:.0f}%).")

    parts.extend(scenario_warnings)

    parts.append(
        f"Изначальный план LLM: {old_direction} при входе {orig_entry}, стоп {orig_stop}. "
        f"Он не проходит риск-фильтр."
    )

    trading["reasoning"] = " ".join(parts).strip()
    result["trading"] = trading


# --- Граф ---
class AgentState(TypedDict):
    """Состояние агента, передаваемое между узлами графа."""
    market_snapshot: str
    portfolio: Dict[str, Any]
    image_b64: str
    user_comment: str
    past_context: List[Dict[str, Any]]
    knowledge_context: List[Dict[str, Any]]
    events_digest: str
    news_context: str
    web_context: str
    portfolio_history: List[Dict[str, Any]]
    chat_history: List[Dict[str, str]]
    past_chat_context: List[Dict[str, str]]
    journal: List[Dict[str, Any]]
    draft: Dict[str, Any]
    chart_reading: str
    analysis_steps: List[str]
    historical_analogies: List[Dict[str, str]]
    trading: Dict[str, Any]
    portfolio_advice: Dict[str, Any]
    prediction_summary: str
    confidence: int
    market_regime: str
    scenarios: Dict[str, Any]
    triggers_to_watch: List[str]
    invalidation_factors: List[str]
    reasoning: str
    error: str
    kv_cache_mb: float
    token_length: int


def _extract_text(content: Any) -> str:
    """Извлекает текст из ответа LLM."""
    if isinstance(content, list):
        return "".join(b if isinstance(b, str) else b.get("text", "") for b in content)
    return str(content)


def _parse_json(raw: str) -> Dict[str, Any]:
    """Ищет JSON в ответе LLM и парсит его."""
    raw = raw.strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("JSON не найден в ответе модели")
    return json.loads(match.group(0))


def _build_user_prompt(state: AgentState) -> str:
    """Собирает контекст из всех источников для передачи в LLM."""
    knowledge_text = "\n\n".join(
        [f"--- ФРАГМЕНТ СТАТЬИ ---\n{k.get('text', '')}" for k in state["knowledge_context"]]
    )
    history_text = "\n".join(
        [f"{h['date']}: ${h['total_balance']:.2f} (PnL {h['total_pnl_pct']:.1f}%)" for h in state["portfolio_history"]]
    ) or "История пока пуста."

    events_digest = state.get("events_digest") or "Полный список исторических событий недоступен."

    journal = state.get("journal") or []
    if journal:
        journal_lines = []
        for j in journal:
            date = j.get("date", "")
            jtype = j.get("type", "")
            asset = j.get("asset", "")
            amount = j.get("amount_usd")
            note = j.get("note", "")
            if amount is not None:
                journal_lines.append(
                    f"[{date}] {jtype} по {asset} ({amount:+.2f}$): {note}"
                )
            else:
                journal_lines.append(
                    f"[{date}] {jtype} по {asset}: {note}"
                )
        journal_text = "\n".join(journal_lines)
    else:
        journal_text = "История операций пока пуста."

    chat_history_text = ""
    chat_history = state.get("chat_history") or []
    if chat_history:
        formatted = "\n".join(
            f"{m.get('role', 'user')}: {m.get('content', '')}" for m in chat_history
        )
        chat_history_text = f"\nИСТОРИЯ ДИАЛОГА:\n{formatted}\n"

    past_chat_text = ""
    past_chats = state.get("past_chat_context") or []
    if past_chats:
        formatted_past = "\n".join(
            [
                f"[{p.get('date', '')}]\n"
                f"Пользователь: {p.get('user_message', '')}\n"
                f"Агент: {p.get('assistant_message', '')}"
                for p in past_chats
            ]
        )
        past_chat_text = f"\nПОХОЖИЕ ПРОШЛЫЕ ОБСУЖДЕНИЯ:\n{formatted_past}\n"

    return f"""ТЕКУЩИЙ РЫНОК:
{state['market_snapshot']}

ПОРТФЕЛЬ:
{json.dumps(state['portfolio'], ensure_ascii=False)}

ИСТОРИЯ ОПЕРАЦИЙ (JOURNAL):
{journal_text}

ИСТОРИЯ ПОРТФЕЛЯ:
{history_text}

БАЗА ЗНАНИЙ (СТАТЬЯ УЧИТЕЛЯ — семантический топ-5):
{knowledge_text}

ПОЛНЫЙ СПИСОК ИСТОРИЧЕСКИХ СОБЫТИЙ:
{events_digest}

СВЕЖИЕ НОВОСТИ ИЗ TELEGRAM (текущая ситуация для сравнения):
{state['news_context']}

ДОПОЛНИТЕЛЬНЫЙ КОНТЕКСТ ИЗ ИНТЕРНЕТА (текущая ситуация для сравнения):
{state.get('web_context', 'Не использовался')}

ПРОШЛЫЕ ПРОГНОЗЫ (Trading Journal):
{json.dumps(state['past_context'], ensure_ascii=False, indent=2)}
{chat_history_text}{past_chat_text}
КОММЕНТАРИЙ ПОЛЬЗОВАТЕЛЯ К ГРАФИКУ:
"{state['user_comment'] or 'нет'}"

Проанализируй всё и дай прогноз."""


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """Узел 1: Сбор контекста из Qdrant и парсера новостей."""
    logger.info("Retrieve: Поиск в Qdrant и сбор новостей")
    past = find_similar_situations(state["market_snapshot"], limit=3)
    knowledge = find_knowledge("макро сценарий, зоны покупки, ключевые уровни, DCA", limit=5)
    events_digest = get_all_events_digest()
    news = get_relevant_news_for_ai(state["market_snapshot"][:500])
    portfolio_history = get_portfolio_history(limit=5)

    return {
        "past_context": past,
        "knowledge_context": knowledge,
        "events_digest": events_digest,
        "news_context": news,
        "portfolio_history": portfolio_history,
    }


def search_past_chat_node(state: AgentState) -> Dict[str, Any]:
    """Узел 2: Поиск в семантической памяти прошлых обсуждений."""
    logger.info("Search Past Chat: Поиск в семантической памяти диалогов")
    query = state.get("user_comment") or "анализ рынка"
    past_chats = search_past_chat(query, limit=settings.CHAT_HISTORY_SEARCH_LIMIT)
    return {"past_chat_context": past_chats}


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Узел 3: Веб-поиск через Exa."""
    logger.info("Web Search: Проверка аномалий через Exa")
    news_context = state["news_context"]

    if "резкое падение" not in news_context and "резкий рост" not in news_context:
        query = "Why is Bitcoin moving today?"
        try:
            results = exa_client.search(query, type="auto", num_results=3, contents={"highlights": True})
            web_text = "\n\n".join(
                [f"--- {r.title} ({r.url}) ---\n{r.highlights[0] if r.highlights else r.text[:500]}" for r in results.results]
            )
            return {"web_context": f"Результаты веб-поиска:\n{web_text}"}
        except Exception as e:
            logger.error(f"Ошибка Exa: {e}")
            return {"web_context": "Веб-поиск не дал результатов."}

    return {"web_context": "Веб-поиск не требуется."}


def analyze_node(state: AgentState) -> Dict[str, Any]:
    """Узел 4: Генерация черновика прогноза."""
    logger.info("Analyze: Генерация черновика прогноза")
    user_prompt = _build_user_prompt(state)

    try:
        if state.get("image_b64"):
            message = HumanMessage(content=[
                {"type": "text", "text": ANALYSIS_PROMPT + "\n\n" + user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{state['image_b64']}"}},
            ])
            response = analysis_llm.invoke([message])
        else:
            response = analysis_llm.invoke([SystemMessage(content=ANALYSIS_PROMPT), HumanMessage(content=user_prompt)])

        raw_resp = _extract_text(response.content)
        draft = _parse_json(raw_resp)

        seq_len = len(ANALYSIS_PROMPT + user_prompt) + len(raw_resp)
        kv_mb = calculate_kv_cache(seq_len)

        return {
            "draft": draft,
            "error": "",
            "kv_cache_mb": kv_mb,
            "token_length": seq_len,
        }
    except Exception as e:
        logger.error(f"Ошибка генерации прогноза: {e}")
        return {"draft": {}, "error": str(e)}


def verify_node(state: AgentState) -> Dict[str, Any]:
    """Узел 5: Верификация + программный R:R-guard."""
    logger.info("Verify: Верификация прогноза")
    draft = state["draft"]

    user_prompt = f"КОНТЕКСТ:\n{state['market_snapshot']}\n\nЧЕРНОВИК:\n{json.dumps(draft, ensure_ascii=False, indent=2)}\n\nПроверь и выдай финальный JSON."

    try:
        response = verify_llm.invoke([SystemMessage(content=VERIFY_PROMPT), HumanMessage(content=user_prompt)])
        result = _parse_json(_extract_text(response.content))
    except Exception as e:
        logger.error(f"Ошибка верификации: {e}")
        result = draft

    # Передаём market_snapshot в guard (нужен current_price BTC)
    result = _fix_risk_reward(result, market_snapshot=state.get("market_snapshot", ""))

    return {
        "chart_reading": result.get("chart_reading", "") or "",
        "analysis_steps": result.get("analysis_steps", []) or [],
        "historical_analogies": result.get("historical_analogies", []) or [],
        "trading": result.get("trading", {}) or {},
        "portfolio_advice": result.get("portfolio", {}) or {},
        "prediction_summary": result.get("prediction_summary", "") or "",
        "confidence": int(result.get("confidence", 0) or 0),
        "market_regime": result.get("market_regime", "") or "",
        "scenarios": result.get("scenarios", {}) or {},
        "triggers_to_watch": result.get("triggers_to_watch", []) or [],
        "invalidation_factors": result.get("invalidation_factors", []) or [],
        "reasoning": result.get("reasoning", "") or "",
        "error": "",
    }


workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("search_past_chat", search_past_chat_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("analyze", analyze_node)
workflow.add_node("verify", verify_node)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "search_past_chat")
workflow.add_edge("search_past_chat", "web_search")
workflow.add_edge("web_search", "analyze")

workflow.add_conditional_edges(
    "analyze",
    lambda s: "verify" if not s.get("error") else "end",
    {"verify": "verify", "end": END},
)
workflow.add_edge("verify", END)

agent_app = workflow.compile()


@observe(name="crypto_agent_run")
def generate_ai_prediction(
    market_snapshot: str,
    portfolio: Dict[str, Any],
    image_b64: str = "",
    user_comment: str = "",
    session_id: str = "default_session",
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """Запускает граф агента и возвращает финальный прогноз."""
    config = {
        "callbacks": [langfuse_handler],
        "metadata": {"langfuse_session_id": session_id}
    }

    journal = portfolio.get("journal", []) if isinstance(portfolio, dict) else []

    result = agent_app.invoke(
        {
            "market_snapshot": market_snapshot,
            "portfolio": portfolio,
            "image_b64": image_b64,
            "user_comment": user_comment,
            "chat_history": chat_history or [],
            "journal": journal,
        },
        config=config,
    )

    try:
        trace_id = langfuse.get_current_trace_id()
    except Exception as e:
        logger.warning(f"Не удалось получить trace_id из Langfuse: {e}")
        trace_id = None

    kv_mb = result.get("kv_cache_mb")
    token_len = result.get("token_length")
    past_chat_count = len(result.get("past_chat_context", []))
    analogies_count = len(result.get("historical_analogies", []))

    if kv_mb is not None:
        langfuse.update_current_span(
            metadata={
                "kv_cache_mb": kv_mb,
                "token_length": token_len,
                "past_chat_matches": past_chat_count,
                "historical_analogies_count": analogies_count,
            }
        )

    result["trace_id"] = trace_id
    return result