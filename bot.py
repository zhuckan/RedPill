import asyncio
import base64
import json
import logging
import os
import re
import signal
import warnings
from typing import Dict, List, Optional

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, CommandStart, BaseFilter
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from langchain_core.messages import HumanMessage

from config import settings
from core.ai_agent import analysis_llm, generate_ai_prediction
from core.formatters import (
    build_ai_details_message,
    build_ai_main_message,
    build_market_message,
    build_market_text_for_ai,
)
from core.market_data import get_full_market_snapshot
from core.news_parser import fetch_and_save_news
from core.rag_memory import (
    cleanup_old_news,
    init_db,
    init_news_db,
    save_chat_exchange_to_qdrant,
    save_portfolio_snapshot,
    save_prediction,
    update_prediction_outcome,
)

# Подавляем шумные предупреждения от pandas/yfinance
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

bot = Bot(token=settings.TELEGRAM_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

OWNER_ID = int(settings.TELEGRAM_CHAT_ID)

# --- Персистентная история диалога ---
CHAT_HISTORY_FILE = settings.CHAT_HISTORY_FILE
MAX_HISTORY_MESSAGES = settings.MAX_LOCAL_HISTORY

# Метка кнопки «Портфель» — чтобы не путать с текстом
PORTFOLIO_BUTTON = "💼 Портфель"


def _load_histories() -> Dict[int, List[Dict[str, str]]]:
    """Загружает истории из JSON при старте бота."""
    if not os.path.exists(CHAT_HISTORY_FILE):
        return {}
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {int(k): v for k, v in raw.items()}
    except Exception as e:
        logger.error(f"Ошибка загрузки истории: {e}")
        return {}


def _save_histories() -> None:
    """Сохраняет истории в JSON."""
    try:
        os.makedirs(os.path.dirname(CHAT_HISTORY_FILE), exist_ok=True)
        with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(CHAT_HISTORIES, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Ошибка сохранения истории: {e}")


CHAT_HISTORIES: Dict[int, List[Dict[str, str]]] = _load_histories()


def get_chat_history(chat_id: int) -> List[Dict[str, str]]:
    """Возвращает историю диалога для конкретного чата."""
    return CHAT_HISTORIES.get(chat_id, [])


def append_to_history(chat_id: int, role: str, content: str) -> None:
    """Добавляет сообщение в историю, обрезает до лимита и сохраняет в файл."""
    history = CHAT_HISTORIES.setdefault(chat_id, [])
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_MESSAGES:
        CHAT_HISTORIES[chat_id] = history[-MAX_HISTORY_MESSAGES:]
    _save_histories()


class IsOwner(BaseFilter):
    """Фильтр для проверки, что сообщение пишет владелец бота."""
    async def __call__(self, event: Message) -> bool:
        return event.from_user.id == OWNER_ID


def split_message(text: str, limit: int = 4000) -> List[str]:
    """Безопасно разбивает длинное сообщение на части, не ломая HTML-теги."""
    if len(text) <= limit:
        return [text]

    parts = []
    current_part = ""
    for line in text.split("\n"):
        if len(current_part) + len(line) + 1 > limit:
            parts.append(current_part)
            current_part = line
        else:
            current_part += ("\n" if current_part else "") + line

    if current_part:
        parts.append(current_part)
    return parts


def main_menu_kb() -> ReplyKeyboardMarkup:
    """Клавиатура с одной кнопкой «Портфель»."""
    return ReplyKeyboardMarkup(
        keyboard=[[KeyboardButton(text=PORTFOLIO_BUTTON)]],
        resize_keyboard=True,
        is_persistent=True,
    )


@dp.message(CommandStart(), IsOwner())
async def cmd_start(message: Message):
    """Приветственное сообщение с инструкцией."""
    await message.answer(
        "👋 Привет. Я Edward \"Teddy\" Daniels.\n\n"
        "📸 <b>Отправь скриншот графика TradingView</b> — я проанализирую рынок и портфель, покажу свои рассуждения и дам план.\n\n"
        "💬 <b>После прогноза можно задать уточняющий вопрос текстом</b> (например, «А что по ETH?», «А если пробьёт $74k?»).\n\n"
        "📝 <b>Чтобы внести результат прогноза</b> — напиши свободным текстом, указав ID.\n\n"
        "💼 <b>Кнопка «Портфель»</b> — покажу текущее состояние портфеля.\n\n"
        "🔄 <b>/reset</b> — очистить локальную историю сессии.",
        parse_mode="HTML",
        reply_markup=main_menu_kb(),
    )


@dp.message(F.text == PORTFOLIO_BUTTON, IsOwner())
async def handle_portfolio_button(message: Message):
    """Показывает только блок портфеля (без отчёта и без ИИ-анализа)."""
    status = await message.answer("💼 Собираю данные по портфелю...")

    snapshot = await asyncio.to_thread(get_full_market_snapshot)
    if not snapshot:
        await status.edit_text("❌ Не удалось собрать метрики.")
        return

    full_report = build_market_message(snapshot)
    marker = "💼 <b>ПОРТФЕЛЬ</b>"

    if marker in full_report:
        # Берём всё от маркера до конца
        portfolio_text = full_report[full_report.index(marker):]
    else:
        portfolio_text = "Нет данных по портфелю."

    for chunk in split_message(portfolio_text):
        await message.answer(chunk, parse_mode="HTML")

    await status.delete()


async def run_analysis(
    message: Message,
    image_b64: Optional[str],
    user_comment: str,
) -> None:
    """Единая логика анализа (для фото и текстовых follow-up вопросов)."""
    status = await message.answer("📰 Собираю новости и метрики...")

    try:
        await asyncio.to_thread(cleanup_old_news)
        await asyncio.to_thread(fetch_and_save_news)
    except Exception as e:
        logger.error(f"Ошибка парсинга новостей: {e}")

    snapshot = await asyncio.to_thread(get_full_market_snapshot)
    if not snapshot:
        await status.edit_text("❌ Не удалось собрать метрики.")
        return

    await asyncio.to_thread(save_portfolio_snapshot, snapshot["portfolio"])

    if image_b64 is not None:
        report = build_market_message(snapshot)
        for chunk in split_message(report):
            await message.answer(chunk, parse_mode="HTML")

    await status.edit_text("🧠 Edward Daniels анализирует данные...")
    session_id = f"tg_{message.chat.id}"
    market_text = build_market_text_for_ai(snapshot)

    user_content = user_comment or "Прислал скриншот графика"
    history_for_agent = get_chat_history(message.chat.id) + [
        {"role": "user", "content": user_content}
    ]

    result = await asyncio.to_thread(
        generate_ai_prediction,
        market_text,
        snapshot["portfolio"],
        image_b64 or "",
        user_comment,
        session_id=session_id,
        chat_history=history_for_agent,
    )

    if result.get("error"):
        await status.edit_text(f"❌ Ошибка агента: {result['error']}")
        return

    # Подкладываем снимок портфеля, чтобы formatters вывел цены и PnL
    result["_portfolio_snapshot"] = snapshot["portfolio"]

    main_text = build_ai_main_message(result)
    details_text = build_ai_details_message(result)

    await status.edit_text("📤 Отправляю прогноз...")

    for chunk in split_message(main_text):
        await message.answer(chunk, parse_mode="HTML")

    if details_text and details_text.count("\n") > 2:
        for chunk in split_message(details_text):
            await message.answer(chunk, parse_mode="HTML")

    prediction_summary = result.get("prediction_summary", "") or result.get("prediction", "")

    if image_b64 is not None:
        prediction_id = await asyncio.to_thread(
            save_prediction,
            market_text,
            prediction_summary,
            result.get("reasoning", ""),
            snapshot["portfolio"]["details"],
            result.get("trace_id"),
        )
        await message.answer(
            f"🆔 <b>ID прогноза:</b> <code>{prediction_id}</code>\n"
            f"<i>Запиши его. Через 3-7 дней опиши свободным текстом, что произошло.</i>",
            parse_mode="HTML",
        )

    await asyncio.to_thread(
        save_chat_exchange_to_qdrant,
        user_content,
        prediction_summary,
        session_id,
    )

    append_to_history(message.chat.id, "user", user_content)
    append_to_history(message.chat.id, "assistant", prediction_summary)

    await status.delete()


@dp.message(F.photo, IsOwner())
async def handle_photo(message: Message):
    """Хендлер для фото."""
    await message.answer("🖼 Вижу график. Готовлю анализ...")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    b64_image = base64.b64encode(file_bytes.read()).decode("utf-8")
    user_comment = message.caption or ""

    await run_analysis(message, b64_image, user_comment)


async def handle_feedback(message: Message, point_id: int, text: str) -> None:
    """Обрабатывает фидбек по прогнозу (текст с ID)."""
    status = await message.answer("🧠 Парсирую твой комментарий...")

    extract_prompt = f"""Проанализируй текст пользователя и верни СТРОГО валидный JSON (без markdown, без комментариев).

ПРАВИЛА:
1. verdict — одно из трёх значений: "Сработал", "Не сработал", "Частично сработал".
   - Если пользователь говорит, что цена дошла до цели, но потом упала/выросла обратно, или говорит "почти", "частично", "не идеально", "но через день упали" -> "Частично сработал"
   - Если "ты был прав", "сработал", "угадал", "как и предсказал" и цена дошла до цели -> "Сработал"
   - Если "не сработал", "ошибся", "не угадал", "не прав", "противоположное" -> "Не сработал"
2. summary — краткое резюме В 1-2 ПРЕДЛОЖЕНИЯХ (что произошло по факту, без воды, максимум 200 символов).

ТЕКСТ ПОЛЬЗОВАТЕЛЯ:
\"\"\"{text}\"\"\"

ПРИМЕР:
{{"verdict": "Частично сработал", "summary": "Цена дошла до зоны входа 78133, дала рост до 81100, но затем откатилась до 76000."}}

ТВОЙ ОТВЕТ (только JSON):"""

    verdict = "Частично сработал"
    summary = text[:200]
    raw = ""

    try:
        resp = analysis_llm.invoke([HumanMessage(content=extract_prompt)])
        raw = resp.content if isinstance(resp.content, str) else str(resp.content)
        raw = raw.strip()

        # Чистим возможные markdown-обертки ```json ... ```
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]

        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            json_str = m.group(0)
            # Простая чистка от возможных артефактов
            json_str = json_str.replace("'", '"')
            data = json.loads(json_str)
            verdict = data.get("verdict", "Частично сработал")
            summary = data.get("summary", text[:200])

    except Exception as e:
        logger.error(f"Ошибка парсинга фидбека: {e}. Raw: {raw[:200]}")
        tl = text.lower()

        # ВАЖНО: Сначала проверяем на "частично", чтобы не сработало правило "сработал"
        if any(k in tl for k in ["частично", "почти", "не идеально", "но через", "но потом", "откатилась"]):
            verdict = "Частично сработал"
        elif any(k in tl for k in ["не сработал", "ошибся", "не угадал", "не прав"]):
            verdict = "Не сработал"
        elif any(k in tl for k in ["ты был прав", "сработал", "угадал", "как и предсказ", "молодец"]):
            verdict = "Сработал"

        # Если LLM упала, делаем резюме хотя бы чуть чище
        summary = f"Фидбек: {text[:150]}..."

    ok = await asyncio.to_thread(update_prediction_outcome, point_id, summary, verdict)

    if ok:
        await status.edit_text(
            f"✅ Прогноз <code>{point_id}</code> обновлен.\n"
            f"Вердикт: <b>{verdict}</b>\n"
            f"Резюме: {summary}\n\n"
            f"<i>Запомнил. Буду учитывать в следующих прогнозах.</i>",
            parse_mode="HTML",
        )
    else:
        await status.edit_text(f"❌ Не удалось обновить прогноз {point_id}.")


@dp.message(F.text, IsOwner())
async def handle_text(message: Message):
    """Роутинг: /reset, фидбек с ID или follow-up вопрос."""
    text = message.text.strip()

    if text.lower() in ("/reset", "/clear"):
        CHAT_HISTORIES.pop(message.chat.id, None)
        _save_histories()
        await message.answer(
            "🔄 Локальная история очищена.\n"
            "<i>Семантическая память в Qdrant не тронута.</i>",
            parse_mode="HTML",
        )
        return

    id_match = re.search(r"\b(\d{10,})\b", text)
    if id_match:
        await handle_feedback(message, int(id_match.group(1)), text)
        return

    await run_analysis(message, None, text)


async def main_async():
    """Точка входа: инициализация БД, настройка сигналов, запуск polling."""
    init_db()
    init_news_db()
    logger.info("Бот запущен. Ожидаю сообщения.")
    logger.info(f"Загружено историй диалогов: {len(CHAT_HISTORIES)} чатов.")

    loop = asyncio.get_running_loop()

    def _stop():
        logger.info("Получен сигнал остановки. Прекращаю polling...")
        asyncio.create_task(dp.stop_polling())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _stop)
        except NotImplementedError:
            pass

    try:
        await dp.start_polling(bot)
    finally:
        logger.info("Закрываю сессию бота...")
        await bot.session.close()
        logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Прервано пользователем.")
    except Exception as e:
        logger.error(f"Фатальная ошибка: {e}")