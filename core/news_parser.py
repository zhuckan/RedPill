"""Парсер новостей из Telegram-канала через Telethon.

Модуль читает новые сообщения из заданного канала, сохраняет их в
Markdown-файл (для истории) и в Qdrant (для семантического поиска).
Использует механизм `last_id` для того, чтобы не парсить одни и те же
сообщения дважды.
"""
import asyncio
import logging
import os
from datetime import datetime
from typing import Optional

import qrcode
from telethon import TelegramClient

from config import settings
from core.rag_memory import save_news_to_qdrant, search_relevant_news

logger = logging.getLogger(__name__)


def _get_current_news_file() -> str:
    """Возвращает путь к Markdown-файлу с новостями за текущий месяц."""
    month = datetime.now().strftime("%Y-%m")
    return os.path.join(settings.NEWS_DIR, f"news_{month}.md")


def _read_last_id(path: str) -> int:
    """Читает ID последнего обработанного сообщения из файла.

    Первая строка файла имеет формат: `<!-- last_id: 12345 -->`.
    Если файла нет или ID не найден, возвращает 0.
    """
    if not os.path.exists(path):
        return 0

    with open(path, "r", encoding="utf-8") as f:
        first_line = f.readline()

    if first_line.startswith("<!-- last_id: "):
        try:
            return int(first_line.split(": ")[1].replace(" -->", "").strip())
        except ValueError:
            return 0
    return 0


def _write_last_id(path: str, new_id: int, old_id: int) -> None:
    """Обновляет `last_id` в первой строке файла.

    Заменяет старый ID на новый. Используется после успешного парсинга,
    чтобы при следующем запуске не тянуть те же сообщения снова.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    content = content.replace(
        f"<!-- last_id: {old_id} -->",
        f"<!-- last_id: {new_id} -->",
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _read_existing(path: str) -> str:
    """Читает содержимое файла (для проверки, что новость уже записана)."""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


async def _authorize_via_qr(client: TelegramClient) -> None:
    """Авторизация через QR-код (первый запуск Telethon).

    Генерирует QR-код, сохраняет в файл и ждет, пока пользователь
    отсканирует его через Telegram. Таймаут — 120 секунд.
    """
    logger.info("Требуется авторизация Telethon, генерирую QR-код")
    qr_login = await client.qr_login()

    qr = qrcode.QRCode()
    qr.add_data(qr_login.url)
    qr.make()
    qr.make_image().save(settings.QR_CODE_FILE)

    print("=" * 60)
    print(f"Открой файл {settings.QR_CODE_FILE} и отсканируй его в Telegram:")
    print("Настройки -> Устройства -> Подключить устройство")
    print("=" * 60)

    try:
        await qr_login.wait(timeout=120)
        logger.info("QR-код успешно отсканирован")
    except asyncio.TimeoutError:
        logger.error("Время ожидания QR-кода истекло")


async def _fetch_and_save_news_async() -> int:
    """Асинхронно парсит новые сообщения из Telegram-канала.

    Логика идемпотентности:
    1. Читаем `last_id` из файла (последнее обработанное сообщение).
    2. Тянем сообщения с `min_id=last_id`.
    3. Для каждого сообщения проверяем, нет ли его уже в файле
       (по маркеру `**[дата]** текст`). Если есть — пропускаем.
    4. После успешной записи обновляем `last_id` в первой строке файла.

    Returns:
        Количество сохраненных сообщений.
    """
    if not settings.TELEGRAM_API_ID or not settings.TELEGRAM_API_HASH:
        logger.error("TELEGRAM_API_ID или TELEGRAM_API_HASH не заданы в .env")
        return 0

    os.makedirs(settings.NEWS_DIR, exist_ok=True)
    news_file = _get_current_news_file()
    last_saved_id = _read_last_id(news_file)
    existing_content = _read_existing(news_file)

    client = TelegramClient(
        settings.TELEGRAM_SESSION_NAME,
        settings.TELEGRAM_API_ID,
        settings.TELEGRAM_API_HASH,
        device_model=settings.TELEGRAM_DEVICE_MODEL,
        system_version=settings.TELEGRAM_SYSTEM_VERSION,
        app_version=settings.TELEGRAM_APP_VERSION,
    )

    try:
        await client.connect()

        if not await client.is_user_authorized():
            await _authorize_via_qr(client)

        channel = await client.get_entity(settings.TELEGRAM_CHANNEL)
        messages = await client.get_messages(
            channel,
            limit=settings.TELEGRAM_FETCH_LIMIT,
            min_id=last_saved_id,
        )

        if not messages:
            logger.info("Новых сообщений нет")
            return 0

        # Telethon возвращает сообщения в порядке "новые сверху",
        # переворачиваем, чтобы обрабатывать их хронологически.
        messages = list(reversed(messages))
        max_id = last_saved_id
        saved_count = 0

        with open(news_file, "a+", encoding="utf-8") as f:
            # Если файл пустой — инициализируем его заголовком
            if os.path.getsize(news_file) == 0:
                f.write(
                    f"<!-- last_id: {last_saved_id} -->\n"
                    f"# Новости за {datetime.now().strftime('%B %Y')}\n\n"
                )

            for msg in messages:
                # max_id двигаем до проверки текста: если последнее сообщение
                # было без текста (только картинка), мы все равно не хотим
                # его парсить повторно.
                max_id = max(max_id, msg.id)

                if not msg.text:
                    continue

                date_str = msg.date.strftime("%Y-%m-%d %H:%M")
                text = msg.text.replace("\n", " ")
                marker = f"**[{date_str}]** {text}"

                # Идемпотентность: если запись уже есть в файле,
                # пропускаем (например, после падения процесса посередине).
                if marker in existing_content:
                    continue

                f.write(f"{marker}\n\n")
                save_news_to_qdrant(date_str, text)
                existing_content += f"{marker}\n\n"
                saved_count += 1

        _write_last_id(news_file, max_id, last_saved_id)
        logger.info(f"Сохранено новых новостей: {saved_count}")
        return saved_count

    except Exception as e:
        logger.error(f"Ошибка парсинга Telegram: {e}")
        return 0
    finally:
        if client.is_connected():
            await client.disconnect()


def fetch_and_save_news() -> int:
    """Синхронная обертка над `_fetch_and_save_news_async`.

    Важно: `asyncio.run()` падает, если вызвать его внутри уже работающего
    event loop (например, из async-функции бота). Поэтому бот вызывает
    эту функцию через `asyncio.to_thread()`, чтобы запустить её в отдельном
    потоке с собственным event loop.

    Returns:
        Количество сохраненных сообщений.
    """
    return asyncio.run(_fetch_and_save_news_async())


def get_relevant_news_for_ai(market_context: str) -> str:
    """Возвращает релевантные новости для передачи в промпт ИИ-агента.

    Args:
        market_context: Краткое описание рыночной ситуации (используется
            для формирования семантического запроса в Qdrant).

    Returns:
        Строка с новостями в формате `[дата] текст`, разделенная переводами
        строк. Если новостей не найдено — возвращает сообщение об этом.
    """
    query = f"Криптовалюта, ФРС, ставка, инфляция, биткоин, {market_context}"
    news = search_relevant_news(query, limit=7)

    if not news:
        return "Релевантных новостей не найдено."

    return "\n".join(f"[{n['date']}] {n['text']}" for n in news)