import os
from dotenv import load_dotenv

# Загружаем переменные окружения из .env файла
load_dotenv()

# --- Секреты и ключи API ---
TELEGRAM_TOKEN: str = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")

# Ключи для парсера Telegram (Telethon)
TELEGRAM_API_ID: int = int(os.getenv("TELEGRAM_API_ID", 0))
TELEGRAM_API_HASH: str = os.getenv("TELEGRAM_API_HASH", "")

# Ключи для Langfuse (мониторинг)
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")

# --- Модели Gemini ---
GEMINI_PRIMARY_MODEL = "gemini-3.8-flash"
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"

# --- Эндпоинты API ---
COINGECKO_MARKETS_URL = "https://api.coingecko.com/api/v3/coins/markets"
COINGECKO_GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"
BINANCE_OI_URL = "https://fapi.binance.com/futures/data/openInterestHist"
FOREX_FACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FEAR_GREED_URL = "https://api.alternative.me/fng/"

# --- Пути к локальным файлам ---
ASSETS_FILE = "assets.json"
PORTFOLIO_FILE = "portfolio.json"
NEWS_DIR = "knowledge_base/news"
MENTOR_ARTICLE_FILE = "knowledge_base/mentor/article_2026-10-28.md"
CHART_LEGEND_FILE = "knowledge_base/chart_legend.md"
QR_CODE_FILE = "telegram_login_qr.png"

# --- Настройки Telegram-парсера ---
TELEGRAM_SESSION_NAME = "redpill_news_session"
TELEGRAM_CHANNEL = "cryptoattack24"
TELEGRAM_DEVICE_MODEL = "MacBook Pro"
TELEGRAM_SYSTEM_VERSION = "macOS 15"
TELEGRAM_APP_VERSION = "4.15.2"
TELEGRAM_FETCH_LIMIT = 1000

# --- Векторная база данных (Qdrant) ---
QDRANT_HOST = "http://localhost:6333"
QDRANT_TRADING_COLLECTION = "trading_memory"
QDRANT_NEWS_COLLECTION = "news_memory"
QDRANT_KNOWLEDGE_COLLECTION = "knowledge_memory"
QDRANT_PORTFOLIO_COLLECTION = "portfolio_history"
QDRANT_CHAT_HISTORY_COLLECTION = "chat_history_memory"  # семантическая память диалогов
QDRANT_VECTOR_SIZE = 1024

# --- Ограничения для парсера новостей ---
NEWS_TTL_DAYS = 90
NEWS_MAX_POINTS = 10000

# --- Chat History ---
CHAT_HISTORY_FILE = "knowledge_base/chat_history.json"  # Персистентность локальной истории
MAX_LOCAL_HISTORY = 20  # Сколько последних сообщений держать в «горячем» контексте
CHAT_HISTORY_SEARCH_LIMIT = 5  # Сколько прошлых диалогов доставать из Qdrant по смыслу