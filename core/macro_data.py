"""Сбор макроэкономических данных из внешних источников.

Модуль агрегирует информацию о традиционных рынках (SPX, золото, нефть, DXY),
доминации криптовалют, экономическом календаре и индексе страха/жадности.
Используется ботом для формирования контекста ИИ-агенту.
"""
import logging
from typing import Dict, List, Optional

import requests
import yfinance as yf

from config import settings

logger = logging.getLogger(__name__)


def fetch_macro_assets() -> Dict[str, Dict[str, float]]:
    """Получает котировки макро-активов через Yahoo Finance.

    Загружает данные за последние два дня, чтобы рассчитать изменение
    за 24 часа (разница между двумя последними ценами закрытия).

    Returns:
        Словарь вида {название: {"price": float, "chg_24h": float}}.
        При ошибке возвращает пустой словарь.
    """
    # Маппинг: человекочитаемое имя -> тикер Yahoo Finance
    tickers = {
        "SPX": "^GSPC",
        "GOLD": "GC=F",
        "OIL": "CL=F",
        "DXY": "DX-Y.NYB",
    }
    macro_data: Dict[str, Dict[str, float]] = {}

    try:
        data = yf.download(
            list(tickers.values()),
            period="2d",
            interval="1d",
            progress=False,
        )
    except Exception as e:
        logger.error(f"Ошибка Yahoo Finance: {e}")
        return macro_data

    for name, ticker in tickers.items():
        try:
            closes = data["Close"][ticker].dropna()
            if len(closes) < 2:
                continue

            current = float(closes.iloc[-1])
            previous = float(closes.iloc[-2])
            macro_data[name] = {
                "price": round(current, 2),
                "chg_24h": round(((current - previous) / previous) * 100, 2),
            }
        except (KeyError, IndexError, ZeroDivisionError) as e:
            logger.warning(f"Ошибка обработки макро {ticker}: {e}")

    return macro_data


def fetch_crypto_dominance() -> Dict[str, float]:
    """Получает доминацию BTC и USDT на рынке через CoinGecko.

    Returns:
        Словарь вида {"BTC.D": float, "USDT.D": float}.
        При ошибке возвращает пустой словарь.
    """
    try:
        response = requests.get(settings.COINGECKO_GLOBAL_URL, timeout=10)
        response.raise_for_status()
        percentages = response.json().get("data", {}).get("market_cap_percentage", {})
        return {
            "BTC.D": round(percentages.get("btc", 0), 2),
            "USDT.D": round(percentages.get("usdt", 0), 2),
        }
    except Exception as e:
        logger.error(f"Ошибка CoinGecko Global: {e}")
        return {}


def fetch_economic_calendar() -> List[Dict[str, Optional[str]]]:
    """Получает ближайшие важные экономические события (High Impact).

    Загружает календарь Forex Factory и фильтрует события по валютам
    (USD, EUR, CNY, JPY) и уровню важности (High).

    Returns:
        Список словарей с полями country, title, date, forecast, previous.
        Максимум 5 событий. При ошибке — пустой список.
    """
    try:
        response = requests.get(settings.FOREX_FACTORY_CALENDAR_URL, timeout=10)
        response.raise_for_status()
        events = response.json()
    except Exception as e:
        logger.error(f"Ошибка календаря: {e}")
        return []

    target_currencies = {"USD", "EUR", "CNY", "JPY"}

    important_events = [
        {
            "country": event.get("country"),
            "title": event.get("title"),
            "date": event.get("date"),
            "forecast": event.get("forecast"),
            "previous": event.get("previous"),
        }
        for event in events
        if event.get("impact") == "High" and event.get("country") in target_currencies
    ]

    return important_events[:5]


def fetch_fear_and_greed() -> Dict[str, Optional[str]]:
    """Получает индекс страха и жадности (Fear & Greed) от Alternative.me.

    Returns:
        Словарь вида {"value": int, "classification": str}.
        При ошибке возвращает пустой словарь.
    """
    try:
        response = requests.get(settings.FEAR_GREED_URL, timeout=10)
        response.raise_for_status()
        entry = response.json().get("data", [])[0]
        return {
            "value": int(entry.get("value", 0)),
            "classification": entry.get("value_classification", "Unknown"),
        }
    except Exception as e:
        logger.error(f"Ошибка Fear & Greed: {e}")
        return {}