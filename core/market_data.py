"""Сбор рыночных данных из внешних API.

Модуль агрегирует информацию по криптоактивам (цены, RSI, фандинг, OI)
и портфелю пользователя. Используется ботом для формирования отчета
и контекста для ИИ-агента.
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import requests

from config import settings
from core.macro_data import (
    fetch_crypto_dominance,
    fetch_economic_calendar,
    fetch_fear_and_greed,
    fetch_macro_assets,
)

logger = logging.getLogger(__name__)


def load_assets_config() -> Dict[str, str]:
    """Загружает список отслеживаемых активов из JSON-файла.

    Returns:
        Словарь вида {тикер: coingecko_id} или пустой словарь при ошибке.
    """
    try:
        with open(settings.ASSETS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Файл {settings.ASSETS_FILE} не найден")
        return {}


def load_portfolio() -> dict:
    """Загружает текущий портфель пользователя из JSON-файла.

    Returns:
        Словарь с данными портфеля или пустой словарь при ошибке.
    """
    try:
        with open(settings.PORTFOLIO_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Файл {settings.PORTFOLIO_FILE} не найден")
        return {}


def fetch_market_data(asset_ids: List[str]) -> Optional[Dict]:
    """Получает базовые рыночные данные из CoinGecko.

    Args:
        asset_ids: Список идентификаторов активов (например, ["bitcoin", "ethereum"]).

    Returns:
        Словарь вида {asset_id: данные} или None при ошибке.
    """
    params = {
        "vs_currency": "usd",
        "ids": ",".join(asset_ids),
        "price_change_percentage": "7d,30d",
    }
    try:
        response = requests.get(settings.COINGECKO_MARKETS_URL, params=params, timeout=15)
        response.raise_for_status()
        return {asset["id"]: asset for asset in response.json()}
    except Exception as e:
        logger.error(f"Ошибка CoinGecko: {e}")
        return None


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Вычисляет индекс относительной силы (RSI) по списку цен.

    Args:
        prices: Список цен закрытия (от старых к новым).
        period: Период расчета RSI (по умолчанию 14).

    Returns:
        Значение RSI от 0 до 100 или None, если данных недостаточно.
    """
    if len(prices) < period + 1:
        return None

    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    gains = [d if d > 0 else 0 for d in deltas]
    losses = [-d if d < 0 else 0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    return 100 - (100 / (1 + (avg_gain / avg_loss)))


def fetch_rsi(ticker: str, interval: str) -> Optional[float]:
    """Получает RSI для указанного актива и таймфрейма с Binance.

    Args:
        ticker: Тикер актива (например, "BTC").
        interval: Таймфрейм (например, "4h", "1d", "1w", "1M").

    Returns:
        Значение RSI или None при ошибке.
    """
    params = {"symbol": f"{ticker.upper()}USDT", "interval": interval, "limit": 100}
    try:
        response = requests.get(settings.BINANCE_KLINES_URL, params=params, timeout=10)
        response.raise_for_status()
        return calculate_rsi([float(c[4]) for c in response.json()])
    except Exception as e:
        logger.warning(f"Ошибка RSI {ticker}/{interval}: {e}")
        return None


def fetch_all_rsis(ticker: str) -> Tuple[Optional[float], ...]:
    """Получает RSI по всем четырем таймфреймам (4h, 1d, 1w, 1M).

    Args:
        ticker: Тикер актива.

    Returns:
        Кортеж из 4 значений RSI (или None для тех, что не удалось получить).
    """
    intervals = ["4h", "1d", "1w", "1M"]
    results = []
    for interval in intervals:
        results.append(fetch_rsi(ticker, interval))
        time.sleep(0.2)  # Binance банит за частые запросы
    return tuple(results)


def fetch_funding_rate(ticker: str) -> Optional[float]:
    """Получает текущую ставку фандинга для фьючерса.

    Args:
        ticker: Тикер актива.

    Returns:
        Ставка фандинга в процентах или None при ошибке.
    """
    params = {"symbol": f"{ticker.upper()}USDT"}
    try:
        response = requests.get(settings.BINANCE_FUNDING_URL, params=params, timeout=10)
        response.raise_for_status()
        return float(response.json().get("lastFundingRate", 0)) * 100
    except Exception as e:
        logger.warning(f"Ошибка фандинга {ticker}: {e}")
        return None


def fetch_open_interest(ticker: str) -> dict:
    """Получает открытый интерес (Open Interest) и его изменение за 24 часа.

    Args:
        ticker: Тикер актива.

    Returns:
        Словарь {"value": float, "chg_24h": float}. При ошибке — нули.
    """
    params = {"symbol": f"{ticker.upper()}USDT", "period": "1d", "limit": 2}
    try:
        response = requests.get(settings.BINANCE_OI_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2:
            return {"value": 0.0, "chg_24h": 0.0}

        current = float(data[-1]["sumOpenInterestValue"])
        previous = float(data[-2]["sumOpenInterestValue"])
        change = ((current - previous) / previous) * 100 if previous > 0 else 0.0
        return {"value": current, "chg_24h": change}
    except Exception as e:
        logger.warning(f"Ошибка OI {ticker}: {e}")
        return {"value": 0.0, "chg_24h": 0.0}


def get_full_market_snapshot() -> dict:
    """Собирает полный снимок рынка и портфеля.

    Возвращает словарь со всеми данными: активы, макро, доминация,
    календарь, страх/жадность и детали портфеля с PnL.
    """
    assets_config = load_assets_config()
    if not assets_config:
        return {}

    market_data = fetch_market_data(list(assets_config.values()))
    if not market_data:
        return {}

    snapshot = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "assets": {},
        "macro": {},
        "dominance": {},
        "calendar": [],
        "fear_and_greed": {},
        "portfolio": {
            "total_invested": 0.0,
            "total_current_value": 0.0,
            "cash": 0.0,
            "details": {},
        },
    }

    for ticker, asset_id in assets_config.items():
        if asset_id not in market_data:
            continue

        data = market_data[asset_id]
        rsi_4h, rsi_1d, rsi_1w, rsi_1m = fetch_all_rsis(ticker)
        oi = fetch_open_interest(ticker)

        # Взвешенный фандинг: (ставка в долях) * (OI в USD)
        funding = fetch_funding_rate(ticker)
        weighted_funding = (funding / 100) * oi["value"] if funding is not None and oi["value"] else 0.0

        snapshot["assets"][ticker] = {
            "price": data.get("current_price", 0),
            "chg_24h": data.get("price_change_percentage_24h") or 0,
            "chg_7d": data.get("price_change_percentage_7d_in_currency") or 0,
            "chg_30d": data.get("price_change_percentage_30d_in_currency") or 0,
            "rsi_4h": rsi_4h,
            "rsi_1d": rsi_1d,
            "rsi_1w": rsi_1w,
            "rsi_1m": rsi_1m,
            "funding": funding,
            "oi_value": oi["value"],
            "oi_chg_24h": oi["chg_24h"],
            "weighted_funding": weighted_funding,
        }

    # Добавляем макро-данные и остальные блоки
    snapshot["macro"] = fetch_macro_assets()
    snapshot["dominance"] = fetch_crypto_dominance()
    snapshot["calendar"] = fetch_economic_calendar()
    snapshot["fear_and_greed"] = fetch_fear_and_greed()

    # Обработка портфеля
    portfolio_raw = load_portfolio()
    if not portfolio_raw:
        return snapshot

    cash = float(portfolio_raw.get("USDT", 0))
    snapshot["portfolio"]["cash"] = cash
    total_invested = 0.0
    total_current = 0.0

    for ticker, info in portfolio_raw.items():
        # Пропускаем служебные поля и журнал операций
        if ticker in ("USDT", "initial_capital", "journal") or ticker not in snapshot["assets"]:
            continue

        current_price = snapshot["assets"][ticker]["price"]
        amount = float(info.get("amount", 0))
        avg_price = float(info.get("avg_price", 0))

        current_val = amount * current_price
        cost_val = amount * avg_price
        pnl = current_val - cost_val

        total_invested += cost_val
        total_current += current_val

        snapshot["portfolio"]["details"][ticker] = {
            "amount": amount,
            "avg_price": avg_price,
            "current_val": current_val,
            "pnl": pnl,
            "pnl_pct": (pnl / cost_val * 100) if cost_val > 0 else 0.0,
        }

    snapshot["portfolio"]["total_invested"] = total_invested
    snapshot["portfolio"]["total_current_value"] = total_current

    # Обработка журнала операций
    journal = portfolio_raw.get("journal", [])
    snapshot["portfolio"]["journal"] = journal

    # Считаем реализованный PnL напрямую из журнала
    journal_realized_pnl = sum(
        item.get("amount_usd", 0.0)
        for item in journal
        if item.get("type") in ("realized_loss", "realized_profit")
    )

    initial_capital = float(portfolio_raw.get("initial_capital", total_invested + cash))
    unrealized_pnl = total_current - total_invested
    total_balance = total_current + cash
    total_pnl = total_balance - initial_capital

    # Используем журнал, если он есть, иначе математическую разницу
    realized_pnl = journal_realized_pnl if journal else (total_invested + cash) - initial_capital

    snapshot["portfolio"]["initial_capital"] = initial_capital
    snapshot["portfolio"]["unrealized_pnl"] = unrealized_pnl
    snapshot["portfolio"]["realized_pnl"] = realized_pnl
    snapshot["portfolio"]["total_balance"] = total_balance
    snapshot["portfolio"]["total_pnl"] = total_pnl
    snapshot["portfolio"]["total_pnl_pct"] = (
        (total_pnl / initial_capital * 100) if initial_capital > 0 else 0.0
    )

    return snapshot