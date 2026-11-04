import html
import json
from datetime import datetime
from typing import Any, Dict, List, Optional


# --- Вспомогательные функции форматирования чисел ---

def format_price(price: float) -> str:
    """Форматирует цену актива в зависимости от её величины."""
    if price >= 1:
        return f"${price:,.2f}"
    return f"${price:,.4f}"


def format_money(value: float) -> str:
    """Форматирует денежные суммы (баланс, PnL и т.д.)."""
    return f"${value:,.2f}"


def _fmt_rsi(value: Optional[float]) -> str:
    """Приводит значение RSI к целому числу или прочерку."""
    return f"{value:.0f}" if value is not None else "-"


def _fmt_funding(value: Optional[float]) -> str:
    """Форматирует ставку фандинга."""
    return f"{value:+.3f}%" if value is not None else "N/A"


# --- Маппинг действий и направлений в человекочитаемый вид ---

def _fmt_trading_direction(direction: str) -> str:
    """Превращает 'wait|long|short' в человекочитаемую метку."""
    mapping = {
        "wait": "⏳ ЖДАТЬ",
        "long": "🟢 ЛОНГ",
        "short": "🔴 ШОРТ",
    }
    return mapping.get((direction or "").lower(), "⏳ ЖДАТЬ")


def _fmt_portfolio_action(action: str) -> str:
    """Превращает 'hold|average_down|sell|wait|buy' в человекочитаемую метку."""
    mapping = {
        "hold": "🟡 ДЕРЖАТЬ",
        "average_down": "🟢 УСРЕДНЯТЬ",
        "sell": "🔴 ПРОДАВАТЬ",
        "wait": "⏳ ЖДАТЬ",
        "buy": "🔵 КУПИТЬ",
    }
    return mapping.get((action or "").lower(), "⏳ ЖДАТЬ")


def _fmt_rating(rating: str) -> str:
    """Превращает 'interesting|neutral|not_interesting' в метку для внепортфельных активов."""
    mapping = {
        "interesting": "🟢 ИНТЕРЕСЕН",
        "neutral": "⚪ НЕЙТРАЛЕН",
        "not_interesting": "🔴 НЕ ИНТЕРЕСЕН",
    }
    return mapping.get((rating or "").lower(), "")


def _fmt_risk_reward(value: Any) -> str:
    """Форматирует R:R в вид '1:2.5'. Если 0 или пусто — возвращает прочерк."""
    try:
        rr = float(value)
    except (TypeError, ValueError):
        return "—"
    if rr <= 0:
        return "—"
    return f"1:{rr:.1f}"


def _fmt_journal_type(jtype: str) -> str:
    """Человекочитаемая метка для типа записи журнала."""
    mapping = {
        "realized_loss": "📉 Убыток зафиксирован",
        "realized_profit": "📈 Прибыль зафиксирована",
        "average_down": "🟢 Усреднение",
        "buy": "🔵 Покупка",
        "sell": "🔴 Продажа",
    }
    return mapping.get((jtype or "").lower(), f"• {jtype}")


def _short_date(date_str: str) -> str:
    """Из '2026-11-04' делает '04.11'. Если формат не тот — возвращает как есть."""
    try:
        dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        return dt.strftime("%d.%m")
    except (ValueError, TypeError):
        return date_str


# --- Сборщики сообщений для Telegram ---

def build_market_message(snapshot: Dict[str, Any]) -> str:
    """Формирует HTML-отчёт по рынку и портфелю (технический блок)."""
    blocks = [
        f"📊 <b>MARKET REPORT</b> — {snapshot['timestamp']}\n━━━━━━━━━━━━━━━━━━━━━━━━\n"
    ]

    macro = snapshot.get("macro", {})
    if macro:
        blocks.append("🌍 <b>MACRO</b>")
        for name, data in macro.items():
            blocks.append(f"{name}: <code>{data['price']}</code> ({data['chg_24h']:+.1f}%)")
        dom = snapshot.get("dominance", {})
        blocks.append(f"BTC.D: {dom.get('BTC.D', 0)}% | USDT.D: {dom.get('USDT.D', 0)}%\n")

    fng = snapshot.get("fear_and_greed", {})
    if fng:
        blocks.append(f"😨 <b>Fear & Greed:</b> {fng.get('value')} ({fng.get('classification')})\n")

    cal = snapshot.get("calendar", [])
    if cal:
        blocks.append("📅 <b>СОБЫТИЯ</b>")
        for ev in cal:
            blocks.append(f"• [{ev['country']}] {ev['date'][:10]}: {ev['title']}")
        blocks.append("")

    blocks.append("━━━━━━━━━━━━━━━━━━━━━━━━\n🔹 <b>АКТИВЫ</b>")
    for ticker, data in snapshot["assets"].items():
        rsi_str = (
            f"{_fmt_rsi(data['rsi_4h'])}/{_fmt_rsi(data['rsi_1d'])}/"
            f"{_fmt_rsi(data['rsi_1w'])}/{_fmt_rsi(data['rsi_1m'])}"
        )
        oi_b = data["oi_value"] / 1_000_000_000
        blocks.append(
            f"🔹 <b>{ticker}</b>\n"
            f"💰 <code>{format_price(data['price'])}</code>\n"
            f"📈 24h: {data['chg_24h']:+.1f}% | 7d: {data['chg_7d']:+.1f}% | 30d: {data['chg_30d']:+.1f}%\n"
            f"📉 RSI: <code>{rsi_str}</code>\n"
            f"⚖️ Фандинг: <code>{_fmt_funding(data['funding'])}</code>\n"
            f"💸 Взвеш.: <code>${data.get('weighted_funding', 0):,.0f}</code>\n"
            f"🏦 OI: <code>${oi_b:.2f}B ({data['oi_chg_24h']:+.1f}%)</code>\n"
        )

    # --- Портфель ---
    p = snapshot["portfolio"]
    if p.get("details"):
        blocks.append("💼 <b>ПОРТФЕЛЬ</b>\n━━━━━━━━━━━━━━━━━━━━━━━━")

        for ticker, info in p["details"].items():
            sign = "+" if info["pnl"] >= 0 else ""
            emoji = "🟢" if info["pnl"] >= 0 else "🔴"
            cur = snapshot["assets"][ticker]["price"]
            blocks.append(
                f"{emoji} <b>{ticker}</b>: {info['amount']} шт.\n"
                f"Средняя: ${info['avg_price']:,.4f} | Текущая: ${cur:,.4f}\n"
                f"Стоимость: <code>{format_money(info['current_val'])}</code>\n"
                f"PnL: <code>{sign}{format_money(info['pnl'])} ({sign}{info['pnl_pct']:.1f}%)</code>\n"
            )

        total_sign = "+" if p["total_pnl"] >= 0 else ""
        real_sign = "+" if p["realized_pnl"] >= 0 else ""
        unreal_sign = "+" if p.get("unrealized_pnl", 0) >= 0 else ""

        blocks.append(
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🏦 Капитал: <code>{format_money(p['initial_capital'])}</code>\n"
            f"├ Вложено: {format_money(p['total_invested'])}\n"
            f"├ Кэш: {format_money(p['cash'])}\n"
            f"└ Зафиксировано: <b>{real_sign}{format_money(p['realized_pnl'])}</b>\n\n"
            f"📊 PnL открытых: {unreal_sign}{format_money(p.get('unrealized_pnl', 0))}\n"
            f"📊 Общий PnL: <code>{total_sign}{format_money(p['total_pnl'])} ({total_sign}{p['total_pnl_pct']:.1f}%)</code>\n"
            f"💰 <b>Баланс: <code>{format_money(p['total_balance'])}</code></b>"
        )

        journal = p.get("journal", []) or []
        if journal:
            blocks.append("\n📝 <b>Последние операции:</b>")
            for j in journal[-3:]:
                jtype = _fmt_journal_type(j.get("type", ""))
                asset = j.get("asset", "")
                date = _short_date(j.get("date", ""))
                amount = j.get("amount_usd")
                if amount is not None:
                    sign = "+" if amount >= 0 else ""
                    blocks.append(
                        f"• <b>{date}</b> · {asset} · {jtype} "
                        f"(<code>{sign}{format_money(amount)}</code>)"
                    )
                else:
                    blocks.append(f"• <b>{date}</b> · {asset} · {jtype}")

    return "\n".join(blocks)


def build_ai_main_message(ai_result: Dict[str, Any]) -> str:
    """Формирует основное сообщение: анализ, исторические аналогии,
    торговый план и совет по портфелю."""
    chart_reading = html.escape(str(ai_result.get("chart_reading", "") or ""))
    analysis_steps = ai_result.get("analysis_steps", []) or []
    historical_analogies = ai_result.get("historical_analogies", []) or []
    trading = ai_result.get("trading", {}) or {}
    portfolio_advice = ai_result.get("portfolio_advice", {}) or {}
    confidence = ai_result.get("confidence", 0)
    regime = html.escape(str(ai_result.get("market_regime", "") or ""))
    scenarios = ai_result.get("scenarios", {}) or {}
    triggers = ai_result.get("triggers_to_watch", []) or []
    invalidators = ai_result.get("invalidation_factors", []) or []

    # --- Шапка ---
    timestamp = datetime.now().strftime("%d.%m · %H:%M")
    meta_parts = []
    if confidence:
        meta_parts.append(f"<b>{confidence}%</b>")
    if regime:
        meta_parts.append(html.escape(regime))
    meta_line = " · ".join(meta_parts)

    parts = [
        f"🕵️ <b>Edward \"Teddy\" Daniels</b>",
        f"<i>{timestamp}{' · ' + meta_line if meta_line else ''}</i>",
        "",
    ]

    # --- Блок «Как я думал» ---
    if analysis_steps:
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append("🧠 <b>КАК Я ДУМАЛ</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        for step in analysis_steps:
            parts.append(f"› {html.escape(str(step))}")
        parts.append("")

    # --- Блок «Исторические аналогии» ---
    if historical_analogies:
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append("📚 <b>ИСТОРИЧЕСКИЕ АНАЛОГИИ</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        for analogy in historical_analogies:
            if not isinstance(analogy, dict):
                continue
            event = html.escape(str(analogy.get("event", "")))
            similarity = html.escape(str(analogy.get("similarity", "")))
            difference = html.escape(str(analogy.get("difference", "")))
            pattern = html.escape(str(analogy.get("pattern", "")))

            parts.append(f"▸ <b>{event}</b>")
            if similarity:
                parts.append(f"  <b>Похоже:</b> {similarity}")
            if difference:
                parts.append(f"  <b>Отличается:</b> {difference}")
            if pattern:
                parts.append(f"  <b>Паттерн:</b> {pattern}")
            parts.append("")

    # --- Блок «Трейдинг» ---
    if trading:
        asset = html.escape(str(trading.get("asset", "BTC")))
        direction_raw = (trading.get("direction") or "").lower()
        direction = _fmt_trading_direction(direction_raw)
        stop = html.escape(str(trading.get("stop_loss", "—")))
        tp_list = trading.get("take_profit", []) or []
        tp_str = " → ".join(html.escape(str(t)) for t in tp_list) if tp_list else "—"
        rr_str = _fmt_risk_reward(trading.get("risk_reward", 0))
        trading_reason = html.escape(str(trading.get("reasoning", "")))

        # При wait: если есть рекомендованный entry — показываем его вместо пустого
        rec_entry = trading.get("_recommended_entry")
        if direction_raw == "wait" and rec_entry:
            try:
                entry = f"⏳ ждём зону ${float(rec_entry):,.0f}"
            except (TypeError, ValueError):
                entry = "⏳ ждём зону"
        else:
            entry = html.escape(str(trading.get("entry", "—")))

        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append(f"📈 <b>ТРЕЙДИНГ · {asset}</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append(f"Позиция:  <b>{direction}</b>")
        parts.append(f"Вход:     <code>{entry}</code>")
        parts.append(f"Стоп:     <code>{stop}</code>")
        parts.append(f"Тейк:     <code>{tp_str}</code>")
        parts.append(f"R:R:      <code>{rr_str}</code>")
        if trading_reason:
            parts.append(f"<i>{trading_reason}</i>")
        parts.append("")

    # --- Блок «Портфель» ---
    if portfolio_advice:
        portfolio_data = ai_result.get("_portfolio_snapshot") or {}
        details = portfolio_data.get("details", {}) or {}

        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append("💼 <b>ПОРТФЕЛЬ · ИНВЕСТОР</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        for ticker, advice in portfolio_advice.items():
            if ticker == "cash":
                continue
            if not isinstance(advice, dict):
                continue

            action = _fmt_portfolio_action(advice.get("action", ""))
            reason = html.escape(str(advice.get("reason", "")))
            rating = advice.get("rating", "")
            rating_label = _fmt_rating(rating)

            asset_info = details.get(ticker, {})
            if asset_info:
                # Актив в портфеле
                pnl_pct = asset_info.get("pnl_pct", 0)
                emoji = "🟢" if pnl_pct >= 0 else "🔴"
                price = asset_info.get("current_price", 0)
                if not price:
                    price = asset_info.get("current_val", 0) / max(asset_info.get("amount", 1), 1)
                parts.append(
                    f"{emoji} <b>{ticker}</b> · ${price:,.4f} · {pnl_pct:+.1f}%"
                )
                parts.append(f"  {action}")
            else:
                # Актив вне портфеля
                if rating_label:
                    parts.append(f"⚪ <b>{ticker}</b> · {rating_label}")
                else:
                    parts.append(f"⚪ <b>{ticker}</b> · <i>не в портфеле</i>")
                parts.append(f"  {action}")

            if reason:
                parts.append(f"  <i>{reason}</i>")
            parts.append("")

        cash_advice = portfolio_advice.get("cash")
        if isinstance(cash_advice, dict) and cash_advice:
            action = _fmt_portfolio_action(cash_advice.get("action", ""))
            reason = html.escape(str(cash_advice.get("reason", "")))
            parts.append("💵 <b>Кэш</b>")
            parts.append(f"  {action}")
            if reason:
                parts.append(f"  <i>{reason}</i>")
            parts.append("")

    # --- Блок «Контекст» ---
    if scenarios or triggers or invalidators:
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append("📊 <b>КОНТЕКСТ</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")

        labels = {"base_case": "базовый", "bull_case": "бычий", "bear_case": "медвежий"}
        scenario_parts = []
        for key, label in labels.items():
            sc = scenarios.get(key) or {}
            prob = sc.get("probability", 0)
            if prob:
                scenario_parts.append(f"{label} {prob}%")
        if scenario_parts:
            parts.append(f"Сценарии: {' · '.join(scenario_parts)}")

        if triggers:
            parts.append("<b>Триггеры:</b>")
            for t in triggers:
                parts.append(f"› {html.escape(str(t))}")

        if invalidators:
            parts.append("<b>Инвалидаторы:</b>")
            for inv in invalidators:
                parts.append(f"› {html.escape(str(inv))}")

        parts.append("")

    # --- Блок «Чтение графика» ---
    if chart_reading:
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append("🖼 <b>ЧТЕНИЕ ГРАФИКА</b>")
        parts.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        parts.append(f"<i>{chart_reading}</i>")

    return "\n".join(parts).rstrip()


def build_ai_details_message(ai_result: Dict[str, Any]) -> str:
    """Оставлено для совместимости. Возвращает пустую строку."""
    return ""


def build_market_text_for_ai(snapshot: Dict[str, Any]) -> str:
    """Формирует компактный текстовый контекст по рынку для LLM."""
    lines = []
    for ticker, data in snapshot["assets"].items():
        lines.append(
            f"{ticker}: ${data['price']}, "
            f"RSI: {_fmt_rsi(data['rsi_4h'])}/{_fmt_rsi(data['rsi_1d'])}/"
            f"{_fmt_rsi(data['rsi_1w'])}/{_fmt_rsi(data['rsi_1m'])}, "
            f"Funding: {_fmt_funding(data['funding'])}, "
            f"WeightedFunding: {data.get('weighted_funding', 0):,.0f}, "
            f"OI: {data['oi_chg_24h']:+.1f}%"
        )
    lines.append(f"\nMacro: {json.dumps(snapshot.get('macro', {}))}")
    lines.append(f"Dominance: {json.dumps(snapshot.get('dominance', {}))}")
    lines.append(f"Fear & Greed: {json.dumps(snapshot.get('fear_and_greed', {}))}")
    lines.append(f"Upcoming Events: {json.dumps(snapshot.get('calendar', []))}")
    lines.append(f"Balance: ${snapshot['portfolio']['total_balance']}")

    return "\n".join(lines)