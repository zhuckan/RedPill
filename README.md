# RedPill

A Telegram bot for crypto trading that I built for myself and now use daily. You send it a screenshot of a TradingView chart, and it gives you back a market analysis, a trading plan for BTC, and a short note on what to do with your portfolio. After a few days you tell it what actually happened — and it remembers.

I started it because I wanted a second opinion on my positions that wasn't emotionally attached to them. Over a few months of iterating, it turned into something I think has commercial value: an agent that produces numerical decisions (entry, stop, target, R:R) where the *discipline* is enforced in Python rather than trusted to the LLM, and that keeps a persistent journal of its own past calls and outcomes. The combination — LLM for reasoning, code for discipline, a vector store for memory — is not crypto-specific. I happened to build it for crypto because that's the market I know, but the same pattern applies anywhere a wrong answer has a real cost.

---

## What it actually does

When you send a chart, this is roughly the pipeline:

1. The image goes to Gemini along with a legend explaining my custom TradingView markings (pink trendlines, purple Fibo levels, Volume Profile bands, specific MAs).
2. It pulls live data from Binance and CoinGecko: price, RSI on four timeframes, funding rate, open interest, weighted funding, plus macro (SPX, DXY, gold, oil), BTC dominance, Fear & Greed, and this week's economic calendar.
3. It reads the last N messages from a Telegram news channel I follow, stores them in Qdrant, and retrieves whatever looks most relevant to the current situation.
4. If the local news doesn't explain the move, it does a quick Exa search.
5. It runs the whole thing through a LangGraph workflow where Gemini drafts a prediction, a second Gemini call plays devil's advocate and checks it, and then Python code applies hard risk rules on top.
6. The result is a structured message with: what I looked at, historical analogies, a BTC trading plan, portfolio advice, and scenarios.
7. The prediction is saved to Qdrant with an ID. A few days later I reply in plain text with what happened, and the agent updates the record.

The part I care about most is step 7. Without it, this is just another AI that sounds confident. With it, the agent builds up a memory of its own mistakes and stops repeating them.

---

## Where this pattern applies

- **Prop trading desks and small crypto funds.** The R:R guard and the self-learning journal map directly onto how a desk already thinks about risk. The agent is a fast first-pass analyst that never forgets the last time it was wrong about a setup.
- **Crypto research and advisory shops.** The pipeline (live metrics → news → historical analogs → structured JSON) is close to what a research team produces manually, but consistently and in seconds.
- **Crypto wealth managers.** Portfolio advice with a real journal (positions, averaged-down entries, realized losses) is closer to what an RIA does than what a "trading bot" does.
- **Educational platforms.** Historical analogies and structured chart reading make good teaching material.
- **Any agent that outputs numbers.** Swap BTC for equities, commodities, or credit — the discipline pattern and the journal design are the same.

---

## Why the R:R rules are enforced in Python

The single biggest problem with LLM-driven trading advice is that the model is happy to draw you a plan that looks reasonable and has terrible risk/reward. It'll happily say "long at $77.5k, stop at $71k, first target at $82k" — that's a 1:0.6 setup, and no amount of prompting reliably fixes it.

So I moved the important part into code. Every plan the LLM produces goes through `_fix_risk_reward()` in `core/ai_agent.py`, which does the following:

- Computes R:R at the **worst** entry point of the range — the top of the range for a long, the bottom for a short. This is what you'd actually pay if the market fills you, not the midpoint.
- If R:R is below 2.0, the plan gets forced into `direction = "wait"`, the entry is cleared, and the code calculates the entry that *would* give 2.0 from the same stop and target.
- If the entry range is wider than 2 %, it gets narrowed to 0.8 % around its midpoint. Wide ranges are usually a sign the model is hedging, not committing.
- If the entry is more than 2 % away from the current price, it becomes `wait`. It doesn't matter how good the setup is if you'd need the price to move 5 % before you can enter — that's not a plan, that's a wish.
- Stops that land inside the bear-case target (for longs) or above the bull-case (for shorts) get flagged. Otherwise you'd be long with a stop that's *below* where you expect the market to go if you're wrong.

What the user sees in Telegram is the corrected version. If the plan was bad, they see "⏳ ждём зону $77,467" with the reasoning explaining what the original plan was and why it didn't pass.

This isn't perfect — the model still occasionally produces garbage — but at least garbage doesn't reach Telegram dressed up as a plan.

---

## Historical analogies

The agent has a curated set of 39 historical market events in `knowledge_base/events/manipulations.json`: COVID crash, LUNA/UST, FTX, 3AC, the CME futures launch, BlackRock's ETF filing, the Blue Sweep midterms, the Binance exploit, and so on. Each one is structured as `trigger`, `pre_sentiment`, `reaction`, `liquidations`, `post_sentiment`, `trajectory`, `lesson` — not just a date and a headline.

When the agent analyzes a chart, it doesn't just do a semantic search over these events and call it a day. It receives the **entire** digest in the prompt (name, date, type, lesson) and is instructed to pick the 1–3 events that actually match the current situation, with a similarity / difference / pattern breakdown for each. This was a fix I made after watching it cheerfully compare everything to "Fed Rate Cut 2025" regardless of context.

The output goes into a `historical_analogies` field in the JSON and gets rendered as a separate block in Telegram. It's the most useful part of the response for me — not because historical parallels predict the future, but because they remind me of specific traps I've forgotten about.

---

## What it knows about my portfolio

The bot reads `portfolio.json`. It has my open positions with average prices, cash, and a `journal` array with past operations like "averaged down SOL" or "realized loss on DOGE". The realized PnL comes straight from the journal — it doesn't try to reconstruct it from initial_capital minus current balance, which was always slightly off thanks to commissions.

For assets I own, the agent gives `hold / average_down / sell / wait` with a short reason. It also sees the journal, so it won't cheerfully tell me to average down the same asset I averaged down yesterday.

For assets I'm watching but don't own (currently LINK, XRP, ASTER), it gives one of three verdicts — `interesting`, `neutral`, or `not_interesting` — plus a concrete analysis: RSI across timeframes, funding and OI reading, a specific price level, and a trigger. I got tired of reading "asset outside the portfolio, monitoring reactions" as an answer, so I explicitly forbade that phrasing in the prompt.

---

## Stack

Nothing exotic.

- **aiogram 3** for the bot. One owner, no buttons except a portfolio shortcut.
- **LangGraph** for the workflow. Four nodes plus a verify step, all synchronous.
- **Gemini 3.8 Flash** with a fallback to 3.5 Flash Lite. Vision for charts.
- **fastembed** with `multilingual-e5-large` (1024 dims). Runs locally, no API cost.
- **Qdrant** in Docker. Four collections: predictions, news, knowledge, portfolio history.
- **Langfuse** for tracing. Optional but useful — every prediction is a trace with prompt size, KV-cache estimate, and chat session grouping.
- **Exa** for web search fallback.
- **Telethon** for reading the news channel.

---

## Setup

You'll need Python 3.12 (I use `uv`, but plain `pip` works), Docker for Qdrant, and API keys for Telegram, Gemini, and Exa. The news parser also needs a Telegram `api_id`/`api_hash` from `my.telegram.org`.

```bash
cp .env.example .env
# fill in the keys

docker compose up -d              # Qdrant on :6333
uv sync                           # or python -m venv .venv && pip install -r requirements.txt

python -m scripts.load_mentor     # loads a mentor article into Qdrant
python -m scripts.load_events     # loads the 39 historical events

python bot.py
```

First run of the news parser takes about 9 minutes — it pulls ~1000 messages from the channel. Subsequent runs are 5–15 seconds.

The bot is single-user. `TELEGRAM_CHAT_ID` in `.env` restricts it to one chat; messages from anyone else are silently ignored.

---

## Project layout

```
RedPill/
├── bot.py                      # Telegram entry point
├── config/settings.py          # All env vars + constants
├── core/
│   ├── ai_agent.py             # LangGraph + R:R guard
│   ├── formatters.py           # Telegram HTML
│   ├── market_data.py          # Binance, CoinGecko, portfolio math
│   ├── macro_data.py           # Yahoo Finance, F&G, calendar
│   ├── news_parser.py          # Telethon → MD + Qdrant
│   └── rag_memory.py           # Qdrant collections + events digest
├── scripts/
│   ├── load_mentor.py
│   ├── load_events.py
│   ├── reset_runtime.py        # Wipes predictions + chat history
│   ├── reset_knowledge_base.py # Wipes everything, including events
│   ├── dump_chunks.py
│   ├── visualize_embeddings.py # UMAP / t-SNE
│   ├── evaluate_rag.py
│   └── evaluate_retrieval.py
├── knowledge_base/
│   ├── events/manipulations.json
│   ├── chart_legend.md
│   ├── mentor/                 # gitignored
│   └── news/                   # gitignored
├── portfolio.json              # gitignored
├── assets.json                 # which tickers to track
├── docker-compose.yml          # Qdrant
└── .env                        # gitignored
```

`assets.json` maps tickers to CoinGecko slugs, e.g. `{"ASTER": "aster-2"}`. Adding a new asset is just adding a line here — the pipeline picks it up on the next run.

---

## Things I know are still rough

- **Wide entry ranges.** The code narrows them, but the model still tries to draw $74k–$77k sometimes. A better prompt or a stricter numeric pre-check on the model side would help.
- **The second Gemini call (verify) doesn't add as much value as I hoped.** It catches obvious contradictions, but it also occasionally rewrites a decent plan into a worse one. I keep it because the cost is low and it saves me from the worst outputs, but I don't think it's essential.
- **The `historical_analogies` field sometimes picks a case that's only superficially similar.** Not a huge deal — I read them, I ignore the bad ones — but there's room for a better retrieval strategy (e.g. embedding the current market snapshot and searching by similarity first, then letting the model re-rank).
- **The economic calendar is noisy.** Forex Factory occasionally rate-limits, and the parser silently skips it.
- **`yfinance` prints a wall of `Pandas4Warning`s on startup.** Cosmetic, I've been too lazy to filter them out.

---

## Honest disclaimer

This is a tool I use to organize my own thinking about a market I find interesting. It is not investment advice, and the fact that it sounds confident doesn't mean it's right. If you build your own copy, please don't trade real money on its output until you've watched it be wrong a few times and understand exactly how it fails.