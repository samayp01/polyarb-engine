"""
Backtesting

Simulates trading signals on historical market data to evaluate strategy performance.

Tests whether semantically similar markets tend to resolve the same way
(both YES or both NO).
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import BACKTEST_SAVE_RESULTS, MIN_EMBEDDING_SIMILARITY
from .utils.client import fetch_closed_markets
from .utils.embeddings import cluster_markets
from .utils.models import Outcome

logger = logging.getLogger(__name__)

BACKTEST_RESULTS_FILE = Path("data/backtest_results.json")


@dataclass
class Trade:
    """A simulated trade."""

    source_id: str
    target_id: str
    direction: str  # BUY or SELL
    entry_price: float
    exit_price: float
    similarity: float

    @property
    def pnl(self):
        """Profit/loss in price units."""
        if self.direction == "BUY":
            return self.exit_price - self.entry_price
        return self.entry_price - self.exit_price

    @property
    def won(self):
        return self.pnl > 0


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    trades: list
    total_markets: int
    related_pairs: int

    @property
    def total_trades(self):
        return len(self.trades)

    @property
    def winning_trades(self):
        return sum(1 for t in self.trades if t.won)

    @property
    def win_rate(self):
        if not self.trades:
            return 0.0
        return self.winning_trades / len(self.trades)

    @property
    def total_pnl(self):
        return sum(t.pnl for t in self.trades)

    @property
    def avg_pnl(self):
        if not self.trades:
            return 0.0
        return self.total_pnl / len(self.trades)

    def summary(self):
        """Return summary string."""
        lines = [
            f"Markets analyzed: {self.total_markets}",
            f"Related pairs found: {self.related_pairs}",
            f"Trades simulated: {self.total_trades}",
            f"Win rate: {self.win_rate:.1%}",
            f"Total PnL: {self.total_pnl:+.2f}",
            f"Avg PnL per trade: {self.avg_pnl:+.3f}",
        ]
        return "\n".join(lines)


def run_backtest(min_similarity=None, max_markets=500, verbose=False):
    """
    Run backtest on historical closed markets.

    Tests whether semantically similar markets resolve the same way.
    For each pair of similar markets, we check if they both resolved YES
    or both resolved NO (agreement) vs opposite outcomes (disagreement).

    A high agreement rate suggests the correlation-based trading strategy
    has predictive value.

    Args:
        min_similarity: Minimum similarity to consider markets related.
        max_markets: Maximum number of markets to fetch.
        verbose: If True, print example pairs for debugging.

    Returns:
        BacktestResult with trades and statistics.
    """
    if min_similarity is None:
        min_similarity = MIN_EMBEDDING_SIMILARITY

    logger.info("Fetching closed markets...")
    markets = fetch_closed_markets(limit=100, max_pages=max_markets // 100)
    logger.info(f"Fetched {len(markets)} closed markets")

    if len(markets) < 10:
        logger.warning("Not enough markets for backtest")
        return BacktestResult(trades=[], total_markets=len(markets), related_pairs=0)

    # Filter to only clearly resolved markets (price > 0.9 or < 0.1)
    clear_markets = [m for m in markets if m.yes_price > 0.9 or m.yes_price < 0.1]
    logger.info(f"Clearly resolved markets: {len(clear_markets)} of {len(markets)}")

    # Find related pairs
    logger.info("Finding related market pairs...")
    pairs = cluster_markets(clear_markets, min_similarity=min_similarity)
    logger.info(f"Found {len(pairs)} related pairs")

    if verbose and pairs:
        print("\nExample pairs:")
        for market_a, market_b, sim in pairs[:5]:
            outcome_a = "YES" if market_a.yes_price > 0.5 else "NO"
            outcome_b = "YES" if market_b.yes_price > 0.5 else "NO"
            agreed = "+" if outcome_a == outcome_b else "-"
            print(f"  [{agreed}] sim={sim:.2f}")
            print(f"      A: {market_a.question[:60]}... -> {outcome_a}")
            print(f"      B: {market_b.question[:60]}... -> {outcome_b}")

    # Test correlation hypothesis: do similar markets resolve the same way?
    trades = []
    all_results = []

    for market_a, market_b, similarity in pairs:
        trade = _test_correlation(market_a, market_b, similarity)
        trades.append(trade)

        # Record for saving
        outcome_a = "YES" if market_a.yes_price > 0.5 else "NO"
        outcome_b = "YES" if market_b.yes_price > 0.5 else "NO"
        all_results.append(
            {
                "source_id": market_a.id,
                "source_question": market_a.question,
                "source_outcome": outcome_a,
                "target_id": market_b.id,
                "target_question": market_b.question,
                "target_outcome": outcome_b,
                "similarity": similarity,
                "agreed": outcome_a == outcome_b,
            }
        )

    logger.info(f"Tested {len(trades)} market pairs")

    # Save results if enabled
    if BACKTEST_SAVE_RESULTS and all_results:
        _save_backtest_results(all_results, trades, len(clear_markets), min_similarity)

    return BacktestResult(
        trades=trades,
        total_markets=len(clear_markets),
        related_pairs=len(pairs),
    )


def _test_correlation(market_a, market_b, similarity):
    """
    Test if two similar markets resolved the same way.

    We're testing the hypothesis: "semantically similar markets should
    resolve with the same outcome (both YES or both NO)."

    Returns a Trade where:
    - direction = "BUY" (we bet they agree)
    - entry_price = 0.5 (fair odds)
    - exit_price = 1.0 if they agreed, 0.0 if they disagreed
    - pnl = +0.5 if correct, -0.5 if wrong
    """
    outcome_a = Outcome.YES if market_a.yes_price > 0.5 else Outcome.NO
    outcome_b = Outcome.YES if market_b.yes_price > 0.5 else Outcome.NO

    # Did they resolve the same way?
    agreed = outcome_a == outcome_b

    return Trade(
        source_id=market_a.id,
        target_id=market_b.id,
        direction="BUY",  # We're betting on agreement
        entry_price=0.5,  # Fair odds baseline
        exit_price=1.0 if agreed else 0.0,
        similarity=similarity,
    )


def _save_backtest_results(all_results, trades, total_markets, min_similarity):
    """Save backtest results to JSON file for inspection."""
    BACKTEST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)

    agreed_count = sum(1 for r in all_results if r["agreed"])

    data = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "min_similarity": min_similarity,
        },
        "summary": {
            "total_markets": total_markets,
            "total_pairs_analyzed": len(all_results),
            "agreed": agreed_count,
            "disagreed": len(all_results) - agreed_count,
            "agreement_rate": agreed_count / len(all_results) if all_results else 0,
            "trades_executed": len(trades),
        },
        "pairs": all_results,
    }

    with BACKTEST_RESULTS_FILE.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved backtest results to {BACKTEST_RESULTS_FILE}")
