"""
Backtest Engine

Evaluates the quality of learned market relationships using historical data.

Proper methodology:
1. Split markets into train (70%) and test (30%) sets
2. Build correlation patterns from training set only
3. For each test market, predict outcome based on correlated training markets
4. Compare predictions to actual test outcomes
"""

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path

from .deterministic_bootstrap import init_deterministic
from .models import BacktestResult, Outcome
from .graph import EventGraph

# Initialize deterministic behavior
init_deterministic(seed=42)

logger = logging.getLogger(__name__)

OUTCOMES_FILE = Path("data/market_outcomes.json")


@dataclass
class Trade:
  """Simulated trade result."""
  test_market_id: str
  train_market_id: str
  similarity: float
  train_outcome: Outcome
  predicted_outcome: Outcome
  actual_outcome: Outcome
  profitable: bool


class BacktestEngine:
  """
  Backtests correlation-based predictions with proper train/test split.

  Splits markets into train/test sets:
  - Learns correlation patterns from training pairs
  - Tests predictions on held-out markets
  """

  def __init__(self, graph=None):
    self.graph = graph or EventGraph()
    self.outcomes = self._load_outcomes()

  def _load_outcomes(self):
    """Load historical market outcomes."""
    if not OUTCOMES_FILE.exists():
      logger.warning(f"No outcomes file found at {OUTCOMES_FILE}")
      return {}

    with open(OUTCOMES_FILE) as f:
      data = json.load(f)

    return {mid: Outcome(val) for mid, val in data.items()}

  def run(self, test_fraction=0.3):
    """
    Run backtest with proper train/test split.

    Args:
      test_fraction: Fraction of markets to hold out for testing.

    Returns:
      BacktestResult with performance metrics.
    """
    edges = self.graph.get_valid()

    if not edges:
      logger.warning("No valid edges to backtest")
      return self._empty_result()

    if not self.outcomes:
      logger.warning("No historical outcomes loaded - run 'python run.py build' first")
      return self._empty_result()

    # Get all unique markets from edges
    all_market_ids = set()
    for e in edges:
      all_market_ids.add(e.from_market_id)
      all_market_ids.add(e.to_market_id)

    # Filter to markets with known outcomes
    markets_with_outcomes = [m for m in all_market_ids if m in self.outcomes]

    if len(markets_with_outcomes) < 10:
      logger.warning("Not enough markets with outcomes for train/test split")
      return self._empty_result()

    # Split markets into train/test
    random.seed(42)
    random.shuffle(markets_with_outcomes)
    split_idx = int(len(markets_with_outcomes) * (1 - test_fraction))
    train_markets = set(markets_with_outcomes[:split_idx])
    test_markets = set(markets_with_outcomes[split_idx:])

    logger.info(f"Train markets: {len(train_markets)}, Test markets: {len(test_markets)}")

    # Get edges where BOTH markets are in training set (learn correlation patterns)
    train_edges = [
      e for e in edges
      if e.from_market_id in train_markets and e.to_market_id in train_markets
    ]

    # Build correlation lookup: for each market pair, did they resolve the same way?
    correlations = {}
    for e in train_edges:
      out1 = self.outcomes.get(e.from_market_id)
      out2 = self.outcomes.get(e.to_market_id)
      if out1 and out2:
        correlated = (out1 == out2)
        key = (e.from_market_id, e.to_market_id)
        correlations[key] = (correlated, e.similarity)

    logger.info(f"Learned {len(correlations)} correlation patterns from training data")

    # Find edges that connect train markets to test markets
    test_edges = [
      e for e in edges
      if (e.from_market_id in train_markets and e.to_market_id in test_markets) or
         (e.from_market_id in test_markets and e.to_market_id in train_markets)
    ]

    logger.info(f"Found {len(test_edges)} edges connecting train to test markets")

    # Make predictions for test markets
    trades = []
    for edge in test_edges:
      trade = self._make_prediction(edge, train_markets, test_markets, correlations)
      if trade:
        trades.append(trade)

    return self._compute_results(trades)

  def _make_prediction(self, edge, train_markets, test_markets, correlations):
    """
    Predict test market outcome based on training data.

    Core hypothesis: Semantically similar markets tend to resolve the same way.
    We DON'T use edge deltas (which would leak test info) - only similarity.
    """
    # Determine which is train and which is test
    if edge.from_market_id in train_markets and edge.to_market_id in test_markets:
      train_id = edge.from_market_id
      test_id = edge.to_market_id
    else:
      train_id = edge.to_market_id
      test_id = edge.from_market_id

    train_outcome = self.outcomes.get(train_id)
    actual_outcome = self.outcomes.get(test_id)

    if not train_outcome or not actual_outcome:
      return None

    # Core hypothesis: similar markets resolve the same way
    # (Don't use delta - it encodes actual outcomes and would leak info)
    predicted = train_outcome

    profitable = (predicted == actual_outcome)

    return Trade(
      test_market_id=test_id,
      train_market_id=train_id,
      similarity=edge.similarity,
      train_outcome=train_outcome,
      predicted_outcome=predicted,
      actual_outcome=actual_outcome,
      profitable=profitable,
    )

  def _empty_result(self):
    """Return empty backtest result."""
    return BacktestResult(
      total_signals=0,
      profitable_signals=0,
      total_pnl=0.0,
      avg_pnl_per_signal=0.0,
      hit_rate=0.0,
      avg_decay_vs_lag=0.0,
    )

  def _compute_results(self, trades):
    """Compute backtest metrics."""
    if not trades:
      return self._empty_result()

    profitable = [t for t in trades if t.profitable]
    hit_rate = len(profitable) / len(trades)

    # Compute PnL (assume fixed bet size, win = +1, lose = -1)
    total_pnl = len(profitable) - (len(trades) - len(profitable))
    avg_pnl = total_pnl / len(trades)

    # Analyze by similarity bucket
    buckets = [
      (0.95, 1.0),
      (0.90, 0.95),
      (0.85, 0.90),
      (0.80, 0.85),
      (0.75, 0.80),
    ]

    bucket_stats = []
    for low, high in buckets:
      bucket_trades = [t for t in trades if low <= t.similarity < high]
      if bucket_trades:
        bucket_correct = sum(1 for t in bucket_trades if t.profitable)
        bucket_rate = bucket_correct / len(bucket_trades)
        bucket_stats.append({
          "range": f"{low:.0%}-{high:.0%}",
          "count": len(bucket_trades),
          "correct": bucket_correct,
          "hit_rate": bucket_rate,
        })

    # Build trade details
    details = []
    for t in trades[:20]:
      details.append({
        "test_market": t.test_market_id,
        "train_market": t.train_market_id,
        "similarity": f"{t.similarity:.1%}",
        "train_outcome": t.train_outcome.name,
        "predicted": t.predicted_outcome.name,
        "actual": t.actual_outcome.name,
        "profitable": t.profitable,
      })

    result = BacktestResult(
      total_signals=len(trades),
      profitable_signals=len(profitable),
      total_pnl=total_pnl / 100,  # Normalize for display
      avg_pnl_per_signal=avg_pnl / 100,
      hit_rate=hit_rate,
      avg_decay_vs_lag=0.0,
      signals=details,
    )

    # Store bucket stats for printing
    result.bucket_stats = bucket_stats
    return result


def print_results(result):
  """Print backtest results."""
  print("\n" + "=" * 60)
  print("BACKTEST RESULTS")
  print("=" * 60)

  if result.total_signals == 0:
    print("\nNo trades to evaluate.")
    return

  print(f"\nTotal predictions: {result.total_signals}")
  print(f"Correct: {result.profitable_signals}")
  print(f"Hit rate: {result.hit_rate:.1%}")

  # Baseline comparison
  baseline = 0.5  # Random guessing
  improvement = (result.hit_rate - baseline) / baseline * 100
  print(f"\nBaseline (random): 50.0%")
  print(f"Improvement: {improvement:+.1f}%")

  # Analyze by similarity threshold
  if hasattr(result, 'bucket_stats') and result.bucket_stats:
    print("\n" + "-" * 60)
    print("ACCURACY BY SIMILARITY THRESHOLD")
    print("-" * 60)
    print("(Does higher similarity = better predictions?)\n")
    for bucket in result.bucket_stats:
      bar_len = int(bucket['hit_rate'] * 40)
      bar = "█" * bar_len + "░" * (40 - bar_len)
      print(f"  {bucket['range']:>10}: {bar} {bucket['hit_rate']:5.1%} ({bucket['count']:>4} predictions)")

  if result.signals:
    print("\n" + "-" * 60)
    print("SAMPLE PREDICTIONS")
    print("-" * 60)

    for t in result.signals[:10]:
      status = "✓" if t["profitable"] else "✗"
      print(f"\n[{status}] Test: {t['test_market'][:12]}...")
      print(f"    Based on: {t['train_market'][:12]}... (sim: {t['similarity']})")
      print(f"    Train resolved: {t['train_outcome']}")
      print(f"    Predicted: {t['predicted']}, Actual: {t['actual']}")
