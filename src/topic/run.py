#!/usr/bin/env python3
"""
Polymarket Trading System

Detects mispricing opportunities in prediction markets by analyzing
relationships between semantically similar markets.

Commands:
  build    - Build event graph from current markets
  monitor  - Monitor for signals in real-time
  backtest - Run backtest on historical data
  ingest   - Start continuous data ingestion
  status   - Show system status

Usage:
  python run.py build
  python run.py monitor
  python run.py backtest
  python run.py ingest
  python run.py status
"""

import sys
import logging

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s - %(levelname)s - %(message)s"
)


def cmd_build():
  """Build the event graph from historical resolved markets."""
  import json
  from pathlib import Path
  from client import PolymarketClient
  from graph import EventGraph, build_historical_edges, print_summary

  print("\n" + "=" * 60)
  print("BUILDING EVENT GRAPH FROM HISTORICAL DATA")
  print("=" * 60)

  # Fetch closed/resolved markets
  client = PolymarketClient()
  print("Fetching resolved markets from Polymarket...")
  closed_markets = client.fetch_closed_markets(limit=100, max_pages=20)
  print(f"Fetched {len(closed_markets)} resolved markets")

  if len(closed_markets) < 10:
    print("Not enough resolved markets to build graph.")
    return

  # Build edges from historical outcomes
  edges, market_outcomes = build_historical_edges(closed_markets, min_similarity=0.75)

  if not edges:
    print("No related market pairs found.")
    return

  # Save to graph
  graph = EventGraph()
  graph.clear()  # Start fresh
  graph.add_edges(edges)

  # Save market outcomes for backtesting
  outcomes_file = Path("data/market_outcomes.json")
  outcomes_data = {mid: outcome.value for mid, outcome in market_outcomes.items()}
  with open(outcomes_file, "w") as f:
    json.dump(outcomes_data, f, indent=2)
  print(f"Saved {len(outcomes_data)} market outcomes to {outcomes_file}")

  print_summary(graph)


def cmd_monitor():
  """Monitor for trading signals in real-time."""
  from signals import SignalMonitor

  print("\n" + "=" * 60)
  print("LIVE SIGNAL MONITOR")
  print("=" * 60)
  print("Press Ctrl+C to stop\n")

  monitor = SignalMonitor()
  monitor.run_forever(interval=60)


def cmd_backtest():
  """Run backtest on historical data."""
  from backtest import BacktestEngine, print_results

  print("\n" + "=" * 60)
  print("RUNNING BACKTEST")
  print("=" * 60)

  engine = BacktestEngine()
  result = engine.run()
  print_results(result)


def cmd_ingest():
  """Start continuous data ingestion."""
  from ingestion import IngestionRunner

  print("\n" + "=" * 60)
  print("DATA INGESTION")
  print("=" * 60)
  print("Snapshot interval: 5 minutes")
  print("Resolution check: 1 minute")
  print("Press Ctrl+C to stop\n")

  runner = IngestionRunner(
    snapshot_interval=300,
    resolution_interval=60,
  )
  runner.run_forever()


def cmd_status():
  """Show system status."""
  import json
  from pathlib import Path
  from graph import EventGraph
  from ingestion import ResolutionTracker, SNAPSHOTS_DIR

  print("\n" + "=" * 60)
  print("SYSTEM STATUS")
  print("=" * 60)

  # Graph
  print("\n[Event Graph]")
  graph = EventGraph()
  stats = graph.stats()
  print(f"  Total edges: {stats['total_edges']}")
  print(f"  Valid edges: {stats['valid_edges']}")
  print(f"  Leaders: {stats['unique_leaders']}")
  print(f"  Followers: {stats['unique_followers']}")

  # Resolutions
  print("\n[Resolutions]")
  tracker = ResolutionTracker()
  resolutions = tracker.get_all()
  print(f"  Tracked: {len(resolutions)}")

  # Snapshots
  print("\n[Snapshots]")
  if SNAPSHOTS_DIR.exists():
    files = list(SNAPSHOTS_DIR.glob("snapshots_*.json"))
    print(f"  Files: {len(files)}")
  else:
    print("  No snapshots yet")

  # Signals
  print("\n[Signals]")
  signals_file = Path("data/signals.json")
  if signals_file.exists():
    with open(signals_file) as f:
      signals = json.load(f)
    print(f"  Total: {len(signals)}")
  else:
    print("  No signals yet")


def cmd_help():
  """Show help message."""
  print(__doc__)


def main():
  commands = {
    "build": cmd_build,
    "monitor": cmd_monitor,
    "backtest": cmd_backtest,
    "ingest": cmd_ingest,
    "status": cmd_status,
    "help": cmd_help,
    "--help": cmd_help,
    "-h": cmd_help,
  }

  if len(sys.argv) < 2:
    cmd_help()
    sys.exit(1)

  cmd = sys.argv[1].lower()

  if cmd not in commands:
    print(f"Unknown command: {cmd}")
    cmd_help()
    sys.exit(1)

  try:
    commands[cmd]()
  except KeyboardInterrupt:
    print("\nStopped.")
  except Exception as e:
    logging.error(f"Error: {e}")
    sys.exit(1)


if __name__ == "__main__":
  main()
