#!/usr/bin/env python3
"""
Polymarket Signal System

A pipeline for ingesting Polymarket data, building similarity graphs,
and monitoring for trading signals.

Commands:
  build     - Build similarity graph from resolved markets
  backtest  - Test strategy on historical data
  ingest    - Poll for new market resolutions
  monitor   - Watch for resolutions and emit signals
  status    - Show system status

Usage:
  python -m topic.run build
  python -m topic.run backtest
  python -m topic.run monitor
"""

import logging
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def cmd_ingest():
    """Poll for new market resolutions."""
    from .ingestion import ResolutionTracker

    print("\n=== INGESTING RESOLUTIONS ===\n")

    tracker = ResolutionTracker()
    new = tracker.check_new()

    if new:
        print(f"\nFound {len(new)} new resolutions")
    else:
        print("No new resolutions")


def cmd_build():
    """Build similarity graph from resolved markets."""
    from .config import MARKETS_PER_PAGE, MAX_PAGES, MIN_EMBEDDING_SIMILARITY
    from .graph import build_embedding_graph, save_graph
    from .utils.client import fetch_closed_markets

    print("\n=== BUILDING SIMILARITY GRAPH ===\n")
    print("Creates a graph of semantically similar market pairs.\n")

    print("Fetching resolved markets...")
    markets = fetch_closed_markets(limit=MARKETS_PER_PAGE, max_pages=MAX_PAGES)
    print(f"Fetched {len(markets)} markets")

    if len(markets) < 10:
        print("Not enough markets to build graph.")
        return

    # Filter to clearly resolved markets
    clear_markets = [m for m in markets if m.yes_price > 0.9 or m.yes_price < 0.1]
    print(f"Clearly resolved: {len(clear_markets)} markets")

    print(f"\nBuilding graph (min_similarity={MIN_EMBEDDING_SIMILARITY})...")
    edges = build_embedding_graph(clear_markets)

    if not edges:
        print("No related market pairs found.")
        return

    # Save the graph
    save_graph(edges, clear_markets)

    print("\nGraph built:")
    print(f"  Total edges: {len(edges)}")
    print("  Saved to: data/event_graph.json")

    # Show sample edges
    print("\nSample edges:")
    for edge in edges[:5]:
        source = next((m for m in clear_markets if m.id == edge.from_market_id), None)
        target = next((m for m in clear_markets if m.id == edge.to_market_id), None)
        if source and target:
            print(f"  sim={edge.similarity:.2f}")
            print(f"    Source: {source.question[:55]}...")
            print(f"    Target: {target.question[:55]}...")


def cmd_monitor():
    """Watch for resolutions and emit trading signals."""
    from .graph import EventGraph
    from .ingestion import ResolutionTracker
    from .signals import SignalEngine

    print("\n=== MONITORING FOR SIGNALS ===")
    print("Press Ctrl+C to stop\n")

    graph = EventGraph()
    tracker = ResolutionTracker()
    engine = SignalEngine(graph)

    stats = graph.stats()
    print(f"Graph: {stats['total_edges']} edges, {stats['unique_sources']} sources")
    print(f"Resolutions tracked: {len(tracker.get_all())}")
    print("\nPolling for new resolutions every 60s...\n")

    try:
        while True:
            new_resolutions = tracker.check_new()

            if new_resolutions:
                engine.refresh_markets()
                for res in new_resolutions:
                    signals = engine.on_resolution(res.market_id, res.outcome)
                    for sig in signals:
                        print(f"\n[{sig.direction}] {sig.market_id[:20]}...")
                        print(f"  {sig.current_price:.1%} -> {sig.expected_price:.1%}")

            time.sleep(60)

    except KeyboardInterrupt:
        print("\nStopped.")


def cmd_status():
    """Show system status."""
    from pathlib import Path

    from .graph import EventGraph
    from .ingestion import ResolutionTracker

    print("\n=== SYSTEM STATUS ===\n")

    # Graph
    graph = EventGraph()
    stats = graph.stats()
    print("[Graph]")
    print(f"  Edges: {stats['total_edges']}")
    print(f"  Sources: {stats['unique_sources']}")

    # Resolutions
    tracker = ResolutionTracker()
    print("\n[Resolutions]")
    print(f"  Tracked: {len(tracker.get_all())}")

    # Signals
    signals_file = Path("data/signals.json")
    if signals_file.exists():
        import json

        with signals_file.open() as f:
            signals = json.load(f)
        print("\n[Signals]")
        print(f"  Total: {len(signals)}")
    else:
        print("\n[Signals]")
        print("  None yet")


def cmd_backtest():
    """Run backtest on historical data."""
    from .backtest import run_backtest
    from .config import BACKTEST_MAX_MARKETS, BACKTEST_VERBOSE, MIN_EMBEDDING_SIMILARITY

    # Check for command line options
    min_sim = MIN_EMBEDDING_SIMILARITY
    for arg in sys.argv:
        if arg.startswith("--min-sim="):
            min_sim = float(arg.split("=")[1])

    print("\n=== BACKTESTING ===\n")
    print("Testing: do semantically similar markets resolve the same way?")
    print(f"Min similarity: {min_sim}\n")

    result = run_backtest(
        min_similarity=min_sim,
        max_markets=BACKTEST_MAX_MARKETS,
        verbose=BACKTEST_VERBOSE,
    )

    print("\n" + result.summary())
    print("\nInterpretation:")
    print("  Win rate > 50% = similar markets tend to agree (strategy has edge)")
    print("  Win rate ~ 50% = no predictive value (random)")
    print("  Win rate < 50% = similar markets tend to disagree (inverse strategy)")


def cmd_help():
    print(__doc__)


def main():
    commands = {
        "ingest": cmd_ingest,
        "build": cmd_build,
        "monitor": cmd_monitor,
        "backtest": cmd_backtest,
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
        logger.error(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()
