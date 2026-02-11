"""Tests for backtest module."""

from src.topic.backtest import BacktestResult, Trade


def test_trade_pnl_buy():
    """Test PnL calculation for BUY trades."""
    trade = Trade(
        source_id="source1",
        target_id="target1",
        direction="BUY",
        entry_price=0.50,
        exit_price=0.95,
        similarity=0.85,
    )

    assert abs(trade.pnl - 0.45) < 1e-10
    assert trade.won is True


def test_trade_pnl_sell():
    """Test PnL calculation for SELL trades."""
    trade = Trade(
        source_id="source1",
        target_id="target1",
        direction="SELL",
        entry_price=0.50,
        exit_price=0.05,
        similarity=0.85,
    )

    assert abs(trade.pnl - 0.45) < 1e-10
    assert trade.won is True


def test_trade_loss():
    """Test losing trade."""
    trade = Trade(
        source_id="source1",
        target_id="target1",
        direction="BUY",
        entry_price=0.50,
        exit_price=0.05,
        similarity=0.85,
    )

    assert abs(trade.pnl - (-0.45)) < 1e-10
    assert trade.won is False


def test_backtest_result_stats():
    """Test BacktestResult statistics."""
    trades = [
        Trade("s1", "t1", "BUY", 0.5, 0.95, 0.8),  # win +0.45
        Trade("s2", "t2", "BUY", 0.5, 0.05, 0.8),  # loss -0.45
        Trade("s3", "t3", "SELL", 0.5, 0.05, 0.8),  # win +0.45
    ]

    result = BacktestResult(
        trades=trades,
        total_markets=100,
        related_pairs=50,
    )

    assert result.total_trades == 3
    assert result.winning_trades == 2
    assert abs(result.win_rate - 0.6667) < 0.01
    assert abs(result.total_pnl - 0.45) < 0.01


def test_backtest_result_empty():
    """Test BacktestResult with no trades."""
    result = BacktestResult(trades=[], total_markets=10, related_pairs=0)

    assert result.total_trades == 0
    assert result.win_rate == 0.0
    assert result.total_pnl == 0.0
    assert result.avg_pnl == 0.0


def test_backtest_result_summary():
    """Test BacktestResult summary generation."""
    result = BacktestResult(trades=[], total_markets=100, related_pairs=10)

    summary = result.summary()
    assert "Markets analyzed: 100" in summary
    assert "Related pairs found: 10" in summary
