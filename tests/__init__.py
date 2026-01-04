"""
Unit Tests for Trading Bot

This package contains comprehensive unit tests for all core trading bot functionality.

Test Organization:
- test_scoring.py: Stock scoring logic tests
- test_kelly_weighting.py: Position sizing tests
- test_drawdown_control.py: Risk management tests
- test_vix_regime.py: VIX regime detection tests
- test_indicators.py: Technical indicator tests
- test_execution.py: Execution logic tests
- test_database.py: Database operation tests

Run all tests: pytest
Run with coverage: pytest --cov=src --cov-report=html
Run specific module: pytest tests/test_scoring.py
"""
