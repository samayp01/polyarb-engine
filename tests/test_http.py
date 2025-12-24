"""Tests for HTTP utilities."""

import pytest
import responses

from src.topic.utils.http import make_session, safe_get_json


def test_make_session():
  """Test session creation with retry logic."""
  session = make_session()

  assert session is not None
  assert session.adapters["http://"] is not None
  assert session.adapters["https://"] is not None


@responses.activate
def test_safe_get_json_success():
  """Test successful JSON fetch."""
  responses.add(
    responses.GET,
    "https://api.example.com/data",
    json={"key": "value"},
    status=200,
  )

  session = make_session()
  result = safe_get_json(session, "https://api.example.com/data")

  assert result == {"key": "value"}


@responses.activate
def test_safe_get_json_with_params():
  """Test JSON fetch with query parameters."""
  responses.add(
    responses.GET,
    "https://api.example.com/data",
    json={"result": "ok"},
    status=200,
  )

  session = make_session()
  result = safe_get_json(
    session,
    "https://api.example.com/data",
    params={"limit": 10, "offset": 0},
  )

  assert result == {"result": "ok"}


@responses.activate
def test_safe_get_json_http_error():
  """Test handling of HTTP errors."""
  responses.add(
    responses.GET,
    "https://api.example.com/error",
    status=500,
  )

  session = make_session()
  result = safe_get_json(session, "https://api.example.com/error")

  assert result is None


@responses.activate
def test_safe_get_json_404():
  """Test handling of 404 errors."""
  responses.add(
    responses.GET,
    "https://api.example.com/notfound",
    status=404,
  )

  session = make_session()
  result = safe_get_json(session, "https://api.example.com/notfound")

  assert result is None


@responses.activate
def test_safe_get_json_invalid_json():
  """Test handling of invalid JSON."""
  responses.add(
    responses.GET,
    "https://api.example.com/badjson",
    body="not json",
    status=200,
  )

  session = make_session()
  result = safe_get_json(session, "https://api.example.com/badjson")

  assert result is None


@responses.activate
def test_safe_get_json_with_headers():
  """Test JSON fetch with custom headers."""
  responses.add(
    responses.GET,
    "https://api.example.com/auth",
    json={"authenticated": True},
    status=200,
  )

  session = make_session()
  result = safe_get_json(
    session,
    "https://api.example.com/auth",
    headers={"Authorization": "Bearer token123"},
  )

  assert result == {"authenticated": True}


@responses.activate
def test_safe_get_json_timeout():
  """Test timeout parameter."""
  responses.add(
    responses.GET,
    "https://api.example.com/slow",
    json={"data": "slow"},
    status=200,
  )

  session = make_session()
  result = safe_get_json(
    session,
    "https://api.example.com/slow",
    timeout=60,
  )

  assert result == {"data": "slow"}
