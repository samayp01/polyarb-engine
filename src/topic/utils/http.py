"""HTTP utilities with retry logic and proper error handling."""

import logging
from typing import Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


def make_session(
  max_retries=3,
  backoff_factor=0.5,
  status_forcelist=(429, 500, 502, 503, 504),
  timeout=30,
) -> requests.Session:
  """
  Create a requests Session with retry logic.

  Args:
    max_retries: Maximum number of retries
    backoff_factor: Backoff factor for exponential retry
    status_forcelist: HTTP status codes to retry on
    timeout: Default timeout in seconds

  Returns:
    Configured requests.Session
  """
  session = requests.Session()

  retry_strategy = Retry(
    total=max_retries,
    backoff_factor=backoff_factor,
    status_forcelist=list(status_forcelist),
    allowed_methods=["GET", "POST", "PUT", "DELETE"],
  )

  adapter = HTTPAdapter(max_retries=retry_strategy)
  session.mount("http://", adapter)
  session.mount("https://", adapter)

  return session


def safe_get_json(session: requests.Session, url: str, params=None, headers=None, timeout=30):
  """
  Safely GET JSON from URL with error handling.

  Args:
    session: requests.Session to use
    url: URL to fetch
    params: Query parameters
    headers: HTTP headers
    timeout: Request timeout

  Returns:
    Parsed JSON dict or None on error
  """
  try:
    response = session.get(url, params=params, headers=headers, timeout=timeout)
    response.raise_for_status()
    return response.json()
  except requests.exceptions.HTTPError as e:
    logger.error(f"HTTP error fetching {url}: {e}")
    return None
  except requests.exceptions.RequestException as e:
    logger.error(f"Request error fetching {url}: {e}")
    return None
  except ValueError as e:
    logger.error(f"JSON parse error for {url}: {e}")
    return None
