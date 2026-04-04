from __future__ import annotations

import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "silver_prices.csv"
YEARLY_URL = "https://www.exchange-rates.org/precious-metals/silver-price/pakistan/{year}"

# Rotate User-Agents to avoid datacenter IP blocking
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:126.0) Gecko/20100101 Firefox/126.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ensure_data_file() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        logger.info("Creating data file: %s", DATA_FILE)
        pd.DataFrame(columns=["Date", "Silver_PKR_per_Ounce"]).to_csv(
            DATA_FILE, index=False, encoding="utf-8-sig"
        )


def normalize_price(price_text: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9.]+", "", price_text)
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_date_text(date_text: str, year: int) -> Optional[datetime]:
    date_text = date_text.strip()
    candidates = ["%b %d %Y", "%d %b %Y"]

    for fmt in candidates:
        try:
            return datetime.strptime(f"{date_text} {year}", fmt)
        except ValueError:
            continue

    return None


def _build_session() -> requests.Session:
    """Build a requests Session with realistic browser headers.
    
    Visits the homepage first to collect cookies/tokens, which
    helps bypass bot detection on the data pages.
    """
    session = requests.Session()
    ua = random.choice(USER_AGENTS)
    session.headers.update({
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Cache-Control": "max-age=0",
    })
    
    # Warm up: visit homepage to acquire cookies/session tokens
    try:
        logger.info("Warming up session by visiting homepage...")
        session.get("https://www.exchange-rates.org/", timeout=15)
        time.sleep(random.uniform(1.5, 3.0))
        # Now set Referer for subsequent requests
        session.headers.update({
            "Referer": "https://www.exchange-rates.org/",
            "Sec-Fetch-Site": "same-origin",
        })
    except Exception:
        logger.warning("Homepage warmup failed, proceeding without cookies")
    
    return session


def _fetch_page_with_retry(url: str, max_retries: int = 3) -> Optional[str]:
    """Fetch a page with retry logic and rotating User-Agents.
    
    The real page is ~150KB+. Blocked/bot-detection responses are much shorter
    (~35KB) and don't contain the price table. We verify both size and table presence.
    """
    for attempt in range(1, max_retries + 1):
        session = _build_session()
        try:
            logger.info("Attempt %d/%d: Fetching %s", attempt, max_retries, url)
            
            # Add a small random delay to look more human-like
            if attempt > 1:
                time.sleep(random.uniform(1, 3))
            
            response = session.get(url, timeout=30)
            content_len = len(response.text)
            logger.info("Response status: %d, content length: %d bytes",
                        response.status_code, content_len)

            if response.status_code != 200:
                logger.warning("Attempt %d: Got non-200 status: %d", attempt, response.status_code)
            elif content_len < 50000:
                logger.warning("Attempt %d: Response too short (%d bytes) — likely blocked/bot-detected",
                               attempt, content_len)
            else:
                # Verify the response actually contains a price table
                if "<table" in response.text.lower():
                    logger.info("Attempt %d: Got valid response with table", attempt)
                    return response.text
                else:
                    logger.warning("Attempt %d: Response has no <table> — likely a CAPTCHA page", attempt)

        except requests.exceptions.RequestException as exc:
            logger.warning("Attempt %d: Request failed: %s", attempt, exc)

        if attempt < max_retries:
            wait = min(2 ** attempt, 30) + random.uniform(0, 2)
            logger.info("Waiting %.1f seconds before retry...", wait)
            time.sleep(wait)

    logger.error("All %d attempts failed for %s", max_retries, url)
    return None


def _parse_table(html: str, year: int) -> list[dict]:
    """Parse the silver price table from HTML. Try lxml first, fallback to html.parser."""
    records = []
    
    for parser in ["lxml", "html.parser"]:
        try:
            soup = BeautifulSoup(html, parser)
        except Exception:
            logger.warning("Parser '%s' failed, trying next...", parser)
            continue

        table = soup.find("table")
        if table is None:
            logger.warning("No table found with parser '%s'", parser)
            continue

        rows = table.find_all("tr")
        logger.info("Found table with %d rows using parser '%s'", len(rows), parser)

        for row in rows[1:]:
            columns = row.find_all("td")
            if len(columns) < 2:
                continue

            date_text = columns[0].text.strip()
            price_text = columns[1].text.strip()

            parsed_datetime = parse_date_text(date_text, year)
            price = normalize_price(price_text)

            if parsed_datetime is not None and price is not None:
                records.append({
                    "date": parsed_datetime.date(),
                    "date_iso": parsed_datetime.date().isoformat(),
                    "price": price,
                })

        if records:
            logger.info("Parsed %d price records from table", len(records))
            return records
        else:
            logger.warning("Table found but no valid records parsed with '%s'", parser)

    return records


def read_existing_data() -> pd.DataFrame:
    ensure_data_file()

    try:
        df = pd.read_csv(DATA_FILE)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame(columns=["Date", "Silver_PKR_per_Ounce"])

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    return df


def scrape_latest_price() -> bool:
    ensure_data_file()
    df = read_existing_data()
    
    year = datetime.now().year
    url = YEARLY_URL.format(year=year)
    
    # Fetch the page with retry logic
    html = _fetch_page_with_retry(url)
    if html is None:
        logger.error("Could not fetch silver price page for %d after all retries", year)
        return False

    # Parse the table
    all_records = _parse_table(html, year)
    if not all_records:
        logger.warning("No price records could be parsed from the page")
        # Log a snippet of the response to help debug
        logger.info("Response snippet (first 500 chars): %s", html[:500])
        return False

    # Find new records not in existing data
    existing_dates = set()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        existing_dates = set(df["Date"].dt.date.dropna().tolist())

    new_records = []
    for record in all_records:
        if record["date"] not in existing_dates:
            new_records.append({
                "Date": record["date_iso"],
                "Silver_PKR_per_Ounce": record["price"],
            })
            existing_dates.add(record["date"])

    if not new_records:
        logger.info("No new data found. CSV already has all %d records from the website.", len(all_records))
        return False

    new_df = pd.DataFrame(new_records)
    df = pd.concat([df, new_df], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date")

    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    logger.info("SUCCESS: Appended %d new silver price records. Total rows: %d",
                len(new_records), len(df))
    return True


if __name__ == "__main__":
    scrape_latest_price()
