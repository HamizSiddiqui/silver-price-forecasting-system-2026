from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "silver_prices.csv"
YEARLY_URL = "https://www.exchange-rates.org/precious-metals/silver-price/pakistan/{year}"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

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


def fetch_latest_price_for_year(year: int) -> Optional[tuple[datetime, float]]:
    url = YEARLY_URL.format(year=year)
    logger.info("Fetching silver price page for %s", year)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return None

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
    if table is None:
        logger.warning("No price table found on page: %s", url)
        return None

    latest_record: Optional[tuple[datetime, float]] = None

    for row in table.find_all("tr")[1:]:
        columns = row.find_all("td")
        if len(columns) < 2:
            continue

        date_text = columns[0].text.strip()
        price_text = columns[1].text.strip()

        parsed_date = parse_date_text(date_text, year)
        price = normalize_price(price_text)

        if parsed_date is None or price is None:
            continue

        if latest_record is None or parsed_date > latest_record[0]:
            latest_record = (parsed_date, price)

    return latest_record


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
    logger.info("Fetching silver price page for %s to find all missing data", year)

    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        logger.error("Failed to fetch %s: %s", url, exc)
        return False

    soup = BeautifulSoup(response.text, "lxml")
    table = soup.find("table")
    if table is None:
        logger.warning("No price table found on page: %s", url)
        return False

    existing_dates = set()
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        existing_dates = set(df["Date"].dt.date.dropna().values)

    new_records = []
    
    for row in table.find_all("tr")[1:]:
        columns = row.find_all("td")
        if len(columns) < 2:
            continue

        date_text = columns[0].text.strip()
        price_text = columns[1].text.strip()

        parsed_datetime = parse_date_text(date_text, year)
        price = normalize_price(price_text)

        if parsed_datetime is None or price is None:
            continue
        
        parsed_date = parsed_datetime.date()
        if parsed_date not in existing_dates:
            new_records.append({"Date": parsed_date.isoformat(), "Silver_PKR_per_Ounce": price})
            existing_dates.add(parsed_date)

    if not new_records:
        logger.info("No new data found on the website.")
        return False

    new_df = pd.DataFrame(new_records)
    df = pd.concat([df, new_df], ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.sort_values("Date")

    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    logger.info("Appended %d new silver price records.", len(new_records))
    return True


if __name__ == "__main__":
    scrape_latest_price()
