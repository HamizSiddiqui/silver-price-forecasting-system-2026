from __future__ import annotations

import argparse
import logging
import time

from scraper import scrape_latest_price
from train import train_model
from visualize import main as visualize_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(skip_training_if_no_new_data: bool = True) -> None:
    logger.info("Starting silver price pipeline.")

    try:
        new_data_added = scrape_latest_price()
    except Exception as exc:  # pragma: no cover
        logger.exception("Scraper failed: %s", exc)
        return

    if skip_training_if_no_new_data and not new_data_added:
        logger.info("No new data was added. Skipping model retraining and visualization.")
        return

    try:
        train_model()
    except Exception as exc:  # pragma: no cover
        logger.exception("Training failed: %s", exc)
        return

    try:
        visualize_main()
    except Exception as exc:  # pragma: no cover
        logger.exception("Visualization failed: %s", exc)
        return

    logger.info("Pipeline completed successfully.")


def run_scheduler(interval_hours: float = 24.0, skip_training_if_no_new_data: bool = True) -> None:
    logger.info("Scheduler started. Running every %.1f hours.", interval_hours)

    while True:
        run_pipeline(skip_training_if_no_new_data=skip_training_if_no_new_data)
        logger.info("Sleeping for %.1f hours before the next run.", interval_hours)
        try:
            time.sleep(interval_hours * 3600)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Silver price prediction pipeline: scrape daily data and retrain Prophet model."
    )

    parser.add_argument(
        "--once",
        action="store_true",
        help="Run the pipeline one time and exit.",
    )
    parser.add_argument(
        "--interval-hours",
        type=float,
        default=24.0,
        help="Interval in hours between pipeline runs when scheduler mode is enabled.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain the model even when no new data is appended.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.once:
        run_pipeline(skip_training_if_no_new_data=not args.force_train)
    else:
        run_scheduler(interval_hours=args.interval_hours, skip_training_if_no_new_data=not args.force_train)
