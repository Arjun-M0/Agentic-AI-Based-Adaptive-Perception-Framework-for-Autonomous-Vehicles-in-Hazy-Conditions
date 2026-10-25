"""
python main.py data/mid_review/images --report outputs/reports/mid_review_batch_evaluation.xlsx --detections outputs/detections/mid_review_batch
"""
import argparse
from pathlib import Path

from src.core.logger import setup_logger
from src.core.pipeline import PerceptionPipeline
from src.evaluation.report import write_excel_report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run adaptive perception evaluation.")
    parser.add_argument(
        "source",
        nargs="?",
        default="mid review/sample_dataset_30",
        help="Image file or directory to process.",
    )
    parser.add_argument(
        "--report",
        default="outputs/reports/batch_evaluation.xlsx",
        help="Excel report output path.",
    )
    parser.add_argument(
        "--detections",
        default="outputs/detections",
        help="Directory for annotated detection images.",
    )
    args = parser.parse_args(argv)

    logger = setup_logger()
    logger.info('Initializing Agentic AI-Based Adaptive Perception Framework...')
    pipeline = PerceptionPipeline()
    run = pipeline.run(args.source, args.detections)
    report_path = write_excel_report(run, args.report)
    logger.info(
        "Processed %d images, skipped %d images. Report: %s",
        len(run["results"]),
        len(run["skipped"]),
        Path(report_path).resolve(),
    )
    return run

if __name__ == '__main__':
    main()
