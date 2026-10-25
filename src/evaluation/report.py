from pathlib import Path
from typing import Dict, Iterable

from openpyxl import Workbook
from openpyxl.utils import get_column_letter


def _state_value(state, field: str):
    return getattr(state, field, None) if state is not None else None


def write_excel_report(run: Dict, output_file: str) -> str:
    """Write measured pipeline results and explicit unavailable fields to Excel."""
    workbook = Workbook()
    summary = workbook.active
    summary.title = "Summary"
    summary.append(["Field", "Value"])
    summary_rows = [
        ("Input source", run["source"]),
        ("Processed images", len(run["results"])),
        ("Skipped images", len(run["skipped"])),
        ("mAP", "Unavailable: no ground-truth annotations"),
        ("Precision", "Unavailable: no ground-truth annotations"),
        ("Recall", "Unavailable: no ground-truth annotations"),
    ]
    for row in summary_rows:
        summary.append(row)

    images = workbook.create_sheet("Per Image")
    images.append(
        [
            "Frame ID",
            "Filename",
            "Width",
            "Height",
            "Strategy",
            "Strategy Explanation",
            "Haze Score",
            "Visibility Score",
            "Illumination Score",
            "Contrast Score",
            "CPU %",
            "RAM %",
            "GPU Available",
            "Restoration Latency (ms)",
            "Detection Latency (ms)",
            "Total Latency (ms)",
            "FPS",
            "Detection Count",
            "mAP",
            "Precision",
            "Recall",
        ]
    )
    detections = workbook.create_sheet("Detections")
    detections.append(["Frame ID", "Filename", "Class", "Confidence", "X1", "Y1", "X2", "Y2"])

    for result in run["results"]:
        environment = result["environment"]
        hardware = result["hardware"]
        decision = result["decision"]
        detection = result["detection"]
        evaluation = result["evaluation"]
        images.append(
            [
                result["frame_id"],
                result["filename"],
                result["width"],
                result["height"],
                decision.selected_strategy,
                decision.explanation,
                _state_value(environment, "haze_score"),
                _state_value(environment, "visibility_score"),
                _state_value(environment, "illumination_score"),
                _state_value(environment, "contrast_score"),
                _state_value(hardware, "cpu_utilization"),
                _state_value(hardware, "ram_utilization"),
                _state_value(hardware, "gpu_available"),
                result["restoration_latency_ms"],
                detection["latency_ms"],
                evaluation["latency_ms"],
                evaluation["fps"],
                len(detection["detections"]),
                evaluation["mAP"],
                evaluation["precision"],
                evaluation["recall"],
            ]
        )
        for item in detection["detections"]:
            box = item["box"]
            detections.append(
                [
                    result["frame_id"],
                    result["filename"],
                    item["class_name"],
                    item["confidence"],
                    *box,
                ]
            )

    skipped = workbook.create_sheet("Skipped")
    skipped.append(["Filename", "Path", "Reason"])
    for item in run["skipped"]:
        skipped.append([item["filename"], item["path"], item["error"]])

    strategy_sheet = workbook.create_sheet("Strategy Summary")
    strategy_sheet.append(["Strategy", "Images", "Average Latency (ms)", "Average FPS"])
    strategies = {}
    for result in run["results"]:
        strategy = result["decision"].selected_strategy
        strategies.setdefault(strategy, []).append(result["evaluation"])
    for strategy, evaluations in sorted(strategies.items()):
        strategy_sheet.append(
            [
                strategy,
                len(evaluations),
                sum(item["latency_ms"] for item in evaluations) / len(evaluations),
                sum(item["fps"] for item in evaluations) / len(evaluations),
            ]
        )

    for sheet in workbook.worksheets:
        sheet.freeze_panes = "A2"
        for column_cells in sheet.columns:
            width = min(
                60,
                max(len(str(cell.value or "")) for cell in column_cells) + 2,
            )
            sheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width

    destination = Path(output_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(destination)
    return str(destination)
