from pathlib import Path
from typing import Dict, Iterator, Union

import cv2


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class InputModule:
    """Discovers and loads supported images without modifying source files."""

    def process(self, source: Union[str, Path]) -> Iterator[Dict]:
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Input source does not exist: {source_path}")

        if source_path.is_file():
            candidates = [source_path]
        else:
            candidates = sorted(
                path
                for path in source_path.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        if not candidates:
            raise ValueError(f"No supported images found in input source: {source_path}")

        for path in candidates:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None or image.size == 0:
                yield {
                    "path": str(path),
                    "filename": path.name,
                    "image": None,
                    "error": "OpenCV could not decode the image",
                }
                continue

            height, width = image.shape[:2]
            yield {
                "path": str(path),
                "filename": path.name,
                "image": image,
                "width": width,
                "height": height,
                "error": None,
            }
