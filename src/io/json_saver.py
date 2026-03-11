from __future__ import annotations

import json
import os
from typing import Any


class JsonSaver:
    def save(self, path: str, payload: Any) -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)