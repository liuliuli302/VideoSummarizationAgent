from __future__ import annotations


class MemoryManager:
    def __init__(self, enabled: bool = False, max_history_segments: int | None = None) -> None:
        self.enabled = enabled
        self.max_history_segments = max_history_segments
        self.history: list[str] = []

    def build_context(self) -> str:
        if not self.enabled or not self.history:
            return ""

        items = self.history
        if self.max_history_segments is not None:
            items = items[-self.max_history_segments :]

        return "\n".join(f"Previous segment {index}: {caption}" for index, caption in enumerate(items))

    def append(self, caption: str) -> None:
        cleaned = (caption or "").strip()
        if cleaned:
            self.history.append(cleaned)

    def reset(self) -> None:
        self.history = []