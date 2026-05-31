import time


class RequestPacer:
    """Track request budgets per model so shared quotas are not overrun."""

    def __init__(self, window_seconds: int):
        self.window_seconds = window_seconds
        self.windows: dict[str, dict[str, float | int]] = {}

    def wait_for_capacity(self, bucket: str, rpm: int):
        """Pause when the current request bucket has exhausted its minute window."""
        now = time.time()
        state = self.windows.get(bucket)
        if state is None:
            self.windows[bucket] = {"count": 0, "start": now, "rpm": rpm}
            return

        state["rpm"] = min(int(state["rpm"]), rpm)
        elapsed = now - float(state["start"])
        if elapsed >= self.window_seconds:
            state["count"] = 0
            state["start"] = now
            return

        if int(state["count"]) >= int(state["rpm"]):
            remaining = self.window_seconds - elapsed
            print(f"Rate limit reached for {bucket} - waiting {remaining:.1f}s...")
            time.sleep(remaining)
            self.windows[bucket] = {"count": 0, "start": time.time(), "rpm": int(state["rpm"])}

    def mark_request(self, bucket: str):
        """Record one successful request in the current time window."""
        state = self.windows.setdefault(bucket, {"count": 0, "start": time.time(), "rpm": 1})
        state["count"] = int(state["count"]) + 1

    def reset(self, bucket: str, rpm: int):
        """Reset the current bucket after a long provider backoff or transient failure."""
        self.windows[bucket] = {"count": 0, "start": time.time(), "rpm": rpm}
