import csv
import json
import os
import time
from datetime import datetime


class DailyQuotaExceededError(RuntimeError):
    """Raised when a configured local daily request budget has been exhausted."""


class RequestPacer:
    """Track request budgets per model so shared quotas are not overrun."""

    def __init__(self, window_seconds: int):
        """Initialize the pacer with a rolling window size for per-minute budgets."""
        self.window_seconds = window_seconds
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.usage_path = os.path.join(project_root, "cache", "_model_usage.json")
        self.recent_path = os.path.join(project_root, "cache", "_model_recent_requests.json")
        self.lock_path = os.path.join(project_root, "cache", "_model_rate_limit.lock")
        self.request_log_path = os.path.join(project_root, "request_logs", "model_requests.csv")
        self.usage_cache: dict[str, dict[str, int]] | None = None
        self.recent_cache: dict[str, list[float]] | None = None

    def get_today_key(self) -> str:
        """Return the local calendar day used for daily request accounting."""
        return datetime.now().date().isoformat()

    def load_usage(self) -> dict[str, dict[str, int]]:
        """Load persisted per-day model usage from disk."""
        if self.usage_cache is not None:
            return self.usage_cache

        if not os.path.exists(self.usage_path):
            self.usage_cache = {}
            return self.usage_cache

        try:
            with open(self.usage_path, encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        self.usage_cache = payload
        return self.usage_cache

    def save_usage(self):
        """Persist the current per-day model usage cache to disk."""
        if self.usage_cache is None:
            return

        os.makedirs(os.path.dirname(self.usage_path), exist_ok=True)
        with open(self.usage_path, "w", encoding="utf-8") as file_handle:
            json.dump(self.usage_cache, file_handle, indent=2, ensure_ascii=False)

    def load_recent(self) -> dict[str, list[float]]:
        """Load persisted recent per-model request timestamps."""
        if self.recent_cache is not None:
            return self.recent_cache

        if not os.path.exists(self.recent_path):
            self.recent_cache = {}
            return self.recent_cache

        try:
            with open(self.recent_path, encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            payload = {}

        if not isinstance(payload, dict):
            payload = {}

        recent: dict[str, list[float]] = {}
        for bucket, timestamps in payload.items():
            if not isinstance(bucket, str) or not isinstance(timestamps, list):
                continue
            cleaned: list[float] = []
            for timestamp in timestamps:
                try:
                    cleaned.append(float(timestamp))
                except (TypeError, ValueError):
                    continue
            recent[bucket] = cleaned

        self.recent_cache = recent
        return self.recent_cache

    def save_recent(self):
        """Persist recent per-model request timestamps."""
        if self.recent_cache is None:
            return

        os.makedirs(os.path.dirname(self.recent_path), exist_ok=True)
        with open(self.recent_path, "w", encoding="utf-8") as file_handle:
            json.dump(self.recent_cache, file_handle, indent=2, ensure_ascii=False)

    def acquire_lock(self):
        """Acquire a small cross-process lock for rate-limit state updates."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        while True:
            try:
                file_descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(file_descriptor)
                return
            except FileExistsError:
                time.sleep(0.05)

    def release_lock(self):
        """Release the cross-process rate-limit lock."""
        if os.path.exists(self.lock_path):
            os.remove(self.lock_path)

    def get_daily_count(self, bucket: str) -> int:
        """Return the persisted request count for the current day and model."""
        usage = self.load_usage()
        return int(usage.get(self.get_today_key(), {}).get(bucket, 0))

    def ensure_daily_capacity(self, bucket: str, rpd: int):
        """Raise before sending a request if the configured daily budget is exhausted."""
        current_count = self.get_daily_count(bucket)
        if current_count >= rpd:
            raise DailyQuotaExceededError(
                f"Local daily request budget reached for {bucket}: {current_count}/{rpd}. "
                "Resume after the quota window resets or switch this stage to a different model."
            )

    def prune_recent(self, bucket: str, now: float):
        """Keep only timestamps that still belong to the rolling minute window."""
        recent = self.load_recent()
        timestamps = recent.get(bucket, [])
        cutoff = now - self.window_seconds
        recent[bucket] = [timestamp for timestamp in timestamps if timestamp > cutoff]

    def increment_daily_usage(self, bucket: str):
        """Record one dispatched request in the current day."""
        usage = self.load_usage()
        today_key = self.get_today_key()
        day_usage = usage.setdefault(today_key, {})
        day_usage[bucket] = int(day_usage.get(bucket, 0)) + 1
        self.save_usage()

    def append_request_log(self, bucket: str, request_name: str, requested_at: datetime):
        """Append one human-readable request reservation row to the CSV log."""
        os.makedirs(os.path.dirname(self.request_log_path), exist_ok=True)
        write_header = not os.path.exists(self.request_log_path) or os.path.getsize(self.request_log_path) == 0
        with open(self.request_log_path, "a", encoding="utf-8", newline="") as file_handle:
            writer = csv.DictWriter(file_handle, fieldnames=["model", "request_made", "requested_at"])
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "model": bucket,
                    "request_made": request_name,
                    "requested_at": requested_at.isoformat(timespec="seconds"),
                }
            )

    def acquire_request_slot(self, bucket: str, rpm: int, rpd: int, request_name: str | None = None):
        """Reserve one request slot before dispatch using a rolling 60-second window."""
        if rpm < 1 or rpd < 1:
            raise DailyQuotaExceededError(
                f"Configured request budget is zero for {bucket}: rpm={rpm}, rpd={rpd}. "
                "Switch this stage to an available model or override its limits."
            )

        while True:
            wait_seconds = 0.0
            self.acquire_lock()
            try:
                now = time.time()
                self.ensure_daily_capacity(bucket, rpd)
                self.prune_recent(bucket, now)

                recent = self.load_recent()
                timestamps = recent.setdefault(bucket, [])
                if len(timestamps) < rpm:
                    timestamps.append(now)
                    self.save_recent()
                    self.increment_daily_usage(bucket)
                    self.append_request_log(bucket, request_name or "unspecified", datetime.now())
                    return

                oldest_timestamp = min(timestamps)
                wait_seconds = max((oldest_timestamp + self.window_seconds) - now, 0.05)
            finally:
                self.release_lock()

            print(f"Rate limit reached for {bucket} - waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    def mark_daily_exhausted(self, bucket: str, rpd: int):
        """Persist that the model should be treated as exhausted for the rest of the day."""
        if rpd < 1:
            return

        usage = self.load_usage()
        today_key = self.get_today_key()
        day_usage = usage.setdefault(today_key, {})
        day_usage[bucket] = max(int(day_usage.get(bucket, 0)), rpd)
        self.save_usage()

    def is_daily_quota_error(self, error_body) -> bool:
        """Detect provider responses that indicate a daily request-budget exhaustion."""
        if not isinstance(error_body, dict):
            return False

        error_payload = error_body.get("error")
        if not isinstance(error_payload, dict):
            return False

        details = error_payload.get("details")
        if isinstance(details, list):
            for detail in details:
                if not isinstance(detail, dict):
                    continue
                violations = detail.get("violations")
                if not isinstance(violations, list):
                    continue
                for violation in violations:
                    if not isinstance(violation, dict):
                        continue
                    quota_id = str(violation.get("quotaId", ""))
                    if "PerDay" in quota_id:
                        return True

        message = str(error_payload.get("message", ""))
        return "perday" in message.lower() or "per day" in message.lower()
