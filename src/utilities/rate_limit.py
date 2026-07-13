import csv
import json
import math
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

LOCK_STALE_SECONDS = 120
TOKEN_ESTIMATE_CHARACTERS = 4


class DailyQuotaExceededError(RuntimeError):
    """Raised when a configured local daily request budget has been exhausted."""


class RequestPacer:
    """Coordinate project-level model quotas safely across threads and processes."""

    def __init__(self, window_seconds: int):
        """Initialize persistent quota paths and the rolling window length."""
        self.window_seconds = window_seconds
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.usage_path = os.path.join(project_root, "cache", "_model_usage.json")
        self.recent_path = os.path.join(project_root, "cache", "_model_recent_requests.json")
        self.lock_path = os.path.join(project_root, "cache", "_model_rate_limit.lock")
        self.request_log_path = os.path.join(project_root, "request_logs", "model_requests.csv")
        self.input_tpm = self.get_configured_input_tpm()

    def get_configured_input_tpm(self) -> int:
        """Return the optional project input-token-per-minute limit."""
        configured_tpm = os.getenv("MODEL_INPUT_TPM", "0").strip()
        try:
            return max(int(configured_tpm), 0)
        except ValueError:
            return 0

    def get_today_key(self) -> str:
        """Return the provider quota date using midnight Pacific time."""
        try:
            pacific_timezone = ZoneInfo("America/Los_Angeles")
        except ZoneInfoNotFoundError:
            pacific_timezone = timezone(timedelta(hours=-8))
        return datetime.now(pacific_timezone).date().isoformat()

    def load_json_object(self, filepath: str) -> dict:
        """Load a JSON object and treat missing or partial state as empty."""
        if not os.path.exists(filepath):
            return {}
        try:
            with open(filepath, encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def save_json_object(self, filepath: str, payload: dict):
        """Persist quota state atomically so interrupted writes remain recoverable."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        temp_path = f"{filepath}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        os.replace(temp_path, filepath)

    def remove_stale_lock(self) -> bool:
        """Remove a lock left behind by a terminated process."""
        try:
            lock_age = time.time() - os.path.getmtime(self.lock_path)
        except FileNotFoundError:
            return True
        if lock_age <= LOCK_STALE_SECONDS:
            return False
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass
        return True

    def acquire_lock(self):
        """Acquire the quota-state lock with recovery for abandoned lock files."""
        os.makedirs(os.path.dirname(self.lock_path), exist_ok=True)
        while True:
            try:
                file_descriptor = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                lock_payload = f"pid={os.getpid()} acquired={time.time()}".encode("utf-8")
                os.write(file_descriptor, lock_payload)
                os.close(file_descriptor)
                return
            except FileExistsError:
                self.remove_stale_lock()
                time.sleep(0.05)

    def release_lock(self):
        """Release the quota-state lock after a state transaction."""
        try:
            os.remove(self.lock_path)
        except FileNotFoundError:
            pass

    def normalize_recent_entries(self, payload: dict) -> dict[str, list[dict[str, float | int]]]:
        """Normalize current and legacy recent-request records."""
        recent: dict[str, list[dict[str, float | int]]] = {}
        for bucket, entries in payload.items():
            if not isinstance(bucket, str) or not isinstance(entries, list):
                continue
            normalized_entries: list[dict[str, float | int]] = []
            for entry in entries:
                if isinstance(entry, (int, float)):
                    normalized_entries.append({"timestamp": float(entry), "input_tokens": 0})
                    continue
                if not isinstance(entry, dict):
                    continue
                try:
                    timestamp = float(entry.get("timestamp", 0.0))
                    input_tokens = max(int(entry.get("input_tokens", 0)), 0)
                except (TypeError, ValueError):
                    continue
                normalized_entries.append({"timestamp": timestamp, "input_tokens": input_tokens})
            recent[bucket] = normalized_entries
        return recent

    def estimate_input_tokens(self, prompt: str | None) -> int:
        """Estimate input tokens conservatively when provider counts are unavailable."""
        if not prompt:
            return 0
        return math.ceil(len(prompt) / TOKEN_ESTIMATE_CHARACTERS)

    def append_request_log(self, bucket: str, request_name: str, requested_at: datetime):
        """Append one human-readable request reservation to the CSV log."""
        os.makedirs(os.path.dirname(self.request_log_path), exist_ok=True)
        write_header = not os.path.exists(self.request_log_path) or os.path.getsize(self.request_log_path) == 0
        with open(self.request_log_path, "a", encoding="utf-8", newline="") as file_handle:
            fieldnames = ["model", "request_made", "requested_at"]
            writer = csv.DictWriter(file_handle, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow({"model": bucket, "request_made": request_name, "requested_at": requested_at.isoformat(timespec="seconds")})

    def acquire_request_slot(self, bucket: str, rpm: int, rpd: int, request_name: str | None = None, prompt: str | None = None):
        """Reserve a project-level request slot under RPM, optional TPM, and RPD limits."""
        if rpm < 1 or rpd < 1:
            raise DailyQuotaExceededError(f"Configured request budget is zero for {bucket}: rpm={rpm}, rpd={rpd}. Switch models or override its limits.")

        input_tokens = self.estimate_input_tokens(prompt)
        if self.input_tpm > 0 and input_tokens > self.input_tpm:
            raise DailyQuotaExceededError(f"Estimated prompt size exceeds MODEL_INPUT_TPM for {bucket}: {input_tokens}/{self.input_tpm}.")
        while True:
            wait_seconds = 0.0
            self.acquire_lock()
            try:
                now = time.time()
                today_key = self.get_today_key()
                usage = self.load_json_object(self.usage_path)
                day_usage = usage.get(today_key, {})
                current_daily_count = int(day_usage.get(bucket, 0))
                if current_daily_count >= rpd:
                    raise DailyQuotaExceededError(f"Local daily request budget reached for {bucket}: {current_daily_count}/{rpd}. Resume after the Pacific-time quota reset or switch models.")

                recent_payload = self.load_json_object(self.recent_path)
                recent = self.normalize_recent_entries(recent_payload)
                cutoff = now - self.window_seconds
                entries = [entry for entry in recent.get(bucket, []) if float(entry["timestamp"]) > cutoff]
                recent[bucket] = entries
                used_input_tokens = sum(int(entry["input_tokens"]) for entry in entries)
                rpm_available = len(entries) < rpm
                tpm_available = self.input_tpm < 1 or used_input_tokens + input_tokens <= self.input_tpm

                if rpm_available and tpm_available:
                    entries.append({"timestamp": now, "input_tokens": input_tokens})
                    usage = {today_key: day_usage}
                    day_usage[bucket] = current_daily_count + 1
                    self.save_json_object(self.recent_path, recent)
                    self.save_json_object(self.usage_path, usage)
                    self.append_request_log(bucket, request_name or "unspecified", datetime.now())
                    return

                oldest_timestamp = min(float(entry["timestamp"]) for entry in entries)
                wait_seconds = max(oldest_timestamp + self.window_seconds - now, 0.05)
                self.save_json_object(self.recent_path, recent)
            finally:
                self.release_lock()

            print(f"Rate limit reached for {bucket} - waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    def mark_daily_exhausted(self, bucket: str, rpd: int):
        """Persist a provider-reported daily exhaustion under the state lock."""
        if rpd < 1:
            return
        self.acquire_lock()
        try:
            today_key = self.get_today_key()
            usage = self.load_json_object(self.usage_path)
            day_usage = usage.get(today_key, {})
            day_usage[bucket] = max(int(day_usage.get(bucket, 0)), rpd)
            self.save_json_object(self.usage_path, {today_key: day_usage})
        finally:
            self.release_lock()

    def is_daily_quota_error(self, error_body) -> bool:
        """Detect provider responses that indicate daily request exhaustion."""
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
                    if isinstance(violation, dict) and "PerDay" in str(violation.get("quotaId", "")):
                        return True
        message = str(error_payload.get("message", ""))
        return "perday" in message.lower() or "per day" in message.lower()
