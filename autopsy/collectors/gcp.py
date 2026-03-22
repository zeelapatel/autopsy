"""GCP Cloud Logging collector.

Pulls error-level logs from Google Cloud Logging API using Application Default
Credentials (ADC). Implements the same 4-stage reduction pipeline as the
CloudWatch and Datadog collectors.

Stage 1 (query-level filter): Handled by `severity >= ERROR` in the GCP filter.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from google.api_core import exceptions as gcp_exc
from google.cloud import logging as gcp_logging
from google.cloud.logging_v2 import DESCENDING
from rich.console import Console

from autopsy.collectors.base import BaseCollector, CollectedData
from autopsy.utils.errors import (
    CollectorError,
    GCPAuthError,
    GCPPermissionError,
    NoDataError,
)
from autopsy.utils.log_reduction import apply_token_budget, deduplicate_logs, truncate_entries

console = Console(stderr=True)

MAX_ENTRIES = 200
REQUEST_TIMEOUT = 30

COMMON_RESOURCE_TYPES: dict[str, str] = {
    "cloud_function": "Cloud Functions",
    "cloud_run_revision": "Cloud Run",
    "gce_instance": "Compute Engine VM",
    "gke_container": "GKE Container",
    "gae_app": "App Engine",
    "cloudsql_database": "Cloud SQL",
    "k8s_container": "Kubernetes Container",
    "k8s_pod": "Kubernetes Pod",
}


class GCPCollector(BaseCollector):
    """Collects error-level logs from GCP Cloud Logging API."""

    @property
    def name(self) -> str:
        """Collector identifier."""
        return "gcp"

    def validate_config(self, config: dict) -> bool:
        """Verify GCP credentials and Cloud Logging access.

        Uses list_entries(max_results=1) as a lightweight connectivity probe.

        Args:
            config: The 'gcp' section of AutopsyConfig.

        Returns:
            True if credentials and permissions are valid.

        Raises:
            GCPAuthError: On missing or invalid credentials.
            GCPPermissionError: On insufficient IAM permissions.
            CollectorError: On project not found or API errors.
        """
        project_id = config.get("project_id", "")
        credentials_env = config.get("credentials_env", "GOOGLE_APPLICATION_CREDENTIALS")

        # Check credentials availability before making any API call
        creds_path = os.environ.get(credentials_env, "")
        if not creds_path and not self._gcp_default_creds_available():
            raise GCPAuthError(
                message="GCP credentials not configured.",
                hint=(
                    "Option 1: Set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json\n"
                    "Option 2: Run 'gcloud auth application-default login'\n"
                    "Option 3: Running on GCP? Credentials are automatic via metadata server."
                ),
            )

        try:
            client = gcp_logging.Client(project=project_id)
            list(client.list_entries(max_results=1))
            return True
        except gcp_exc.Unauthenticated as exc:
            raise GCPAuthError(
                message="GCP authentication failed.",
                hint=(
                    "Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON path,\n"
                    "or run 'gcloud auth application-default login' for local development.\n"
                    "Docs: https://cloud.google.com/docs/authentication/application-default-credentials"
                ),
            ) from exc
        except gcp_exc.PermissionDenied as exc:
            raise GCPPermissionError(
                message="Insufficient GCP permissions for Cloud Logging.",
                hint=(
                    "Your service account needs the 'Logs Viewer' role (roles/logging.viewer).\n"
                    "Grant it: gcloud projects add-iam-policy-binding PROJECT_ID \\\n"
                    "  --member='serviceAccount:SA_EMAIL' --role='roles/logging.viewer'\n"
                    "Docs: https://cloud.google.com/logging/docs/access-control"
                ),
            ) from exc
        except gcp_exc.NotFound as exc:
            raise CollectorError(
                message=f"GCP project not found: {project_id}",
                hint=(
                    "Check project_id in config. Find your project ID at:\n"
                    "https://console.cloud.google.com/home/dashboard"
                ),
            ) from exc
        except gcp_exc.GoogleAPICallError as exc:
            raise CollectorError(
                message=f"GCP Cloud Logging API error: {exc}",
                hint="Check your network connectivity and GCP project configuration.",
            ) from exc

    def collect(self, config: dict) -> CollectedData:
        """Pull error-level logs from GCP Cloud Logging and apply reduction pipeline.

        Pipeline stages:
        1. Query-level filter (severity >= ERROR) — handled in GCP filter string.
        2. Deduplication by message template.
        3. Truncation to 500 chars per entry, stack traces to 5 frames.
        4. Token budget hard cap at 6000 tokens (FIFO eviction).

        Args:
            config: The 'gcp' section of AutopsyConfig.

        Returns:
            Normalized CollectedData with deduplicated log entries.

        Raises:
            GCPAuthError: On credential failure.
            GCPPermissionError: On IAM permission issues.
            CollectorError: On API error or connection failure.
            NoDataError: On zero results.
        """
        project_id = config.get("project_id", "")
        resource_type = config.get("resource_type")
        log_filter = config.get("log_filter")
        time_window = int(config.get("time_window", 30))

        end_ts = datetime.now(tz=timezone.utc)
        start_ts = end_ts - timedelta(minutes=time_window)

        filter_parts = [
            "severity >= ERROR",
            f'timestamp >= "{start_ts.strftime("%Y-%m-%dT%H:%M:%SZ")}"',
            f'timestamp <= "{end_ts.strftime("%Y-%m-%dT%H:%M:%SZ")}"',
        ]
        if resource_type:
            filter_parts.append(f'resource.type = "{resource_type}"')
        if log_filter:
            filter_parts.append(f"({log_filter})")

        filter_str = " AND ".join(filter_parts)

        raw_entries: list[dict[str, Any]] = []

        with console.status(
            f"Querying GCP Cloud Logging: {project_id}...", spinner="dots"
        ):
            try:
                client = gcp_logging.Client(project=project_id)
                iterator = client.list_entries(
                    filter_=filter_str,
                    order_by=DESCENDING,
                    page_size=MAX_ENTRIES,
                    timeout=REQUEST_TIMEOUT,
                )
                for entry in iterator:
                    raw_entries.append(self._normalize_entry(entry))
                    if len(raw_entries) >= MAX_ENTRIES:
                        break
            except gcp_exc.Unauthenticated as exc:
                raise GCPAuthError(
                    message="GCP authentication failed.",
                    hint=(
                        "Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON path,\n"
                        "or run 'gcloud auth application-default login' for local development.\n"
                        "Docs: https://cloud.google.com/docs/authentication/application-default-credentials"
                    ),
                ) from exc
            except gcp_exc.PermissionDenied as exc:
                raise GCPPermissionError(
                    message="Insufficient GCP permissions for Cloud Logging.",
                    hint=(
                        "Your service account needs the 'Logs Viewer' role "
                        "(roles/logging.viewer).\n"
                        "Docs: https://cloud.google.com/logging/docs/access-control"
                    ),
                ) from exc
            except gcp_exc.NotFound as exc:
                raise CollectorError(
                    message=f"GCP project not found: {project_id}",
                    hint=(
                        "Check project_id in config. Find your project ID at:\n"
                        "https://console.cloud.google.com/home/dashboard"
                    ),
                ) from exc
            except gcp_exc.GoogleAPICallError as exc:
                raise CollectorError(
                    message=f"GCP Cloud Logging API error: {exc}",
                    hint="Check your network connectivity and GCP project configuration.",
                ) from exc

        if not raw_entries:
            raise NoDataError(
                message=(
                    f"No error-level logs found in GCP project '{project_id}' "
                    f"for the last {time_window} minutes"
                ),
                hint=(
                    "Check your resource_type filter, or remove it to search all resource types.\n"
                    f"Verify logs exist: gcloud logging read 'severity>=ERROR' "
                    f"--limit=5 --project={project_id}"
                ),
            )

        # Stages 2–4: shared reduction pipeline
        deduped = deduplicate_logs(raw_entries, message_key="message")
        deduped = truncate_entries(deduped, message_key="message")
        deduped, truncated = apply_token_budget(
            deduped,
            message_key="message",
            timestamp_key="timestamp",
        )

        resource_suffix = f"; resource_type={resource_type}" if resource_type else ""
        raw_query = (
            f"GCP Cloud Logging: severity>=ERROR; project={project_id}; "
            f"window={time_window}m{resource_suffix}"
        )

        return CollectedData(
            source="gcp",
            data_type="logs",
            entries=deduped,
            time_range=(start_ts, end_ts),
            raw_query=raw_query,
            entry_count=len(raw_entries),
            truncated=truncated,
        )

    def _extract_message(self, entry: Any) -> str:
        """Extract the log message from a GCP log entry.

        Args:
            entry: A GCP log entry object.

        Returns:
            Log message string.
        """
        payload = entry.payload
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return (
                payload.get("message")
                or payload.get("textPayload")
                or payload.get("msg")
                or payload.get("error")
                or str(payload)
            )
        return str(payload)

    def _normalize_entry(self, entry: Any) -> dict[str, Any]:
        """Convert a GCP log entry to standard CollectedData entry format.

        Args:
            entry: A GCP log entry object.

        Returns:
            Normalized dict with timestamp, message, service, host, etc.
        """
        resource = getattr(entry, "resource", None)
        labels: dict[str, str] = {}
        resource_type = ""
        if resource is not None:
            labels = getattr(resource, "labels", {}) or {}
            resource_type = getattr(resource, "type", "") or ""

        service = (
            labels.get("function_name")
            or labels.get("service_name")
            or labels.get("instance_id")
            or resource_type
            or "unknown"
        )
        host = labels.get("instance_id", "") or labels.get("zone", "")

        ts = getattr(entry, "timestamp", None)
        timestamp = ts.isoformat() if ts is not None else ""

        return {
            "timestamp": timestamp,
            "message": self._extract_message(entry),
            "service": service,
            "host": host,
            "source": "gcp",
            "log_level": getattr(entry, "severity", "ERROR") or "ERROR",
            "resource_type": resource_type,
            "log_name": getattr(entry, "log_name", "") or "",
        }

    @staticmethod
    def _gcp_default_creds_available() -> bool:
        """Check if GCP Application Default Credentials are available.

        Returns:
            True if google.auth.default() succeeds (gcloud CLI or metadata server).
        """
        try:
            import google.auth

            google.auth.default()
            return True
        except Exception:
            return False
