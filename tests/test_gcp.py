"""Tests for the GCP Cloud Logging collector."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from autopsy.collectors.gcp import GCPCollector
from autopsy.utils.errors import (
    CollectorError,
    GCPAuthError,
    GCPPermissionError,
    NoDataError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gcp_config(**overrides: Any) -> dict[str, Any]:
    """Build a minimal GCP config dict."""
    cfg: dict[str, Any] = {
        "project_id": "my-proj-123",
        "credentials_env": "GOOGLE_APPLICATION_CREDENTIALS",
        "resource_type": None,
        "log_filter": None,
        "time_window": 30,
    }
    cfg.update(overrides)
    return cfg


def _mock_entry(
    *,
    message: str = "ERROR: something went wrong",
    severity: str = "ERROR",
    resource_type: str = "cloud_function",
    function_name: str = "my-func",
    log_name: str = "projects/my-proj-123/logs/cloudfunction",
    timestamp: datetime | None = None,
) -> MagicMock:
    """Build a mock GCP log entry."""
    entry = MagicMock()
    entry.payload = message
    entry.severity = severity
    entry.log_name = log_name
    entry.timestamp = timestamp or datetime(2026, 3, 15, 2, 47, 12, tzinfo=timezone.utc)
    entry.insert_id = "abc123"
    entry.trace = ""
    resource = MagicMock()
    resource.type = resource_type
    resource.labels = {"function_name": function_name}
    entry.resource = resource
    return entry


# ---------------------------------------------------------------------------
# TestValidateConfig
# ---------------------------------------------------------------------------


class TestValidateConfig:
    def test_validate_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/path/to/sa.json")
        collector = GCPCollector()

        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter([_mock_entry()])
        with patch("google.cloud.logging.Client", return_value=mock_client):
            result = collector.validate_config(_gcp_config())

        assert result is True

    def test_validate_unauthenticated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.Unauthenticated("bad auth")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(GCPAuthError) as exc_info,
        ):
            collector.validate_config(_gcp_config())

        assert "authentication failed" in exc_info.value.message.lower()
        assert "GOOGLE_APPLICATION_CREDENTIALS" in exc_info.value.hint

    def test_validate_permission_denied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.PermissionDenied("denied")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(GCPPermissionError) as exc_info,
        ):
            collector.validate_config(_gcp_config())

        assert "permission" in exc_info.value.message.lower()
        assert "roles/logging.viewer" in exc_info.value.hint

    def test_validate_project_not_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.NotFound("not found")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(CollectorError) as exc_info,
        ):
            collector.validate_config(_gcp_config())

        assert "not found" in exc_info.value.message.lower()

    def test_validate_api_call_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.GoogleAPICallError("api error")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(CollectorError) as exc_info,
        ):
            collector.validate_config(_gcp_config())

        assert "api error" in exc_info.value.message.lower()

    def test_validate_no_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        collector = GCPCollector()

        with (
            patch.object(collector, "_gcp_default_creds_available", return_value=False),
            pytest.raises(GCPAuthError) as exc_info,
        ):
            collector.validate_config(_gcp_config())

        assert "not configured" in exc_info.value.message.lower()
        assert "Option 1" in exc_info.value.hint
        assert "Option 2" in exc_info.value.hint
        assert "Option 3" in exc_info.value.hint


# ---------------------------------------------------------------------------
# TestCollect
# ---------------------------------------------------------------------------


class TestCollect:
    def test_collect_happy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        entries = [_mock_entry(message=f"ERROR {i}") for i in range(3)]
        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter(entries)
        with patch("google.cloud.logging.Client", return_value=mock_client):
            result = collector.collect(_gcp_config())

        assert result.source == "gcp"
        assert result.data_type == "logs"
        assert result.entry_count == 3

    def test_collect_with_resource_type(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()
        captured_filter: list[str] = []

        def fake_list_entries(**kwargs: Any) -> Any:
            captured_filter.append(kwargs.get("filter_", ""))
            return iter([_mock_entry()])

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = fake_list_entries
        with patch("google.cloud.logging.Client", return_value=mock_client):
            collector.collect(_gcp_config(resource_type="cloud_function"))

        assert captured_filter
        assert 'resource.type = "cloud_function"' in captured_filter[0]

    def test_collect_with_custom_filter(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()
        captured_filter: list[str] = []

        def fake_list_entries(**kwargs: Any) -> Any:
            captured_filter.append(kwargs.get("filter_", ""))
            return iter([_mock_entry()])

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = fake_list_entries
        with patch("google.cloud.logging.Client", return_value=mock_client):
            collector.collect(_gcp_config(log_filter="jsonPayload.service=payment"))

        assert captured_filter
        assert "(jsonPayload.service=payment)" in captured_filter[0]

    def test_collect_empty_results_raises_no_data_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter([])
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(NoDataError) as exc_info,
        ):
            collector.collect(_gcp_config())

        assert "no error-level logs" in exc_info.value.message.lower()

    def test_collect_auth_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.Unauthenticated("bad")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(GCPAuthError),
        ):
            collector.collect(_gcp_config())

    def test_collect_permission_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        try:
            from google.api_core import exceptions as gcp_exc
        except ImportError:
            pytest.skip("google-cloud-logging not installed")

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = gcp_exc.PermissionDenied("denied")
        with (
            patch("google.cloud.logging.Client", return_value=mock_client),
            pytest.raises(GCPPermissionError),
        ):
            collector.collect(_gcp_config())

    def test_collect_max_200_entries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        entries = [_mock_entry(message=f"ERROR {i}") for i in range(500)]
        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter(entries)
        with patch("google.cloud.logging.Client", return_value=mock_client):
            result = collector.collect(_gcp_config())

        assert result.entry_count <= 200

    def test_collect_filter_contains_timestamps(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()
        captured_filter: list[str] = []

        def fake_list_entries(**kwargs: Any) -> Any:
            captured_filter.append(kwargs.get("filter_", ""))
            return iter([_mock_entry()])

        mock_client = MagicMock()
        mock_client.list_entries.side_effect = fake_list_entries
        with patch("google.cloud.logging.Client", return_value=mock_client):
            collector.collect(_gcp_config())

        assert captured_filter
        filt = captured_filter[0]
        assert "severity >= ERROR" in filt
        assert "timestamp >=" in filt
        assert "timestamp <=" in filt


# ---------------------------------------------------------------------------
# TestExtractMessage
# ---------------------------------------------------------------------------


class TestExtractMessage:
    def setup_method(self) -> None:
        self.collector = GCPCollector()

    def _make_entry(self, payload: Any) -> MagicMock:
        e = MagicMock()
        e.payload = payload
        return e

    def test_text_payload(self) -> None:
        entry = self._make_entry("simple error message")
        assert self.collector._extract_message(entry) == "simple error message"

    def test_json_payload_message_key(self) -> None:
        entry = self._make_entry({"message": "json error", "level": "ERROR"})
        assert self.collector._extract_message(entry) == "json error"

    def test_json_payload_text_payload_key(self) -> None:
        entry = self._make_entry({"textPayload": "text error"})
        assert self.collector._extract_message(entry) == "text error"

    def test_json_payload_msg_key(self) -> None:
        entry = self._make_entry({"msg": "msg error"})
        assert self.collector._extract_message(entry) == "msg error"

    def test_json_payload_error_key(self) -> None:
        entry = self._make_entry({"error": "error field value"})
        assert self.collector._extract_message(entry) == "error field value"

    def test_json_payload_fallback_to_str(self) -> None:
        payload = {"unknown_key": "data"}
        entry = self._make_entry(payload)
        result = self.collector._extract_message(entry)
        assert result == str(payload)

    def test_proto_or_unknown_payload(self) -> None:
        """Non-string, non-dict payload falls back to str()."""

        class FakeProto:
            def __str__(self) -> str:
                return "proto string"

        entry = self._make_entry(FakeProto())
        assert self.collector._extract_message(entry) == "proto string"


# ---------------------------------------------------------------------------
# TestNormalizeEntry
# ---------------------------------------------------------------------------


class TestNormalizeEntry:
    def setup_method(self) -> None:
        self.collector = GCPCollector()

    def test_normalize_cloud_function(self) -> None:
        entry = _mock_entry(
            resource_type="cloud_function",
            function_name="process-order",
            message="ERROR: timeout",
        )
        result = self.collector._normalize_entry(entry)
        assert result["source"] == "gcp"
        assert result["resource_type"] == "cloud_function"
        assert result["service"] == "process-order"
        assert result["message"] == "ERROR: timeout"
        assert result["log_level"] == "ERROR"

    def test_normalize_gce_instance(self) -> None:
        entry = _mock_entry(resource_type="gce_instance")
        entry.resource.labels = {"instance_id": "inst-123", "zone": "us-central1-a"}
        result = self.collector._normalize_entry(entry)
        assert result["service"] == "inst-123"
        assert result["host"] == "inst-123"
        assert result["resource_type"] == "gce_instance"

    def test_normalize_gke_container(self) -> None:
        entry = _mock_entry(resource_type="gke_container")
        entry.resource.labels = {"service_name": "api-service", "cluster_name": "prod"}
        result = self.collector._normalize_entry(entry)
        assert result["service"] == "api-service"
        assert result["resource_type"] == "gke_container"

    def test_normalize_missing_resource(self) -> None:
        """Entry with no resource attribute should not crash."""
        entry = _mock_entry()
        entry.resource = None
        result = self.collector._normalize_entry(entry)
        assert result["source"] == "gcp"
        assert result["service"] == "unknown"
        assert result["resource_type"] == ""

    def test_normalize_missing_timestamp(self) -> None:
        entry = _mock_entry()
        entry.timestamp = None
        result = self.collector._normalize_entry(entry)
        assert result["timestamp"] == ""


# ---------------------------------------------------------------------------
# TestDeduplicationPipeline
# ---------------------------------------------------------------------------


class TestDeduplicationPipeline:
    def test_dedup_removes_duplicates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        entries = [_mock_entry(message="ERROR: NullPointerException") for _ in range(5)]
        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter(entries)
        with patch("google.cloud.logging.Client", return_value=mock_client):
            result = collector.collect(_gcp_config())

        assert len(result.entries) == 1
        assert result.entries[0]["occurrences"] == 5

    def test_token_budget_applied_when_over_limit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/sa.json")
        collector = GCPCollector()

        # 200 unique entries with long messages → should exceed 6000-token budget
        long_msg = "X" * 600
        entries = [_mock_entry(message=f"{long_msg} entry_{i}") for i in range(200)]
        mock_client = MagicMock()
        mock_client.list_entries.return_value = iter(entries)
        with patch("google.cloud.logging.Client", return_value=mock_client):
            result = collector.collect(_gcp_config())

        assert result.truncated is True


# ---------------------------------------------------------------------------
# TestGCPConfig
# ---------------------------------------------------------------------------


class TestGCPConfig:
    def test_gcp_config_optional_absent(self) -> None:
        from autopsy.config import AutopsyConfig

        cfg = AutopsyConfig(
            aws={"region": "us-east-1", "log_groups": ["/aws/lambda/api"]},
            github={"repo": "owner/repo"},
            ai={},
        )
        assert cfg.gcp is None

    def test_gcp_config_valid(self) -> None:
        from autopsy.config import GCPConfig

        cfg = GCPConfig(project_id="my-proj-123")
        assert cfg.project_id == "my-proj-123"
        assert cfg.credentials_env == "GOOGLE_APPLICATION_CREDENTIALS"
        assert cfg.resource_type is None
        assert cfg.time_window == 30

    def test_gcp_config_full(self) -> None:
        from autopsy.config import GCPConfig

        cfg = GCPConfig(
            project_id="my-proj-123",
            resource_type="cloud_function",
            log_filter="jsonPayload.service=api",
            time_window=15,
        )
        assert cfg.resource_type == "cloud_function"
        assert cfg.log_filter == "jsonPayload.service=api"
        assert cfg.time_window == 15

    def test_project_id_too_short(self) -> None:
        from autopsy.config import GCPConfig

        with pytest.raises(Exception, match="6-30"):
            GCPConfig(project_id="ab")

    def test_project_id_too_long(self) -> None:
        from autopsy.config import GCPConfig

        with pytest.raises(Exception, match="6-30"):
            GCPConfig(project_id="a" * 31)

    def test_time_window_too_small(self) -> None:
        from autopsy.config import GCPConfig

        with pytest.raises(Exception, match="greater than or equal to 5"):
            GCPConfig(project_id="my-proj-123", time_window=2)

    def test_time_window_too_large(self) -> None:
        from autopsy.config import GCPConfig

        with pytest.raises(Exception, match="less than or equal to 60"):
            GCPConfig(project_id="my-proj-123", time_window=61)

    def test_credentials_env_customized(self) -> None:
        from autopsy.config import GCPConfig

        cfg = GCPConfig(project_id="my-proj-123", credentials_env="MY_GCP_CREDS")
        assert cfg.credentials_env == "MY_GCP_CREDS"

    def test_autopsy_config_with_gcp(self) -> None:
        from autopsy.config import AutopsyConfig, GCPConfig

        cfg = AutopsyConfig(
            aws={"region": "us-east-1", "log_groups": ["/aws/lambda/api"]},
            gcp={"project_id": "my-proj-123"},
            github={"repo": "owner/repo"},
            ai={},
        )
        assert cfg.gcp is not None
        assert isinstance(cfg.gcp, GCPConfig)
        assert cfg.gcp.project_id == "my-proj-123"


# ---------------------------------------------------------------------------
# TestOrchestratorIntegration
# ---------------------------------------------------------------------------


class TestOrchestratorIntegration:
    def _make_config(self, *, include_gcp: bool = True) -> Any:
        from autopsy.config import AutopsyConfig

        base: dict[str, Any] = {
            "aws": {"region": "us-east-1", "log_groups": ["/aws/lambda/api"]},
            "github": {"repo": "owner/repo"},
            "ai": {},
        }
        if include_gcp:
            base["gcp"] = {"project_id": "my-proj-123"}
        return AutopsyConfig(**base)

    def test_gcp_collector_added_when_configured(self) -> None:
        from autopsy.diagnosis import DiagnosisOrchestrator

        cfg = self._make_config(include_gcp=True)
        orch = DiagnosisOrchestrator(cfg)
        collectors = orch._get_collectors()
        roles = [getattr(c, "_autopsy_role", c.name) for c in collectors]
        assert "gcp" in roles

    def test_gcp_collector_absent_when_not_configured(self) -> None:
        from autopsy.diagnosis import DiagnosisOrchestrator

        cfg = self._make_config(include_gcp=False)
        orch = DiagnosisOrchestrator(cfg)
        collectors = orch._get_collectors()
        roles = [getattr(c, "_autopsy_role", c.name) for c in collectors]
        assert "gcp" not in roles

    def test_gcp_skipped_when_no_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from autopsy.diagnosis import DiagnosisOrchestrator

        monkeypatch.delenv("GOOGLE_APPLICATION_CREDENTIALS", raising=False)
        cfg = self._make_config(include_gcp=True)
        orch = DiagnosisOrchestrator(cfg)

        aws_dict = cfg.aws.model_dump()
        with patch(
            "autopsy.collectors.gcp.GCPCollector._gcp_default_creds_available",
            return_value=False,
        ):
            tasks = orch._resolve_collector_tasks(aws_dict, None)

        task_roles = [t.role for t in tasks]
        assert "gcp" not in task_roles

    def test_gcp_coexists_with_cloudwatch_and_datadog(self) -> None:
        from autopsy.config import AutopsyConfig
        from autopsy.diagnosis import DiagnosisOrchestrator

        cfg = AutopsyConfig(
            aws={"region": "us-east-1", "log_groups": ["/aws/lambda/api"]},
            datadog={"site": "us1"},
            gcp={"project_id": "my-proj-123"},
            github={"repo": "owner/repo"},
            ai={},
        )
        orch = DiagnosisOrchestrator(cfg)
        collectors = orch._get_collectors()
        roles = [getattr(c, "_autopsy_role", c.name) for c in collectors]
        assert "cloudwatch" in roles
        assert "datadog" in roles
        assert "gcp" in roles

    def test_collector_name_property(self) -> None:
        assert GCPCollector().name == "gcp"
