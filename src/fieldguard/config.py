"""YAML project configuration, validated with pydantic.

A config maps the roles FieldGuard needs (respondent ID, enumerator, cluster, timestamps, GPS ...)
onto the column names of a KoboToolbox / ODK / SurveyCTO export and parameterises every check.
Unknown keys are rejected so that typos surface immediately.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, field_validator, model_validator

from fieldguard.models import Severity

Platform = Literal["auto", "kobo", "odk", "surveycto", "generic"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class ConfigError(ValueError):
    """Raised for invalid FieldGuard configuration."""


def _check_tz(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        ZoneInfo(value)
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ValueError(f"unknown IANA timezone {value!r} (e.g. 'Africa/Nairobi', 'Asia/Karachi', 'UTC')") from exc
    return value


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "project"


class Fieldwork(StrictModel):
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def _order(self) -> Fieldwork:
        if self.start_date and self.end_date and self.end_date < self.start_date:
            raise ValueError("fieldwork.end_date is before fieldwork.start_date")
        return self


class ProjectInfo(StrictModel):
    name: str = "FieldGuard project"
    slug: str | None = None
    platform: Platform = "auto"
    timezone: str = Field("UTC", description="Local timezone of fieldwork (IANA name).")
    naive_timestamps_timezone: str | None = Field(
        None, description="Timezone for start/end values without an offset (default: project timezone)."
    )
    submission_time_timezone: str | None = Field(
        None, description="Timezone of naive submission timestamps (default: platform convention, UTC for Kobo/ODK)."
    )
    fieldwork: Fieldwork = Field(default_factory=Fieldwork)

    @field_validator("timezone", "naive_timestamps_timezone", "submission_time_timezone")
    @classmethod
    def _tz(cls, value: str | None) -> str | None:
        return _check_tz(value)

    @model_validator(mode="after")
    def _slug(self) -> ProjectInfo:
        if not self.slug:
            self.slug = slugify(self.name)
        return self


class SourceConfig(StrictModel):
    """Remote data source (credentials always come from environment variables)."""

    type: Literal["kobo", "odk", "surveycto"]
    server_url: str | None = None
    asset_uid: str | None = None  # Kobo
    project_id: int | str | None = None  # ODK Central
    form_id: str | None = None  # ODK Central xmlFormId / SurveyCTO form id
    page_size: int = Field(1000, ge=1, le=30000)


class DataConfig(StrictModel):
    path: str | None = None
    sheet: str | int | None = 0
    delimiter: str | None = None
    encoding: str = "utf-8-sig"
    source: SourceConfig | None = None


class GPSRoles(StrictModel):
    lat: str | None = None
    lon: str | None = None
    accuracy: str | None = None
    point: str | None = Field(None, description="Single geopoint column 'lat lon alt accuracy' (Kobo/ODK style).")


class Roles(StrictModel):
    submission_id: str | None = None
    respondent_id: str | None = None
    enumerator_id: str | None = None
    cluster: str | None = None
    team: str | None = None
    start: str | None = None
    end: str | None = None
    submission_time: str | None = None
    survey_date: str | None = None
    duration: str | None = None
    duration_unit: Literal["seconds", "minutes"] = "seconds"
    deviceid: str | None = None
    gps: GPSRoles | None = None


class SpecialCodes(StrictModel):
    dont_know: list[str | int | float] = Field(default_factory=lambda: [-99, "-99", "dk", "dont_know", "don't know"])
    refused: list[str | int | float] = Field(default_factory=lambda: [-98, "-98", "refused", "refuse"])
    not_applicable: list[str | int | float] = Field(default_factory=lambda: [-97, "-97"])


class GridSpec(StrictModel):
    columns: list[str]
    min_items: int = Field(4, ge=2)


class ConstraintRule(StrictModel):
    id: str | None = None
    variable: str | None = None
    min: float | None = None
    max: float | None = None
    soft_min: float | None = None
    soft_max: float | None = None
    allowed: list[str | int | float] | None = None
    expression: str | None = Field(None, description="Must be true, e.g. 'n_children <= hh_size'.")
    when: str | None = Field(None, description="Only evaluate where this expression is true.")
    severity: Severity = Severity.WARNING
    message: str | None = None

    @model_validator(mode="after")
    def _needs_target(self) -> ConstraintRule:
        if not self.variable and not self.expression:
            raise ValueError("a constraint needs `variable` (with min/max/allowed) or `expression`")
        if (
            self.variable
            and not self.expression
            and all(v is None for v in (self.min, self.max, self.soft_min, self.soft_max, self.allowed))
        ):
            raise ValueError(f"constraint on {self.variable!r} needs min/max/soft_min/soft_max/allowed")
        return self


class SkipRule(StrictModel):
    id: str | None = None
    if_: str = Field(alias="if", description="Condition, Python or XLSForm syntax, e.g. \"q5 == 'no'\".")
    then_empty: list[str] = Field(default_factory=list)
    then_required: list[str] = Field(default_factory=list)
    then: str | None = Field(None, description="Expression that must hold when the condition is true.")
    severity: Severity = Severity.WARNING
    message: str | None = None

    @model_validator(mode="after")
    def _needs_consequence(self) -> SkipRule:
        if not (self.then_empty or self.then_required or self.then):
            raise ValueError("a skip-logic rule needs then_empty, then_required or then")
        return self


class RequiredSpec(StrictModel):
    variable: str
    when: str | None = None
    severity: Severity = Severity.WARNING


class Variables(StrictModel):
    key: list[str] = Field(default_factory=list, description="Key outcome variables (enumerator effects, ICC).")
    numeric: list[str] = Field(default_factory=list)
    required: list[str | RequiredSpec] = Field(default_factory=list)
    special_codes: SpecialCodes = Field(default_factory=SpecialCodes)
    grids: dict[str, list[str] | GridSpec] = Field(default_factory=dict)
    constraints: list[ConstraintRule] = Field(default_factory=list)
    skip_logic: list[SkipRule] = Field(default_factory=list)
    nonresponse: list[str] | None = Field(None, description="Variables for DK/refusal rates (default: all answers).")
    exclude: list[str] = Field(default_factory=list, description="Columns never treated as answers (e.g. notes).")

    def grid_specs(self) -> dict[str, GridSpec]:
        return {k: (v if isinstance(v, GridSpec) else GridSpec(columns=v)) for k, v in self.grids.items()}

    def required_specs(self) -> list[RequiredSpec]:
        return [r if isinstance(r, RequiredSpec) else RequiredSpec(variable=r) for r in self.required]


class ScoringRule(StrictModel):
    metric: str
    lo: float = Field(description="Value where risk points start to accrue.")
    hi: float = Field(description="Value where the indicator reaches full points (lo > hi means lower is worse).")
    weight: float = Field(1.0, ge=0)
    label: str | None = None


class ScoringConfig(StrictModel):
    enabled: bool = True
    rules: list[ScoringRule] | None = Field(None, description="Replace the default rule set entirely.")
    rule_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    points_per_weight: float = 10.0
    anomaly_method: Literal["isolation_forest", "lof", "none"] = "isolation_forest"
    anomaly_weight: float = Field(0.25, ge=0, le=1)
    flag_threshold: float = Field(30.0, ge=0, le=100)
    min_enumerators_for_anomaly: int = Field(6, ge=3)
    min_interviews: int = Field(3, ge=1)
    random_state: int = 42  # models always run single-job (see scoring.anomaly); no parallel workers


class BackcheckVariable(StrictModel):
    name: str
    type: Literal[1, 2, 3] = 1
    kind: Literal["categorical", "numeric", "text", "date"] = "categorical"
    abs_tolerance: float = 0.0
    rel_tolerance: float = 0.0
    text_similarity: float = Field(0.85, ge=0, le=1)
    date_tolerance_days: int = 0
    ignore_values: list[str | int | float] = Field(default_factory=list)
    backcheck_name: str | None = Field(None, description="Column name in the back-check data, if different.")


class BackcheckConfig(StrictModel):
    id_column: str | None = Field(None, description="Join key (default: respondent_id role).")
    backchecker_column: str | None = None
    variables: list[BackcheckVariable] = Field(default_factory=list)
    type1_threshold: float = Field(0.10, ge=0, le=1)
    type2_threshold: float = Field(0.10, ge=0, le=1)
    type3_threshold: float = Field(0.10, ge=0, le=1)
    missing_as_value: bool = True
    min_compared: int = Field(5, ge=1)
    sample_rate: float = Field(0.10, gt=0, le=1)
    min_per_enumerator: int = Field(2, ge=0)
    risk_oversample: float = Field(1.0, ge=0)
    flagged_record_weight: float = Field(3.0, ge=1)
    random_share: float = Field(0.5, ge=0, le=1)
    contact_columns: list[str] = Field(default_factory=list)
    exclude_ids: list[str] = Field(default_factory=list)
    seed: int = 12345

    def threshold(self, vtype: int) -> float:
        return {1: self.type1_threshold, 2: self.type2_threshold, 3: self.type3_threshold}[vtype]


class PrivacyConfig(StrictModel):
    pii_columns: list[str] = Field(default_factory=list)
    redact_pii_in_reports: bool = True
    hash_respondent_ids: bool = False
    pseudonymize_enumerators: bool = False
    gps_decimals_in_reports: int | None = Field(3, ge=0, le=6)
    hash_salt_env: str = "FIELDGUARD_HASH_SALT"


OutputFormat = Literal["csv", "parquet", "excel", "html"]


class OutputConfig(StrictModel):
    dir: str = "outputs/{project}/{run_id}"
    formats: list[OutputFormat] = Field(default_factory=lambda: ["csv", "parquet", "excel", "html"])
    max_rows_per_sheet: int = Field(100_000, ge=100, le=1_000_000)
    max_issues_per_check_html: int = Field(200, ge=0)


class SinkConfig(StrictModel):
    postgres: bool = False
    database_url_env: str = "FIELDGUARD_DATABASE_URL"
    schema_name: str = "fieldguard"


class IncrementalConfig(StrictModel):
    enabled: bool = False
    state: Literal["auto", "file", "postgres"] = "auto"
    state_dir: str = ".fieldguard_state"


class FieldGuardConfig(StrictModel):
    project: ProjectInfo = Field(default_factory=ProjectInfo)
    data: DataConfig = Field(default_factory=DataConfig)
    roles: Roles = Field(default_factory=Roles)
    variables: Variables = Field(default_factory=Variables)
    checks: dict[str, dict[str, Any]] = Field(default_factory=dict)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    backcheck: BackcheckConfig = Field(default_factory=BackcheckConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    outputs: OutputConfig = Field(default_factory=OutputConfig)
    sink: SinkConfig = Field(default_factory=SinkConfig)
    incremental: IncrementalConfig = Field(default_factory=IncrementalConfig)
    synthetic: bool = Field(False, description="Marks the data as synthetic; every output is labelled accordingly.")

    _base_dir: Path = PrivateAttr(default_factory=Path.cwd)
    _params: dict[str, BaseModel] = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _validate_checks(self) -> FieldGuardConfig:
        from fieldguard.checks.registry import get_registry

        registry = get_registry()
        unknown = sorted(set(self.checks) - set(registry))
        if unknown:
            raise ValueError(f"unknown check id(s) in `checks`: {unknown}; available: {sorted(registry)}")
        params: dict[str, BaseModel] = {}
        for check_id, spec in registry.items():
            raw = self.checks.get(check_id) or {}
            try:
                params[check_id] = spec.params_model.model_validate(raw)
            except ValidationError as exc:
                raise ValueError(f"invalid parameters for check {check_id!r}: {exc}") from exc
        self._params = params
        return self

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    def params_for(self, check_id: str) -> Any:
        if check_id not in self._params:
            from fieldguard.checks.registry import get_registry

            self._params[check_id] = get_registry()[check_id].params_model()
        return self._params[check_id]

    def resolve_path(self, value: str | Path) -> Path:
        p = Path(value)
        return p if p.is_absolute() else (self._base_dir / p)


def config_from_dict(data: dict[str, Any] | None, base_dir: str | Path | None = None) -> FieldGuardConfig:
    try:
        cfg = FieldGuardConfig.model_validate(data or {})
    except ValidationError as exc:
        raise ConfigError(_format_validation_error(exc)) from exc
    if base_dir is not None:
        cfg._base_dir = Path(base_dir)
    return cfg


def load_config(path: str | Path) -> FieldGuardConfig:
    """Load and validate a YAML config file; relative paths resolve against its folder."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p}: top level of the config must be a mapping")
    return config_from_dict(data, base_dir=p.resolve().parent)


def dump_config(cfg: FieldGuardConfig) -> str:
    return yaml.safe_dump(cfg.model_dump(mode="json", by_alias=True, exclude_none=True), sort_keys=False)


def _format_validation_error(exc: ValidationError) -> str:
    lines = ["invalid FieldGuard config:"]
    for err in exc.errors():
        loc = ".".join(str(x) for x in err["loc"]) or "<root>"
        lines.append(f"  - {loc}: {err['msg']}")
    return "\n".join(lines)
