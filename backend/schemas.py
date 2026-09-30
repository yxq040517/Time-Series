from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Quality(StrictModel):
    missing_cells: int
    missing_ratio: float
    duplicate_timestamps: int
    constant_features: list[str]
    ignored_columns: list[str]
    warnings: list[str]


class Dataset(StrictModel):
    id: str
    name: str
    source: Literal["demo", "upload"]
    created_at: str
    rows: int
    features: list[str]
    start_time: str
    end_time: str
    has_labels: bool
    description: str
    quality: Quality


class RunConfig(StrictModel):
    dataset_id: str
    algorithm: Literal["pca", "temporal", "isolation"]
    train_ratio: float = Field(default=.35, ge=.15, le=.7)
    threshold_quantile: float = Field(default=.99, ge=.90, le=.9999)
    pca_variance: float = Field(default=.9, ge=.5, le=.99)
    window: int = Field(default=8, ge=2, le=32, strict=True)
    min_event_length: int = Field(default=3, ge=1, le=50, strict=True)
    merge_gap: int = Field(default=2, ge=0, le=20, strict=True)


class Metrics(StrictModel):
    precision: float
    recall: float
    f1: float
    auprc: float
    event_recall: float


class ScoreStats(StrictModel):
    min: float
    max: float
    median: float
    p95: float


class Summary(StrictModel):
    points: int
    n_features: int
    train_end: int
    fit_end: int
    threshold: float
    anomaly_points: int
    anomaly_ratio: float
    event_count: int
    duration_ms: float
    explanation_method: str
    score_stats: ScoreStats
    metrics: Metrics | None


class Run(StrictModel):
    id: str
    dataset_id: str
    dataset_name: str
    algorithm: Literal["pca", "temporal", "isolation"]
    config: RunConfig
    status: Literal["queued", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    created_at: str
    completed_at: str | None = None
    error: str | None = None
    summary: Summary | None = None


class Event(StrictModel):
    id: str
    start: int
    end: int
    start_time: str
    end_time: str
    length: int
    anomaly_points: int
    peak_score: float
    mean_score: float
    severity: Literal["high", "medium", "low"]
    top_feature: str
    status: Literal["unreviewed", "confirmed", "false_positive"] = "unreviewed"
    note: str = ""


class Annotation(StrictModel):
    status: Literal["unreviewed", "confirmed", "false_positive"] | None = None
    note: str | None = Field(default=None, max_length=2000)


class BatchAnnotation(StrictModel):
    event_ids: list[str] = Field(min_length=1, max_length=500)
    status: Literal["unreviewed", "confirmed", "false_positive"]
    note: str | None = Field(default=None, max_length=2000)

    @field_validator("event_ids")
    @classmethod
    def unique_events(cls, values: list[str]) -> list[str]:
        if len(set(values)) != len(values) or any(not value.strip() for value in values):
            raise ValueError("事件列表不能重复或包含空标识。")
        return values

    @field_validator("note")
    @classmethod
    def valid_note(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("备注不能为 null；不修改备注时请省略该字段。")
        return value


class FeatureProfile(StrictModel):
    name: str
    count: int
    missing: int
    missing_ratio: float
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    p25: float | None
    median: float | None
    p75: float | None


class CorrelationProfile(StrictModel):
    names: list[str]
    values: list[list[float | None]]


class SamplingProfile(StrictModel):
    mode: Literal["timestamp", "index"]
    median_interval_seconds: float | None
    irregular_intervals: int
    duplicate_intervals: int


class DatasetProfile(StrictModel):
    dataset_id: str
    rows: int
    features: list[FeatureProfile]
    correlation: CorrelationProfile
    sampling: SamplingProfile
    warnings: list[str]


class ReviewCounts(StrictModel):
    unreviewed: int
    confirmed: int
    false_positive: int


class SeverityCounts(StrictModel):
    high: int
    medium: int
    low: int


class FeatureEventCount(StrictModel):
    name: str
    event_count: int


class TimelineBucket(StrictModel):
    start: int
    end: int
    label: str
    anomaly_points: int
    total_points: int


class RunInsights(StrictModel):
    run_id: str
    review: ReviewCounts
    severity: SeverityCounts
    top_features: list[FeatureEventCount]
    timeline: list[TimelineBucket]


class DemoRequest(StrictModel):
    seed: int = Field(default=42, ge=0, le=4294967295, strict=True)
