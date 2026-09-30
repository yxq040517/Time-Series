export type Algorithm = 'pca' | 'temporal' | 'isolation';
export type ReviewStatus = 'unreviewed' | 'confirmed' | 'false_positive';
export interface Dataset {
  id: string; name: string; source: 'demo' | 'upload'; created_at: string;
  rows: number; features: string[]; start_time: string; end_time: string; has_labels: boolean;
  description: string;
  quality: { missing_cells: number; missing_ratio: number; duplicate_timestamps: number; constant_features: string[]; ignored_columns: string[]; warnings: string[] };
}
export interface RunConfig {
  dataset_id: string; algorithm: Algorithm; train_ratio: number; threshold_quantile: number;
  pca_variance: number; window: number; min_event_length: number; merge_gap: number;
}
export interface Summary {
  points: number; n_features: number; train_end: number; fit_end: number; threshold: number;
  anomaly_points: number; anomaly_ratio: number; event_count: number; duration_ms: number;
  explanation_method: string; score_stats: { min: number; max: number; median: number; p95: number };
  metrics: null | { precision: number | null; recall: number | null; f1: number | null; auprc: number | null; event_recall: number | null };
}
export interface Run {
  id: string; dataset_id: string; dataset_name: string; algorithm: Algorithm; config: RunConfig;
  status: 'queued' | 'running' | 'completed' | 'failed'; progress: number; message: string;
  created_at: string; completed_at: string | null; error: string | null; summary: Summary | null;
}
export interface AnomalyEvent {
  id: string; start: number; end: number; start_time: string; end_time: string; length: number;
  anomaly_points: number; peak_score: number; mean_score: number; severity: 'high' | 'medium' | 'low';
  top_feature: string; status: ReviewStatus; note: string;
}
export interface Series {
  indices: number[]; timestamps: string[]; scores: number[]; threshold: number; train_end: number;
  feature_names: string[]; values: Record<string, number[]>; reference: Record<string, number[]>;
  contributions: Record<string, number[]>; flags: number[]; labels: number[] | null;
  sampling: { total_points: number; returned_points: number };
}
export interface Heatmap {
  features: string[]; timestamps: string[]; indices: number[]; values: number[][]; method: string;
  range: { start: number; end: number };
}
export interface Explanation {
  event: AnomalyEvent; method: string;
  top_features: { name: string; contribution: number; share: number; rank: number }[];
  context: { start: number; end: number }; notes: string[];
}
export interface Preview { columns: string[]; rows: Record<string, string | number | null>[] }
export type SampleRange = { start: number; end: number };
export interface FeatureProfile {
  name: string; count: number; missing: number; missing_ratio: number;
  min: number | null; max: number | null; mean: number | null; std: number | null;
  p25: number | null; median: number | null; p75: number | null;
}
export interface DatasetProfile {
  dataset_id: string; rows: number; features: FeatureProfile[];
  correlation: { names: string[]; values: (number | null)[][] };
  sampling: { mode: 'timestamp' | 'index'; median_interval_seconds: number | null; irregular_intervals: number; duplicate_intervals: number };
  warnings: string[];
}
export interface RunInsights {
  run_id: string;
  review: Record<ReviewStatus, number>;
  severity: Record<AnomalyEvent['severity'], number>;
  top_features: { name: string; event_count: number }[];
  timeline: { start: number; end: number; label: string; anomaly_points: number; total_points: number }[];
}
