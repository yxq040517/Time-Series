export type Algorithm = 'pca' | 'temporal' | 'isolation';
export type ReviewStatus = 'unreviewed' | 'confirmed' | 'false_positive';
export interface Dataset {
  id: string; name: string; source: 'demo' | 'upload'; created_at: string;
  rows: number; features: string[]; start_time: string; end_time: string; has_labels: boolean;
  description: string;
  quality: { missing_cells: number; missing_ratio: number; duplicate_timestamps: number; constant_features: string[]; ignored_columns: string[]; warnings: string[] };
}
export const algorithmNames: Record<Algorithm, string> = { pca: 'PCA 重构', temporal: '时序自回归', isolation: '孤立森林' };
export const algorithmDescriptions: Record<Algorithm, string> = {
  pca: '学习正常变量协同结构，使用有效残差子空间检测偏离。',
  temporal: '仅使用过去窗口，以固定 Ridge 自回归模型预测当前值。',
  isolation: '使用 IsolationForest 检测稀有样本；解释是稳健偏离代理，不是精确归因。',
};
export interface TrainingConfig {
  dataset_id: string; name: string; algorithm: Algorithm; fit_start: number; fit_end: number | null;
  calibration_end: number | null; threshold_quantile: number; pca_variance: number; window: number;
}
export interface DetectionConfig {
  dataset_id: string; model_id: string; min_event_length: number; merge_gap: number; stream_id?: string | null;
}
export interface TrainingJob {
  id: string; dataset_id: string; dataset_name: string; config: TrainingConfig;
  status: 'queued' | 'running' | 'completed' | 'failed'; progress: number; message: string;
  created_at: string; completed_at: string | null; error: string | null; model_id: string | null;
}
export interface ModelInfo {
  id: string; name: string; version: number; algorithm: Algorithm; features: string[];
  status: 'ready' | 'published' | 'disabled'; created_at: string; training_id: string;
  dataset_id: string; dataset_name: string; fit_start: number; fit_end: number; calibration_end: number;
  threshold: number; window: number; training_config: TrainingConfig;
  training_summary: Record<string, unknown>; notes: string[]; format_version: number; sklearn_version: string;
}
export function modelCompatibility(model: ModelInfo, features: string[]) {
  const missing = model.features.filter(name => !features.includes(name));
  const unexpected = features.filter(name => !model.features.includes(name));
  return { missing, unexpected, compatible: !missing.length && !unexpected.length };
}
export function reusableDetectionConfig(run: Run, models: ModelInfo[], dataset: Dataset): DetectionConfig | undefined {
  const model = models.find(item => item.id === run.model_id);
  if (run.config.model_id !== run.model_id || run.config.dataset_id !== run.dataset_id) return;
  if (!model || model.status !== 'published' || !modelCompatibility(model, dataset.features).compatible) return;
  const { min_event_length, merge_gap, stream_id } = run.config;
  if (typeof min_event_length !== 'number' || !Number.isInteger(min_event_length) || min_event_length < 1 || min_event_length > 50 ||
      typeof merge_gap !== 'number' || !Number.isInteger(merge_gap) || merge_gap < 0 || merge_gap > 20 ||
      (stream_id != null && (typeof stream_id !== 'string' || stream_id.length > 100))) return;
  return { dataset_id: dataset.id, model_id: model.id, min_event_length, merge_gap,
    ...(model.algorithm === 'temporal' ? { stream_id: typeof stream_id === 'string' ? stream_id : null } : {}) };
}
export function runModelLabel(run: Run): string {
  return run.model_id ? `${run.model_name || run.model_id} · v${run.model_version ?? '—'}` : '旧版检测 · 无独立模型';
}
export interface Summary {
  points: number; n_features: number; train_end?: number | null; fit_end?: number | null; threshold: number;
  scored_points: number; warmup_points: number;
  anomaly_points: number; anomaly_ratio: number; event_count: number; duration_ms: number;
  explanation_method: string; score_stats: { min: number; max: number; median: number; p95: number } | null;
  metrics: null | { precision: number | null; recall: number | null; f1: number | null; auprc: number | null; event_recall: number | null };
}
export interface Run {
  id: string; dataset_id: string; dataset_name: string; algorithm: Algorithm; config: Record<string, unknown>;
  model_id: string | null; model_name: string | null; model_version: number | null;
  status: 'queued' | 'running' | 'completed' | 'failed'; progress: number; message: string;
  created_at: string; completed_at: string | null; error: string | null; summary: Summary | null;
}
export interface AnomalyEvent {
  id: string; start: number; end: number; start_time: string; end_time: string; length: number;
  anomaly_points: number; peak_score: number; mean_score: number; severity: 'high' | 'medium' | 'low';
  top_feature: string; status: ReviewStatus; note: string;
}
export interface Series {
  indices: number[]; timestamps: string[]; scores: (number | null)[]; threshold: number; train_end?: number | null;
  feature_names: string[]; values: Record<string, number[]>; reference: Record<string, (number | null)[]>;
  contributions: Record<string, (number | null)[]>; flags: number[]; scored: boolean[]; labels: number[] | null;
  sampling: { total_points: number; returned_points: number };
}
export interface Heatmap {
  features: string[]; timestamps: string[]; indices: number[]; values: (number | null)[][]; method: string;
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
