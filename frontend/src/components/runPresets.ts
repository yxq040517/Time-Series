import type { Algorithm, RunConfig } from '../types';

export type RunPresetId = 'balanced' | 'sensitive' | 'quiet';
export const runPresets: { id: RunPresetId; label: string; description: string }[] = [
  { id: 'balanced', label: '默认平衡', description: '适合首次分析，兼顾发现与复核成本' },
  { id: 'sensitive', label: '敏感发现', description: '降低阈值，保留短暂异常信号' },
  { id: 'quiet', label: '低噪复核', description: '提高阈值，优先持续异常事件' },
];

const parameters: Record<RunPresetId, Omit<RunConfig, 'dataset_id' | 'algorithm'>> = {
  balanced: { train_ratio: 0.35, threshold_quantile: 0.99, pca_variance: 0.9, window: 8, min_event_length: 3, merge_gap: 2 },
  sensitive: { train_ratio: 0.35, threshold_quantile: 0.975, pca_variance: 0.9, window: 6, min_event_length: 1, merge_gap: 1 },
  quiet: { train_ratio: 0.4, threshold_quantile: 0.995, pca_variance: 0.9, window: 12, min_event_length: 5, merge_gap: 3 },
};

export function getRunPresetConfig(id: RunPresetId, datasetId: string, algorithm: Algorithm = 'pca'): RunConfig {
  return { dataset_id: datasetId, algorithm, ...parameters[id] };
}
