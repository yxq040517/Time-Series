import { useEffect, useState, type FormEvent } from 'react';
import { ChevronDown, Play, SlidersHorizontal, LoaderCircle, Sparkles } from 'lucide-react';
import type { Algorithm, RunConfig } from '../types';
import { getRunPresetConfig, runPresets, type RunPresetId } from './runPresets';

export const algorithmNames: Record<Algorithm, string> = { pca: 'PCA 重构', temporal: '时序自回归', isolation: '孤立森林' };
const descriptions: Record<Algorithm, string> = {
  pca: '学习变量间的正常协同结构，使用重构残差检测偏离。最多保留变量数减一的主成分，确保存在残差方向；适合多变量相关性变化。',
  temporal: '仅使用过去窗口预测当前值，使用 Ridge 自回归预测残差检测变化；不使用未来样本。',
  isolation: '使用 IsolationForest 检测多变量空间中的稀有样本；变量解释为稳健偏离代理，不是模型精确归因。',
};
export function RunConfiguration({ datasetId, busy, onRun, initialConfig }: { datasetId: string; busy: boolean; onRun: (config: RunConfig) => void; initialConfig?: RunConfig }) {
  const [config, setConfig] = useState<RunConfig>(() => initialConfig || getRunPresetConfig('balanced', datasetId));
  const [preset, setPreset] = useState<RunPresetId | 'custom'>(initialConfig ? 'custom' : 'balanced');
  const algorithm = config.algorithm;
  useEffect(() => {
    if (initialConfig) { setConfig(initialConfig); setPreset('custom'); }
  }, [initialConfig]);
  const changeParameter = (key: Exclude<keyof RunConfig, 'dataset_id' | 'algorithm'>, value: string, scale = 1) => {
    setConfig(current => ({ ...current, [key]: value === '' ? Number.NaN : Number(value) / scale }));
    setPreset('custom');
  };
  const numberValue = (key: Exclude<keyof RunConfig, 'dataset_id' | 'algorithm'>, scale = 1) => Number.isFinite(config[key]) ? Number((config[key] * scale).toFixed(4)) : '';
  const choosePreset = (id: RunPresetId) => { setConfig(getRunPresetConfig(id, datasetId, algorithm)); setPreset(id); };
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy) return;
    onRun({ ...config, dataset_id: datasetId,
      pca_variance: algorithm !== 'pca' && !Number.isFinite(config.pca_variance) ? 0.9 : config.pca_variance,
      window: algorithm !== 'temporal' && !Number.isFinite(config.window) ? 8 : config.window,
    });
  };
  return <section className="card configuration" id="configuration"><div className="card-heading"><div className="section-title"><SlidersHorizontal size={18} /><h2>检测配置</h2></div><span className="badge neutral">历史校准</span></div>
    <form onSubmit={submit}>
      <div className="preset-heading"><span><Sparkles size={14} />参数预设</span><small>{preset === 'custom' ? '自定义参数' : '可继续微调'}</small></div>
      <div className="preset-options" role="group" aria-label="检测参数预设">{runPresets.map(item => <button key={item.id} type="button" className={`preset-button ${preset === item.id ? 'active' : ''}`} aria-pressed={preset === item.id} aria-label={item.label} title={item.description} onClick={() => choosePreset(item.id)} disabled={busy}>{item.label}</button>)}</div>
      <p className="preset-description">{preset === 'custom' ? '已调整配置。可复用历史参数，或选择预设重新开始。' : runPresets.find(item => item.id === preset)?.description}</p>
      <label className="field">检测方法<select name="algorithm" aria-label="检测方法" data-testid="algorithm-select" value={algorithm} onChange={(event) => setConfig(current => ({ ...current, algorithm: event.target.value as Algorithm }))} disabled={busy}><option value="pca">PCA · 多变量重构残差</option><option value="temporal">时序自回归 · 历史窗口预测</option><option value="isolation">IsolationForest · 稀有样本检测</option></select></label>
      <p className="method-description">{descriptions[algorithm]}</p>
      <div className="form-grid"><label className="field">历史训练比例（%）<input type="number" name="train_ratio" aria-label="历史训练比例" min={15} max={70} step={1} value={numberValue('train_ratio', 100)} onChange={event => changeParameter('train_ratio', event.target.value, 100)} required disabled={busy} /></label><label className="field">阈值分位数（%）<input type="number" name="threshold_quantile" aria-label="阈值分位数" min={90} max={99.99} step={0.01} value={numberValue('threshold_quantile', 100)} onChange={event => changeParameter('threshold_quantile', event.target.value, 100)} required disabled={busy} /></label></div>
      {algorithm === 'pca' && <label className="field">目标解释方差（%）<input type="number" name="pca_variance" aria-label="目标解释方差" min={50} max={99} step={1} value={numberValue('pca_variance', 100)} onChange={event => changeParameter('pca_variance', event.target.value, 100)} required disabled={busy} /></label>}
      {algorithm === 'temporal' && <label className="field">历史窗口（样本）<input type="number" name="window" aria-label="历史窗口" min={2} max={32} step={1} value={numberValue('window')} onChange={event => changeParameter('window', event.target.value)} required disabled={busy} /></label>}
      <details className="advanced"><summary><ChevronDown size={14} />事件合并设置</summary><div className="form-grid"><label className="field">最短事件（样本）<input type="number" name="min_event_length" aria-label="最短事件长度" min={1} max={50} value={numberValue('min_event_length')} onChange={event => changeParameter('min_event_length', event.target.value)} required disabled={busy} /></label><label className="field">合并间隔（样本）<input type="number" name="merge_gap" aria-label="事件合并间隔" min={0} max={20} value={numberValue('merge_gap')} onChange={event => changeParameter('merge_gap', event.target.value)} required disabled={busy} /></label></div></details>
      <p className="training-note">假设初始训练段主要正常。阈值仅使用训练段内部的历史校准尾段，测试标签不参与建模。模型无法保证发现所有异常。</p>
      <button className="button primary full-width" data-testid="run-button" disabled={busy} type="submit">{busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}{busy ? '正在处理，请稍候' : '开始异常检测'}</button>
    </form>
  </section>;
}
