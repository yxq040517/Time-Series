import { useEffect, useState, type FormEvent } from 'react';
import { Play, SlidersHorizontal, LoaderCircle } from 'lucide-react';
import { algorithmNames, algorithmDescriptions, modelCompatibility, type Dataset, type DetectionConfig, type ModelInfo } from '../types';

export function RunConfiguration({ dataset, models, busy, onRun, initialConfig, onTrain, modelLoading, modelError, onRefresh }: {
  dataset: Dataset; models: ModelInfo[]; busy: boolean; onRun: (config: DetectionConfig) => void;
  initialConfig?: DetectionConfig; onTrain: () => void; modelLoading: boolean; modelError: string; onRefresh: () => void;
}) {
  const usable = models.filter(model => model.status === 'published' && modelCompatibility(model, dataset.features).compatible);
  const [modelId, setModelId] = useState(initialConfig?.model_id || '');
  const [minLength, setMinLength] = useState(String(initialConfig?.min_event_length ?? 3));
  const [mergeGap, setMergeGap] = useState(String(initialConfig?.merge_gap ?? 2));
  const [streamId, setStreamId] = useState(initialConfig?.stream_id || '');
  useEffect(() => {
    if (initialConfig) { setModelId(initialConfig.model_id); setMinLength(String(initialConfig.min_event_length)); setMergeGap(String(initialConfig.merge_gap)); setStreamId(initialConfig.stream_id || ''); }
  }, [initialConfig]);
  const model = usable.find(item => item.id === modelId);
  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (busy || modelLoading || modelError || !model) return;
    onRun({ dataset_id: dataset.id, model_id: model.id, min_event_length: Number(minLength), merge_gap: Number(mergeGap),
      ...(model.algorithm === 'temporal' ? { stream_id: streamId.trim() || null } : {}) });
  };
  return <section className="card configuration" id="configuration"><div className="card-heading"><div className="section-title"><SlidersHorizontal size={18} /><h2>检测配置</h2></div><span className="badge neutral">固定模型推理</span></div>
    {modelLoading && <p role="status">正在读取已发布模型…</p>}
    {modelError && <div className="error-banner" role="alert">{modelError}<button className="text-button" onClick={onRefresh}>重试</button></div>}
    {!modelLoading && !usable.length && <div className="notice"><span>没有已发布且变量兼容的模型。请先独立训练并发布，检测不会自动训练。<button className="text-button" data-testid="train-empty-action" onClick={onTrain}>前往模型训练</button></span></div>}
    <form onSubmit={submit}>
      <label className="field">已发布模型<select name="model_id" aria-label="已发布模型" data-testid="model-select" value={model?.id || ''} onChange={event => { setModelId(event.target.value); setStreamId(''); }} required disabled={busy || modelLoading || !!modelError}><option value="" disabled>选择与当前数据变量兼容的模型</option>{usable.map(item => <option key={item.id} value={item.id}>{item.name} · v{item.version} · {algorithmNames[item.algorithm]}</option>)}</select></label>
      {model && <div className="fixed-model-parameters" data-testid="fixed-model-parameters"><strong>{algorithmNames[model.algorithm]} · v{model.version}</strong><p className="method-description">{algorithmDescriptions[model.algorithm]}</p><dl><dt>训练来源</dt><dd>{model.dataset_name}</dd><dt>拟合范围</dt><dd>#{model.fit_start}–#{model.fit_end - 1}</dd><dt>校准范围</dt><dd>#{model.fit_end}–#{model.calibration_end - 1}</dd><dt>固定阈值</dt><dd>{model.threshold.toPrecision(5)}（{(model.training_config.threshold_quantile * 100).toFixed(2)}% 校准分位数）</dd>{model.algorithm === 'pca' && <><dt>目标解释方差</dt><dd>{(model.training_config.pca_variance * 100).toFixed(1)}%</dd></>}{model.algorithm === 'temporal' && <><dt>固定历史窗口</dt><dd>{model.window} 样本</dd></>}<dt>变量契约</dt><dd>{model.features.join(' / ')}</dd></dl><p className="training-note">缺失值填补、中心与尺度、模型和阈值均来自此训练版本；新数据不参与拟合或校准。</p></div>}
      <div className="form-grid"><label className="field">最短事件（样本）<input type="number" name="min_event_length" aria-label="最短事件长度" min={1} max={50} step={1} value={minLength} onChange={event => setMinLength(event.target.value)} required disabled={busy} /></label><label className="field">合并间隔（样本）<input type="number" name="merge_gap" aria-label="事件合并间隔" min={0} max={20} step={1} value={mergeGap} onChange={event => setMergeGap(event.target.value)} required disabled={busy} /></label></div>
      {model?.algorithm === 'temporal' && <><label className="field">连续流 ID（可选）<input name="stream_id" aria-label="连续流 ID" maxLength={100} value={streamId} onChange={event => setStreamId(event.target.value)} placeholder="留空表示独立文件，每次重新预热" disabled={busy} /></label><p className="training-note">仅同一模型版本和同一流 ID 共享已成功检测的历史。时间必须递增；间断会重新预热。独立文件不借用训练尾段，不可评分点不代表正常。</p></>}
      <button className="button primary full-width" data-testid="run-button" disabled={busy || modelLoading || !!modelError || !model} type="submit">{busy ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}{busy ? '正在处理，请稍候' : '开始异常检测'}</button>
    </form>
    {!!models.length && <details className="quality-warning"><summary>模型可用性与变量兼容情况</summary><ul>{models.map(item => { const compatibility = modelCompatibility(item, dataset.features); return <li key={item.id}>{item.name} v{item.version}：{item.status === 'published' ? '已发布' : item.status === 'ready' ? '未发布' : '已停用'}；{compatibility.compatible ? '变量集合兼容（允许列重排）' : `缺少：${compatibility.missing.join('、') || '无'}；多余：${compatibility.unexpected.join('、') || '无'}`}</li>; })}</ul><button className="text-button" onClick={onTrain}>管理与训练模型</button></details>}
  </section>;
}
