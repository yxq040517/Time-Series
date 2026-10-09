import { useEffect, useRef, useState, type FormEvent } from 'react';
import { AlertCircle, Check, Database, LoaderCircle, Play, RefreshCw } from 'lucide-react';
import { api, isAbort, messageOf } from '../api';
import { algorithmDescriptions, algorithmNames, type Algorithm, type Dataset, type ModelInfo, type TrainingJob } from '../types';
import { timeLabel } from './AnalysisCharts';

const modelStatuses = { ready: '待发布', published: '已发布', disabled: '已停用' };
const trainingStatuses = { queued: '排队中', running: '训练中', completed: '已完成', failed: '失败' };

export function ModelManagement({ datasets, selectedDatasetId, models, modelLoading, modelError, onModelsChanged, onImport }: {
  datasets: Dataset[]; selectedDatasetId: string; models: ModelInfo[]; modelLoading: boolean; modelError: string;
  onModelsChanged: () => void; onImport: () => void;
}) {
  const [datasetId, setDatasetId] = useState(selectedDatasetId || datasets[0]?.id || '');
  const dataset = datasets.find(item => item.id === datasetId);
  const [name, setName] = useState('');
  const [algorithm, setAlgorithm] = useState<Algorithm>('pca');
  const [fitStart, setFitStart] = useState('0');
  const [fitEnd, setFitEnd] = useState('');
  const [calibrationEnd, setCalibrationEnd] = useState('');
  const [quantile, setQuantile] = useState('99');
  const [variance, setVariance] = useState('90');
  const [windowSize, setWindowSize] = useState('8');
  const [jobs, setJobs] = useState<TrainingJob[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [pollError, setPollError] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [operation, setOperation] = useState('');
  const [revision, setRevision] = useState(0);
  const [lastJobId, setLastJobId] = useState('');
  const lifetime = useRef<AbortController | null>(null);
  const pending = useRef(false);
  const knownCompleted = useRef(new Set<string>());
  useEffect(() => {
    const controller = new AbortController(); lifetime.current = controller;
    return () => { controller.abort(); lifetime.current = null; };
  }, []);
  useEffect(() => {
    if (!dataset && datasets[0]) setDatasetId(datasets[0].id);
  }, [dataset, datasets]);
  useEffect(() => {
    if (!dataset) return;
    setFitStart('0'); setFitEnd(String(Math.floor(dataset.rows * .8))); setCalibrationEnd(String(dataset.rows));
  }, [dataset]);
  useEffect(() => {
    const controller = new AbortController(); let timer: number | undefined;
    setLoading(true);
    const poll = async () => {
      try {
        const { items } = await api.trainings(controller.signal);
        if (controller.signal.aborted) return;
        setJobs(items); setLoading(false); setPollError('');
        const completed = items.filter(item => item.status === 'completed' && item.model_id);
        if (completed.some(item => !knownCompleted.current.has(item.id))) onModelsChanged();
        completed.forEach(item => knownCompleted.current.add(item.id));
        if (items.some(item => item.status === 'queued' || item.status === 'running')) timer = window.setTimeout(() => { void poll(); }, 1200);
      } catch (cause) {
        if (!isAbort(cause) && !controller.signal.aborted) { setPollError(messageOf(cause)); setLoading(false); timer = window.setTimeout(() => { void poll(); }, 4000); }
      }
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [revision, onModelsChanged]);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const controller = lifetime.current;
    if (!dataset || pending.current || !controller || controller.signal.aborted) return;
    if (!name.trim()) { setError('请输入非空模型名称。'); return; }
    const start = Number(fitStart), end = Number(fitEnd), calibration = Number(calibrationEnd);
    if (!(Number.isInteger(start) && Number.isInteger(end) && Number.isInteger(calibration) && 0 <= start && start < end && end < calibration && calibration <= dataset.rows)) {
      setError('范围必须满足 0 ≤ 拟合起点 < 拟合终点 < 校准终点 ≤ 数据行数，终点不包含在区间内。'); return;
    }
    pending.current = true; setSubmitting(true); setError('');
    try {
      const job = await api.createTraining({ dataset_id: dataset.id, name: name.trim(), algorithm, fit_start: start, fit_end: end,
        calibration_end: calibration, threshold_quantile: Number(quantile) / 100,
        pca_variance: algorithm === 'pca' ? Number(variance) / 100 : .9, window: algorithm === 'temporal' ? Number(windowSize) : 8 }, controller.signal);
      if (controller.signal.aborted) return;
      setJobs(items => [job, ...items.filter(item => item.id !== job.id)]); setLastJobId(job.id); setRevision(value => value + 1);
    } catch (cause) { if (!isAbort(cause) && !controller.signal.aborted) setError(messageOf(cause)); }
    finally { pending.current = false; if (!controller.signal.aborted) setSubmitting(false); }
  };
  const changeStatus = async (model: ModelInfo, publish: boolean) => {
    const controller = lifetime.current;
    if (!controller || controller.signal.aborted || pending.current) return;
    pending.current = true; setOperation(model.id); setError('');
    try {
      if (publish) await api.publishModel(model.id, controller.signal); else await api.disableModel(model.id, controller.signal);
      if (!controller.signal.aborted) onModelsChanged();
    } catch (cause) { if (!isAbort(cause) && !controller.signal.aborted) setError(messageOf(cause)); }
    finally { pending.current = false; if (!controller.signal.aborted) setOperation(''); }
  };
  return <div className="model-management" data-testid="model-management">
    {(error || pollError || modelError) && <div className="error-banner" role="alert"><AlertCircle size={18} /><span>{[error, pollError, modelError].filter(Boolean).join('；')}</span><button className="text-button" onClick={() => { setRevision(value => value + 1); onModelsChanged(); }}>刷新任务与模型</button></div>}
    <section className="card"><div className="card-heading"><div><h2>独立模型训练</h2><p>先在历史数据上拟合与校准，再发布不可变版本供新数据检测。不会上传任意模型文件。</p></div><span className="badge neutral">训练 ≠ 检测</span></div>
      {!datasets.length ? <div className="empty-state"><Database size={26} /><strong>先导入训练历史</strong><p>建议使用主要正常的历史区间；标签只用于提示，不会自动选择参数。</p><button className="button primary" onClick={onImport}>导入 CSV</button></div> : <form className="model-training-form" onSubmit={event => { void submit(event); }}>
        <div className="form-grid"><label className="field">训练数据集<select aria-label="训练数据集" data-testid="training-dataset" value={datasetId} onChange={event => setDatasetId(event.target.value)} disabled={submitting} required>{datasets.map(item => <option key={item.id} value={item.id}>{item.name} · {item.rows} 样本</option>)}</select></label><label className="field">模型名称<input name="name" aria-label="模型名称" data-testid="training-name" value={name} onChange={event => setName(event.target.value)} maxLength={200} required disabled={submitting} placeholder="同名训练自动生成下一版本" /></label></div>
        <label className="field">训练算法<select aria-label="训练算法" data-testid="training-algorithm" value={algorithm} onChange={event => setAlgorithm(event.target.value as Algorithm)} disabled={submitting}>{Object.entries(algorithmNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label><p className="method-description">{algorithmDescriptions[algorithm]}</p>
        <div className="training-range-grid"><label className="field">拟合起点（含）<input name="fit_start" data-testid="fit-start" aria-label="拟合起点" type="number" min={0} max={(dataset?.rows || 1) - 2} step={1} value={fitStart} onChange={event => setFitStart(event.target.value)} required disabled={submitting} /></label><label className="field">拟合终点 / 校准起点（不含）<input name="fit_end" data-testid="fit-end" aria-label="拟合终点" type="number" min={1} max={(dataset?.rows || 1) - 1} step={1} value={fitEnd} onChange={event => setFitEnd(event.target.value)} required disabled={submitting} /></label><label className="field">校准终点（不含）<input name="calibration_end" data-testid="calibration-end" aria-label="校准终点" type="number" min={2} max={dataset?.rows || 1} step={1} value={calibrationEnd} onChange={event => setCalibrationEnd(event.target.value)} required disabled={submitting} /></label></div>
        <div className="form-grid"><label className="field">阈值分位数（%）<input name="threshold_quantile" aria-label="阈值分位数" type="number" min={90} max={99.99} step={.01} value={quantile} onChange={event => setQuantile(event.target.value)} required disabled={submitting} /></label>{algorithm === 'pca' && <label className="field">目标解释方差（%）<input name="pca_variance" aria-label="目标解释方差" type="number" min={50} max={99} step={1} value={variance} onChange={event => setVariance(event.target.value)} required disabled={submitting} /></label>}{algorithm === 'temporal' && <label className="field">历史窗口（样本）<input name="window" aria-label="历史窗口" type="number" min={2} max={32} step={1} value={windowSize} onChange={event => setWindowSize(event.target.value)} required disabled={submitting} /></label>}</div>
        <div className="notice"><AlertCircle size={17} /><span>拟合区间应主要正常；校准区间若受异常污染会改变固定阈值。{dataset?.has_labels ? '当前数据含标签，服务端会提示污染风险，但不使用标签自动调参。' : '当前数据没有标签，请结合业务背景确认历史质量。'}范围外数据不参与训练；服务端还会校验算法所需样本数。</span></div>
        <button className="button primary" data-testid="training-submit" type="submit" disabled={submitting || !!operation || !dataset}>{submitting ? <LoaderCircle size={17} className="spin" /> : <Play size={17} />}{submitting ? '正在提交训练…' : '开始独立训练'}</button>
      </form>}
    </section>
    <section className="card"><div className="card-heading"><div><h2>真实训练任务</h2><p>切换页面不会取消服务端任务；中断与错误均保留在任务记录中。</p></div><button className="text-button" onClick={() => setRevision(value => value + 1)}><RefreshCw size={15} />刷新</button></div>{loading && <p role="status">正在读取训练任务…</p>}{!loading && !jobs.length && <p className="muted">尚无训练记录。使用上方历史数据创建模型。</p>}<div className="training-jobs">{jobs.map(job => <article key={job.id} className={`training-job ${job.id === lastJobId ? 'latest' : ''}`} data-testid={`training-job-${job.id}`}><div className="progress-title"><div><strong>{job.config.name} · {algorithmNames[job.config.algorithm]}</strong><p>{job.dataset_name} · {timeLabel(job.created_at)} · {trainingStatuses[job.status]}</p></div><strong>{job.progress}%</strong></div><progress value={job.progress} max={100} aria-label={`${job.config.name} 服务端训练进度`} /><p role={job.status === 'failed' ? 'alert' : 'status'}>{job.status === 'failed' ? job.error || job.message : job.message}</p>{job.status === 'completed' && <p className="training-complete"><Check size={15} />模型版本已持久化；请在下方发布后用于检测。<span className="muted">{job.model_id}</span></p>}</article>)}</div></section>
    <section className="card"><div className="card-heading"><div><h2>持久化模型版本<span className="heading-count">{models.length}</span></h2><p>模型参数和变量契约不可变；发布与停用只改变可选状态，保留已有检测历史。</p></div><button className="text-button" onClick={onModelsChanged}><RefreshCw size={15} />刷新模型</button></div>{modelLoading && <p role="status">正在读取模型…</p>}{!modelLoading && !models.length && <div className="empty-state"><Database size={25} /><strong>尚无持久化模型</strong><p>完成独立训练后模型会列在这里；发布前不能用于工作台检测。</p></div>}<div className="model-version-list">{models.map(model => <article className="model-version" key={model.id} data-testid={`model-row-${model.id}`}><div className="card-heading"><div><h3>{model.name} · v{model.version}</h3><p>{algorithmNames[model.algorithm]} · {timeLabel(model.created_at)}</p></div><span className={`badge ${model.status === 'published' ? 'teal' : 'neutral'}`}>{modelStatuses[model.status]}</span></div><dl className="model-facts"><dt>训练来源</dt><dd>{model.dataset_name}（{model.dataset_id}）</dd><dt>拟合 / 校准</dt><dd>#{model.fit_start}–#{model.fit_end - 1} / #{model.fit_end}–#{model.calibration_end - 1}</dd><dt>变量（训练顺序）</dt><dd>{model.features.join(' / ')}</dd><dt>固定阈值</dt><dd>{model.threshold.toPrecision(6)} · {(model.training_config.threshold_quantile * 100).toFixed(2)}% 分位数</dd><dt>固定算法参数</dt><dd>{model.algorithm === 'pca' ? `目标解释方差 ${(model.training_config.pca_variance * 100).toFixed(1)}%` : model.algorithm === 'temporal' ? `过去 ${model.window} 样本窗口` : 'IsolationForest · 稳健偏离解释'}</dd><dt>版本来源</dt><dd>模型 {model.id} · 训练 {model.training_id}</dd><dt>格式 / 运行库</dt><dd>格式 {model.format_version} · scikit-learn {model.sklearn_version}</dd></dl>{!!Object.keys(model.training_summary).length && <details className="preview-details"><summary>训练统计</summary><dl className="model-facts">{Object.entries(model.training_summary).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{typeof value === 'object' ? JSON.stringify(value) : String(value)}</dd></div>)}</dl></details>}{!!model.notes.length && <ul className="model-notes">{model.notes.map((note, index) => <li key={index}>{note}</li>)}</ul>}<div className="model-actions">{model.status !== 'published' && <button className="button primary compact" data-testid={`publish-model-${model.id}`} onClick={() => { void changeStatus(model, true); }} disabled={submitting || !!operation}>{operation === model.id ? <LoaderCircle size={14} className="spin" /> : <Check size={14} />}发布模型</button>}{model.status !== 'disabled' && <button className="button secondary compact" data-testid={`disable-model-${model.id}`} onClick={() => { void changeStatus(model, false); }} disabled={submitting || !!operation}>停用模型</button>}</div></article>)}</div></section>
  </div>;
}
