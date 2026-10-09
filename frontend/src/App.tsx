import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { Activity, AlertCircle, ArrowDownToLine, ArrowRight, BarChart3, Check, ChevronLeft, ChevronRight, CircleHelp, Database, FileText, FlaskConical, GitCompareArrows, History, Layers3, LoaderCircle, Microscope, RefreshCw, RotateCcw, Search, ShieldCheck, SlidersHorizontal, Upload, X } from 'lucide-react';
import { api, download, isAbort, messageOf } from './api';
import { algorithmNames, reusableDetectionConfig, runModelLabel, type AnomalyEvent, type Dataset, type DetectionConfig, type Explanation, type Heatmap, type ModelInfo, type Preview, type ReviewStatus, type Run, type SampleRange, type Series } from './types';
import { ContributionChart, ExplanationChart, FeatureChart, ScoreChart, timeLabel } from './components/AnalysisCharts';
import { ImportDialog } from './components/ImportDialog';
import { RunConfiguration } from './components/RunConfiguration';
import { ModelManagement } from './components/ModelManagement';
import { DatasetProfile } from './components/DatasetProfile';
import { ComparisonPage } from './components/ComparisonPage';
import { EventExplorer } from './components/EventExplorer';
import { InvestigationSummary } from './components/InvestigationSummary';

const counts = new Intl.NumberFormat('zh-CN');
const reviewNames: Record<ReviewStatus, string> = { unreviewed: '待复核', confirmed: '确认异常', false_positive: '误报' };
const statusNames: Record<Run['status'], string> = { queued: '排队中', running: '检测中', completed: '已完成', failed: '失败' };
function Loading({ text = '正在读取分析数据…' }: { text?: string }) { return <div className="loading-state" role="status"><LoaderCircle size={22} className="spin" /><span>{text}</span></div>; }
function Empty({ title, children }: { title: string; children: ReactNode }) { return <div className="empty-state"><Search size={26} /><strong>{title}</strong><p>{children}</p></div>; }

export default function App() {
  const [page, setPage] = useState<'workspace' | 'datasets' | 'history' | 'profile' | 'comparison' | 'models'>('workspace');
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelLoading, setModelLoading] = useState(true);
  const [modelError, setModelError] = useState('');
  const [modelRevision, setModelRevision] = useState(0);
  const refreshModels = useCallback(() => setModelRevision(value => value + 1), []);
  const [datasetId, setDatasetId] = useState('');
  const [initialLoading, setInitialLoading] = useState(true);
  const [preview, setPreview] = useState<Preview | null>(null);
  const [history, setHistory] = useState<Run[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [runId, setRunId] = useState('');
  const [run, setRun] = useState<Run | null>(null);
  const [overview, setOverview] = useState<Series | null>(null);
  const [events, setEvents] = useState<AnomalyEvent[]>([]);
  const [resultLoading, setResultLoading] = useState(false);
  const [resultError, setResultError] = useState('');
  const [range, setRange] = useState<SampleRange>({ start: 0, end: 1 });
  const [feature, setFeature] = useState('');
  const [detailSeries, setDetailSeries] = useState<Series | null>(null);
  const [heatmap, setHeatmap] = useState<Heatmap | null>(null);
  const [seriesLoading, setSeriesLoading] = useState(false);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [detailError, setDetailError] = useState('');
  const [eventId, setEventId] = useState('');
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);
  const [explanationError, setExplanationError] = useState('');
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus>('unreviewed');
  const [reviewNote, setReviewNote] = useState('');
  const [reviewSaving, setReviewSaving] = useState(false);
  const [batchSaving, setBatchSaving] = useState(false);
  const [reviewSaved, setReviewSaved] = useState(false);
  const [error, setError] = useState('');
  const [importOpen, setImportOpen] = useState(false);
  const [demoBusy, setDemoBusy] = useState(false);
  const [creatingRun, setCreatingRun] = useState(false);
  const [downloading, setDownloading] = useState('');
  const [reload, setReload] = useState(0);
  const [setupOpen, setSetupOpen] = useState(false);
  const [initialConfig, setInitialConfig] = useState<DetectionConfig | undefined>();
  const [configRevision, setConfigRevision] = useState(0);
  const [librarySearch, setLibrarySearch] = useState('');
  const [historyAlgorithm, setHistoryAlgorithm] = useState('all');
  const [historyStatus, setHistoryStatus] = useState('all');
  const selection = useRef({ datasetId, runId, eventId });
  const navigation = useRef<HTMLElement>(null);
  const runCreationRevision = useRef(0);
  const reviewOperation = useRef(false);
  const reviewRevision = useRef(0);
  const reviewedEvents = useRef(new Map<string, AnomalyEvent>());
  selection.current = { datasetId, runId, eventId };
  const dataset = datasets.find((item) => item.id === datasetId);
  const completedId = run?.status === 'completed' ? run.id : '';
  const executing = creatingRun || run?.status === 'queued' || run?.status === 'running';
  const selectedEvent = events.find((item) => item.id === eventId);

  const selectRun = useCallback((id: string, rows: number) => {
    if (id && selection.current.runId === id) return;
    setRunId(id); setRun(null); setOverview(null); setEvents([]); setEventId(''); setExplanation(null);
    reviewedEvents.current.clear(); reviewRevision.current += 1;
    setDetailSeries(null); setHeatmap(null); setRange({ start: 0, end: rows }); setResultError(''); setDetailError(''); setReviewSaved(false); setSetupOpen(false);
  }, []);
  const selectDataset = useCallback((item: Dataset) => {
    if (selection.current.datasetId === item.id) return;
    runCreationRevision.current += 1;
    setDatasetId(item.id); setPreview(null); setHistory([]); setFeature(item.features[0] || '');
    selectRun('', item.rows); setError(''); setCreatingRun(false); setInitialConfig(undefined);
  }, [selectRun]);

  useEffect(() => {
    const nav = navigation.current;
    if (!nav) return;
    const revealActive = () => {
      const active = nav.querySelector<HTMLElement>('.nav-item.active');
      if (active && nav.scrollWidth > nav.clientWidth) {
        nav.scrollLeft += active.getBoundingClientRect().left - nav.getBoundingClientRect().left - (nav.clientWidth - active.offsetWidth) / 2;
      }
    };
    revealActive();
    const observer = new ResizeObserver(revealActive);
    observer.observe(nav);
    return () => observer.disconnect();
  }, [page]);

  useEffect(() => {
    const controller = new AbortController();
    setInitialLoading(true);
    api.datasets(controller.signal).then(({ items }) => {
      if (controller.signal.aborted) return;
      setDatasets(items);
      if (!selection.current.datasetId && items[0]) selectDataset(items[0]);
      setError('');
    }).catch((cause: unknown) => { if (!isAbort(cause)) setError(messageOf(cause)); }).finally(() => { if (!controller.signal.aborted) setInitialLoading(false); });
    return () => controller.abort();
  }, [reload, selectDataset]);
  useEffect(() => {
    const controller = new AbortController();
    setModelLoading(true); setModelError('');
    api.models(controller.signal).then(({ items }) => { if (!controller.signal.aborted) setModels(items); })
      .catch((cause: unknown) => { if (!isAbort(cause) && !controller.signal.aborted) setModelError(messageOf(cause)); })
      .finally(() => { if (!controller.signal.aborted) setModelLoading(false); });
    return () => controller.abort();
  }, [modelRevision, reload]);

  useEffect(() => {
    if (!datasetId) return;
    const controller = new AbortController();
    setHistoryLoading(true);
    api.preview(datasetId, controller.signal).then((value) => { if (!controller.signal.aborted && selection.current.datasetId === datasetId) setPreview(value); }).catch((cause: unknown) => { if (!isAbort(cause)) setError(messageOf(cause)); });
    api.runs(datasetId, controller.signal).then(({ items }) => {
      if (controller.signal.aborted || selection.current.datasetId !== datasetId) return;
      setHistory(items);
      if (!selection.current.runId && items[0]) setRunId(items[0].id);
    }).catch((cause: unknown) => { if (!isAbort(cause)) setError(messageOf(cause)); }).finally(() => { if (!controller.signal.aborted) setHistoryLoading(false); });
    return () => controller.abort();
  }, [datasetId, reload]);

  useEffect(() => {
    if (!runId) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const current = await api.run(runId, controller.signal);
        if (controller.signal.aborted) return;
        setRun(current);
        setHistory((items) => [current, ...items.filter((item) => item.id !== current.id)].sort((a, b) => b.created_at.localeCompare(a.created_at)));
        if (current.status === 'queued' || current.status === 'running') timer = window.setTimeout(() => { void poll(); }, 1200);
      } catch (cause) {
        if (!isAbort(cause)) {
          setError(messageOf(cause));
          timer = window.setTimeout(() => { void poll(); }, 4000);
        }
      }
    };
    void poll();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [runId, reload]);

  useEffect(() => {
    if (!completedId || !dataset) return;
    const controller = new AbortController();
    setResultLoading(true); setResultError('');
    Promise.all([api.series(completedId, { start: 0, end: dataset.rows }, dataset.features.slice(0, 1), controller.signal), api.events(completedId, controller.signal)])
      .then(([series, result]) => { if (!controller.signal.aborted && selection.current.runId === completedId) { setOverview(series); setEvents(result.items); } })
      .catch((cause: unknown) => { if (!isAbort(cause)) setResultError(messageOf(cause)); })
      .finally(() => { if (!controller.signal.aborted) setResultLoading(false); });
    return () => controller.abort();
  }, [completedId, dataset, reload]);

  useEffect(() => {
    if (!completedId || !feature) return;
    const controller = new AbortController();
    setSeriesLoading(true); setDetailError(''); setDetailSeries(null);
    api.series(completedId, range, [feature], controller.signal).then((value) => { if (!controller.signal.aborted && selection.current.runId === completedId) setDetailSeries(value); })
      .catch((cause: unknown) => { if (!isAbort(cause)) setDetailError(messageOf(cause)); })
      .finally(() => { if (!controller.signal.aborted) setSeriesLoading(false); });
    return () => controller.abort();
  }, [completedId, range, feature, reload]);

  useEffect(() => {
    if (!completedId) return;
    const controller = new AbortController();
    setHeatmapLoading(true); setHeatmap(null);
    api.heatmap(completedId, range, controller.signal).then((value) => { if (!controller.signal.aborted && selection.current.runId === completedId) setHeatmap(value); })
      .catch((cause: unknown) => { if (!isAbort(cause)) setDetailError(messageOf(cause)); })
      .finally(() => { if (!controller.signal.aborted) setHeatmapLoading(false); });
    return () => controller.abort();
  }, [completedId, range, reload]);

  useEffect(() => {
    if (!completedId || !eventId) return;
    const controller = new AbortController();
    setExplanationLoading(true); setExplanationError(''); setExplanation(null); setReviewSaved(false);
    const requestRevision = reviewRevision.current;
    api.explanation(completedId, eventId, controller.signal).then((value) => {
      if (controller.signal.aborted || selection.current.runId !== completedId || selection.current.eventId !== eventId) return;
      const currentEvent = requestRevision !== reviewRevision.current ? reviewedEvents.current.get(eventId) || value.event : value.event;
      setExplanation({ ...value, event: currentEvent }); setRange(value.context); setFeature(value.top_features[0]?.name || currentEvent.top_feature);
      setReviewStatus(currentEvent.status); setReviewNote(currentEvent.note);
    }).catch((cause: unknown) => { if (!isAbort(cause)) setExplanationError(messageOf(cause)); })
      .finally(() => { if (!controller.signal.aborted) setExplanationLoading(false); });
    return () => controller.abort();
  }, [completedId, eventId, reload]);

  const acceptDataset = (item: Dataset) => {
    setDatasets((items) => [item, ...items.filter((existing) => existing.id !== item.id)]);
    selectDataset(item); setImportOpen(false); setPage('workspace');
  };
  const generateDemo = async () => {
    setDemoBusy(true); setError('');
    try { acceptDataset(await api.demo()); } catch (cause) { setError(messageOf(cause)); } finally { setDemoBusy(false); }
  };
  const startRun = async (config: DetectionConfig) => {
    if (demoBusy || importOpen || executing || selection.current.datasetId !== config.dataset_id) return;
    const requestRevision = ++runCreationRevision.current;
    setCreatingRun(true); setError('');
    try {
      const current = await api.createRun(config);
      if (selection.current.datasetId !== config.dataset_id || requestRevision !== runCreationRevision.current) return;
      selectRun(current.id, dataset?.rows || 1); setRun(current);
      setHistory((items) => [current, ...items]); setPage('workspace');
    } catch (cause) { if (selection.current.datasetId === config.dataset_id && requestRevision === runCreationRevision.current) setError(messageOf(cause)); }
    finally { if (selection.current.datasetId === config.dataset_id && requestRevision === runCreationRevision.current) setCreatingRun(false); }
  };
  const saveReview = async () => {
    if (!completedId || !eventId || reviewOperation.current) return;
    reviewOperation.current = true;
    const targetRun = completedId; const targetEvent = eventId;
    setReviewSaving(true); setExplanationError(''); setReviewSaved(false);
    try {
      const updated = await api.review(targetRun, targetEvent, reviewStatus, reviewNote);
      if (selection.current.runId !== targetRun || selection.current.eventId !== targetEvent) return;
      reviewedEvents.current.set(updated.id, updated); reviewRevision.current += 1;
      setEvents((items) => items.map((item) => item.id === updated.id ? updated : item));
      setExplanation((value) => value ? { ...value, event: updated } : value); setReviewSaved(true);
    } catch (cause) { if (selection.current.runId === targetRun && selection.current.eventId === targetEvent) setExplanationError(messageOf(cause)); }
    finally { reviewOperation.current = false; setReviewSaving(false); }
  };
  const exportFile = async (kind: 'csv' | 'report' | 'source') => {
    const path = kind === 'source' ? `/datasets/${encodeURIComponent(datasetId)}/export.csv` : `/runs/${encodeURIComponent(completedId)}/${kind === 'csv' ? 'export.csv' : 'report.html'}`;
    setDownloading(kind); setError('');
    try { await download(path, kind === 'report' ? 'ChronoLens分析报告.html' : 'ChronoLens数据.csv'); }
    catch (cause) { setError(messageOf(cause)); } finally { setDownloading(''); }
  };
  const changeRange = useCallback((next: SampleRange) => setRange((current) => current.start === next.start && current.end === next.end ? current : next), []);
  const batchReview = async (ids: string[], status: ReviewStatus) => {
    const targetRun = completedId;
    if (!targetRun) throw new Error('请先选择已完成的检测任务。');
    if (reviewOperation.current) throw new Error('正在保存复核，请等待当前保存完成。');
    reviewOperation.current = true; setBatchSaving(true);
    try {
      const { items } = await api.batchReview(targetRun, ids, status);
      if (selection.current.runId !== targetRun) return;
      const changes = new Map(items.map(item => [item.id, item]));
      items.forEach(item => reviewedEvents.current.set(item.id, item)); reviewRevision.current += 1;
      setEvents(previous => previous.map(item => changes.get(item.id) || item));
      setExplanation(previous => previous && changes.has(previous.event.id) ? { ...previous, event: changes.get(previous.event.id)! } : previous);
      if (changes.has(selection.current.eventId)) { setReviewStatus(status); setReviewSaved(false); }
    } finally { reviewOperation.current = false; setBatchSaving(false); }
  };
  const reuseConfiguration = (item: Run) => {
    if (!dataset) return;
    const config = reusableDetectionConfig(item, models, dataset);
    if (!config) return;
    setInitialConfig(config); setConfigRevision(value => value + 1); setSetupOpen(true); setPage('workspace');
    window.setTimeout(() => document.getElementById('configuration')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80);
  };
  const pageNames = { workspace: '分析工作台', datasets: '数据集管理', history: '检测历史', profile: '数据诊断', comparison: '检测任务对比', models: '模型训练与管理' };
  const descriptions = { workspace: '使用已发布模型检测新数据，连接信号、解释与每一次判断。', datasets: '从原始数据开始，为每份分析建立可靠的基础。', history: '追溯模型版本与检测结果，让分析可重复、判断可复核。', profile: '了解变量分布、采样规律与数据质量，再开始建模。', comparison: '在同一份数据上，比较不同模型版本与事件选项的实际检测表现。', models: '独立训练、持久化版本、发布与停用；推理始终使用固定参数。' };
  const libraryItems = datasets.filter(item => `${item.name} ${item.description} ${item.features.join(' ')}`.toLowerCase().includes(librarySearch.toLowerCase()));
  const historyItems = history.filter(item => (historyAlgorithm === 'all' || item.algorithm === historyAlgorithm) && (historyStatus === 'all' || item.status === historyStatus));
  const summary = run?.summary;

  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#" onClick={(event) => { event.preventDefault(); setPage('workspace'); }} aria-label="ChronoLens 首页"><span className="brand-mark"><Activity size={23} /></span><span>ChronoLens<small>时序异常检测与解释</small></span></a>
      <div className="sidebar-caption">WORKSPACE / 分析空间</div><nav ref={navigation} aria-label="主导航">
        <button className={page === 'workspace' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('workspace')}><BarChart3 size={18} />分析工作台<ChevronRight size={14} /></button>
        <button aria-label="数据集管理" className={page === 'datasets' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('datasets')}><Database size={18} />数据集管理<span className="nav-count" aria-hidden="true">{datasets.length}</span></button>
        <button className={page === 'profile' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('profile')}><Microscope size={18} />数据诊断</button>
        <button className={page === 'models' ? 'nav-item active' : 'nav-item'} data-testid="models-nav" onClick={() => setPage('models')}><Layers3 size={18} />模型训练与管理</button>
        <button className={page === 'comparison' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('comparison')}><GitCompareArrows size={18} />检测任务对比</button>
        <button className={page === 'history' ? 'nav-item active' : 'nav-item'} onClick={() => setPage('history')}><History size={18} />检测历史</button>
      </nav><div className="sidebar-current"><span className="sidebar-caption">当前数据集</span>{dataset ? <><strong>{dataset.name}</strong><span>{counts.format(dataset.rows)} 样本 · {dataset.features.length} 变量</span><span className={dataset.source === 'demo' ? 'source-tag demo' : 'source-tag'}>{dataset.source === 'demo' ? '合成演示 · 非真实生产数据' : '本地 CSV 数据'}</span></> : <p>尚未选择数据集</p>}</div>
      <div className="sidebar-methods"><span className="sidebar-caption">三种方法，一份证据</span><span><i />PCA 多变量重构</span><span><i />时序自回归预测</span><span><i />IsolationForest 稀有检测</span></div><div className="sidebar-bottom"><ShieldCheck size={19} /><div><strong>本地分析，数据不出设备</strong><span>ChronoLens / Research workspace</span></div></div>
    </aside>
    <div className="main-shell"><header className="topbar"><div className="breadcrumb"><span>研究工作空间</span><ChevronRight size={14} /><strong>{pageNames[page]}</strong></div><div className="top-actions"><button className="button secondary compact" data-testid="demo-button" onClick={() => { void generateDemo(); }} disabled={demoBusy}>{demoBusy ? <LoaderCircle size={15} className="spin" /> : <FlaskConical size={15} />}<span>{demoBusy ? '生成中' : '生成演示'}</span></button><button className="button primary compact" data-testid="import-button" onClick={() => setImportOpen(true)}><Upload size={15} /><span>导入 CSV</span></button></div></header>
      <main className="content"><div className="page-heading"><div><span className="eyebrow">TIME SERIES INTELLIGENCE</span><h1>{pageNames[page]}<span className="heading-dot">.</span></h1><p>{descriptions[page]}</p></div><div className="workspace-label"><span className="status-dot" />本地工作空间<span className="workspace-version">LOCAL</span></div></div>
      {error && <div className="error-banner" role="alert"><AlertCircle size={18} /><span>{error}</span><button className="text-button" onClick={() => setReload((value) => value + 1)}>重试</button><button className="icon-button" aria-label="关闭错误提示" onClick={() => setError('')}><X size={15} /></button></div>}
      {initialLoading && <Loading text="正在连接本地数据工作台…" />}
      {!initialLoading && !datasets.length && <section className="welcome card"><div className="welcome-graphic"><Activity size={55} /><div className="signal-line" /></div><span className="eyebrow">从一段时序开始</span><h2>连接数据，建立你的第一份分析</h2><p>导入真实 CSV，或生成带有已知异常标签的合成演示数据。<br />检测结果、变量解释与人工复核都来自实际计算，不预填任何结论。</p><div className="welcome-actions"><button className="button primary" onClick={() => setImportOpen(true)}><Upload size={17} />导入我的数据</button><button className="button secondary" onClick={() => { void generateDemo(); }} disabled={demoBusy}><FlaskConical size={17} />体验合成演示<ArrowRight size={15} /></button></div><div className="welcome-steps"><span><b>01</b>导入与质量校验</span><span><b>02</b>独立训练与发布</span><span><b>03</b>解释、复核与导出</span></div></section>}
      {!!datasets.length && <div className="dataset-toolbar"><div className="dataset-selector"><Database size={18} /><label htmlFor="dataset-select">当前数据集</label><select id="dataset-select" data-testid="dataset-select" value={datasetId} onChange={(event) => { const item = datasets.find((entry) => entry.id === event.target.value); if (item) selectDataset(item); }}><option value="" disabled>请选择数据集</option>{datasets.map((item) => <option key={item.id} value={item.id}>{item.name}{item.source === 'demo' ? ' · 合成演示' : ''}</option>)}</select></div>{dataset && <div className="dataset-toolbar-meta"><span>{counts.format(dataset.rows)} 样本</span><span>{dataset.features.length} 变量</span><span className={`badge ${dataset.source === 'demo' ? 'violet' : 'teal'}`}>{dataset.source === 'demo' ? '合成数据' : 'CSV 导入'}</span></div>}</div>}
      {page === 'datasets' && <><div className="library-summary"><div><Database size={24} /><span>数据集总数<strong>{datasets.length}</strong></span></div><div><Layers3 size={24} /><span>累计样本<strong>{counts.format(datasets.reduce((sum, item) => sum + item.rows, 0))}</strong></span></div><div><ShieldCheck size={24} /><span>存储方式<strong className="fact-text">本地持久化</strong></span></div></div><section className="card"><div className="card-heading"><div><span className="section-kicker">DATA LIBRARY</span><h2>你的数据集</h2><p>查看数据规模、质量和来源，直接进入分析或诊断。</p></div><label className="search-field library-search"><Search size={16} /><input aria-label="搜索数据集" placeholder="搜索数据集或变量" value={librarySearch} onChange={event => setLibrarySearch(event.target.value)} /></label></div><div className="table-scroll"><table><thead><tr><th>名称 / 来源</th><th>数据规模</th><th>质量提示</th><th>导入时间</th><th>操作</th></tr></thead><tbody>{libraryItems.map(item => <tr key={item.id}><td><strong>{item.name}</strong><small>{item.source === 'demo' ? '合成演示' : '本地 CSV'} · {item.features.slice(0, 3).join(' / ')}</small></td><td>{counts.format(item.rows)} 行<small>{item.features.length} 个变量</small></td><td><span className={`badge ${item.quality.warnings.length ? 'warm' : 'teal'}`}>{item.quality.warnings.length ? `${item.quality.warnings.length} 条提示` : '解析通过'}</span></td><td>{timeLabel(item.created_at)}</td><td><div className="table-actions"><button className="text-button" onClick={() => { selectDataset(item); setPage('workspace'); }}>分析<ArrowRight size={13} /></button><button className="text-button" onClick={() => { selectDataset(item); setPage('profile'); }}>诊断</button></div></td></tr>)}</tbody></table>{!libraryItems.length && <Empty title="没有匹配的数据集">调整搜索条件，或导入新的时序数据。</Empty>}</div></section></>}
      {page === 'history' && dataset && <section className="card"><div className="card-heading"><div><span className="section-kicker">DETECTION LOG</span><h2>检测记录<span className="heading-count">{history.length}</span></h2><p>{dataset.name} · 模型版本、事件配置与结果保存在本地；旧版无独立模型的记录只读。</p></div><button className="text-button" onClick={() => setReload(value => value + 1)}><RefreshCw size={15} />刷新</button></div>
        <div className="history-toolbar"><select aria-label="筛选检测算法" value={historyAlgorithm} onChange={event => setHistoryAlgorithm(event.target.value)}><option value="all">全部算法</option>{Object.entries(algorithmNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><select aria-label="筛选任务状态" value={historyStatus} onChange={event => setHistoryStatus(event.target.value)}><option value="all">全部任务状态</option>{Object.entries(statusNames).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select><button className="text-button" onClick={() => setPage('comparison')}><GitCompareArrows size={16} />比较检测结果</button></div>
        {historyLoading ? <Loading /> : historyItems.length ? <div className="table-scroll"><table><thead><tr><th>模型版本 / 方法</th><th>创建时间</th><th>状态</th><th>异常事件</th><th>可评分 / 预热</th><th>异常比例</th><th>操作</th></tr></thead><tbody>{historyItems.map(item => <tr key={item.id}><td><strong>{runModelLabel(item)}</strong><small>{algorithmNames[item.algorithm]}</small></td><td>{timeLabel(item.created_at)}</td><td><span className={`badge ${item.status === 'completed' ? 'teal' : item.status === 'failed' ? 'danger' : 'warm'}`}>{statusNames[item.status]}</span></td><td>{item.summary?.event_count ?? '—'}</td><td>{item.summary ? `${item.summary.scored_points} / ${item.summary.warmup_points}` : '—'}</td><td>{item.summary?.scored_points ? `${(item.summary.anomaly_ratio * 100).toFixed(2)}%` : '—'}</td><td><div className="table-actions"><button className="text-button" onClick={() => { selectRun(item.id, dataset.rows); setPage('workspace'); }}>打开<ArrowRight size={13} /></button><button className="text-button" disabled={modelLoading || !!modelError || !reusableDetectionConfig(item, models, dataset)} title="仅可复用仍已发布、兼容的独立模型检测配置" onClick={() => reuseConfiguration(item)}>复用配置</button></div></td></tr>)}</tbody></table></div> : <Empty title={history.length ? '没有符合条件的检测任务' : '还没有检测记录'}>调整筛选条件，或选择已发布模型发起检测；无模型时先完成独立训练。</Empty>}
      </section>}
      {page === 'profile' && dataset && <DatasetProfile key={datasetId} dataset={dataset} />}
      {page === 'models' && <ModelManagement datasets={datasets} selectedDatasetId={datasetId} models={models} modelLoading={modelLoading} modelError={modelError} onModelsChanged={refreshModels} onImport={() => setImportOpen(true)} />}
      {page === 'comparison' && dataset && <ComparisonPage key={datasetId} dataset={dataset} runs={history} onOpenRun={item => { selectRun(item.id, dataset.rows); setPage('workspace'); }} />}
      {page === 'workspace' && dataset && <>
        <div className="notice trust-notice"><CircleHelp size={17} /><span><strong>解释不等于因果。</strong>变量贡献描述与异常同步出现的偏离；训练段假设主要正常，请结合业务背景复核。{dataset.source === 'demo' && ' 当前数据为合成演示，不代表真实生产表现。'}</span></div>
        <details className="setup-panel" open={!completedId || setupOpen} onToggle={event => { if (completedId) setSetupOpen(event.currentTarget.open); }}><summary><span><SlidersHorizontal size={18} /><strong>数据与检测配置</strong><small>{completedId ? "展开查看数据或开始新的检测" : "准备数据，选择已发布模型"}</small></span><ChevronRight size={17} /></summary><div className="setup-grid"><section className="card dataset-card"><div className="card-heading"><div className="section-title"><Database size={18} /><h2>数据概览</h2></div><button className="text-button" onClick={() => { void exportFile('source'); }} disabled={!!downloading}><ArrowDownToLine size={14} />源数据</button></div><h3>{dataset.name}</h3><p className="muted dataset-description">{dataset.description || '已导入本地时序数据。'}</p><div className="dataset-facts"><div><span>样本数</span><strong>{counts.format(dataset.rows)}</strong></div><div><span>数值变量</span><strong>{dataset.features.length}</strong></div><div><span>缺失比例</span><strong>{(dataset.quality.missing_ratio * 100).toFixed(2)}<small>%</small></strong></div><div><span>评估标签</span><strong className="fact-text">{dataset.has_labels ? '已提供' : '未提供'}</strong></div></div><div className="time-extent">{timeLabel(dataset.start_time)}<ArrowRight size={13} />{timeLabel(dataset.end_time)}</div>
          {dataset.quality.warnings.length > 0 && <details className="quality-warning"><summary><AlertCircle size={14} />数据质量提示（{dataset.quality.warnings.length}）</summary><ul>{dataset.quality.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul></details>}
          <details className="preview-details"><summary><Layers3 size={15} />查看前 8 行数据<span className="muted">只读预览</span></summary>{preview ? <div className="table-scroll preview-table"><table><thead><tr>{preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead><tbody>{preview.rows.map((row, index) => <tr key={index}>{preview.columns.map((column) => <td key={column}>{row[column] === null ? <span className="muted">缺失</span> : typeof row[column] === 'number' ? Number(row[column]).toLocaleString('zh-CN', { maximumFractionDigits: 4 }) : String(row[column] ?? '')}</td>)}</tr>)}</tbody></table></div> : <Loading text="正在读取预览…" />}</details>
        </section><RunConfiguration key={`${datasetId}-${configRevision}`} initialConfig={initialConfig} dataset={dataset} models={models} modelLoading={modelLoading} modelError={modelError} onRefresh={refreshModels} onTrain={() => setPage('models')} busy={!!executing || demoBusy || importOpen} onRun={(config) => { void startRun(config); }} /></div></details>
        {!!history.length && <div className="run-selector"><History size={15} /><label htmlFor="run-select">检测记录</label><select id="run-select" value={runId} onChange={(event) => selectRun(event.target.value, dataset.rows)}><option value="">选择一条检测记录</option>{history.map((item) => <option value={item.id} key={item.id}>{runModelLabel(item)} · {algorithmNames[item.algorithm]} · {timeLabel(item.created_at)} · {statusNames[item.status]}</option>)}</select><button className="text-button" onClick={() => setPage('history')}>全部记录<ArrowRight size={13} /></button></div>}
        {!runId && !creatingRun && <section className="card ready-card"><Activity size={24} /><div><h3>数据已就绪，选择已发布模型开始检测</h3><p>检测不会训练或重新校准模型。尚无兼容模型时，请先在「模型训练与管理」完成独立训练与发布。</p><button className="text-button" onClick={() => setPage('models')}>前往模型训练</button></div></section>}
        {(creatingRun || (runId && !run)) && <section className="card"><Loading text={creatingRun ? '正在提交检测任务…' : '正在读取检测记录…'} /></section>}
        {run && (run.status === 'running' || run.status === 'queued') && <section className="card progress-card" role="status"><div className="progress-title"><LoaderCircle className="spin" size={21} /><div><h3>{algorithmNames[run.algorithm]} · {statusNames[run.status]}</h3><p>{run.message}</p></div><strong>{run.progress}%</strong></div><progress value={run.progress} max={100} aria-label="服务端检测进度" /><small>显示服务端实际阶段进度；切换数据集不会停止已经提交的任务。</small></section>}
        {run?.status === 'failed' && <div className="error-banner" role="alert"><AlertCircle size={20} /><div><strong>检测未完成</strong><p>{run.error || run.message}</p></div></div>}
        {run && completedId && summary && <>
          <nav className="analysis-jumps" aria-label="分析章节"><a href="#score-overview"><Activity size={15} />信号总览</a><a href="#variable-analysis"><Layers3 size={15} />变量分析</a><a href="#event-review"><Search size={15} />事件复核</a><button onClick={() => setPage('comparison')}><GitCompareArrows size={15} />检测任务对比</button></nav>
          <div className="results-heading"><div><span className="eyebrow">{run.model_id ? '固定模型推理结果' : '旧版检测结果（无独立模型）'}</span><h2>{runModelLabel(run)}<span className="badge teal"><Check size={12} />已完成</span></h2><p>{algorithmNames[run.algorithm]} · {run.model_id ? `模型 ID ${run.model_id}；本次数据未参与拟合或阈值校准。` : `保留原始历史结果${summary.train_end != null ? `；旧版训练区间 #0–#${summary.train_end - 1}` : ''}。该配置不可复用。`}</p></div><div className="export-actions"><button className="button secondary compact" data-testid="export-csv" onClick={() => { void exportFile('csv'); }} disabled={!!downloading}><ArrowDownToLine size={15} />{downloading === 'csv' ? '导出中…' : '标注 CSV'}</button><button className="button secondary compact" data-testid="export-report" onClick={() => { void exportFile('report'); }} disabled={!!downloading}><FileText size={15} />{downloading === 'report' ? '导出中…' : 'HTML 报告'}</button></div></div>
          <div className="notice" data-testid="inference-notes"><CircleHelp size={17} /><span>{run.message}</span></div>
          <div className="kpi-grid"><div className="kpi-card"><span>异常事件<Layers3 size={15} /></span><strong>{counts.format(summary.event_count)}<small>个</small></strong><p>按当前间隔与最短长度聚合</p></div><div className="kpi-card"><span>异常样本<Activity size={15} /></span><strong>{counts.format(summary.anomaly_points)}<small>点</small></strong><p>{summary.scored_points ? `占 ${counts.format(summary.scored_points)} 个可评分样本 ${(summary.anomaly_ratio * 100).toFixed(2)}%` : '全部不可评分 / 预热，不代表正常'}</p></div><div className="kpi-card"><span>固定校准阈值<Layers3 size={15} /></span><strong className="numeric-threshold">{summary.threshold.toPrecision(4)}</strong><p>来自实际模型版本；不同方法分数不可直接比较</p></div><div className="kpi-card"><span>可评分 / 预热<History size={15} /></span><strong data-testid="scorable-count">{counts.format(summary.scored_points)}<small>/ {counts.format(summary.warmup_points)}</small></strong><p>{counts.format(summary.points)} 总样本 · {summary.n_features} 变量 · {(summary.duration_ms / 1000).toFixed(2)} 秒</p></div></div>
          {summary.warmup_points > 0 && <div className="notice warmup-notice" data-testid="warmup-notice"><AlertCircle size={17} /><span>{summary.warmup_points} 个样本不可评分 / 预热：分数与参考为空，不计入异常比例、统计或评估，不能视为正常。</span></div>}
          <InvestigationSummary key={completedId} runId={completedId} events={events} loading={resultLoading} error={resultError} /><section className="card score-card" id="score-overview"><div className="card-heading"><div><h2>异常分数总览</h2><p>拖动底部滑块或滚轮缩放，联动下方变量分析。横轴为零基样本序号；空分数表示不可评分，不代表正常。</p></div><button className="button secondary compact" data-testid="reset-range" onClick={() => { setRange({ start: 0, end: dataset.rows }); setEventId(''); setExplanation(null); }}><RotateCcw size={14} />全范围</button></div>{resultError ? <div className="error-banner" role="alert">{resultError}<button className="text-button" onClick={() => setReload((value) => value + 1)}>重新加载</button></div> : resultLoading ? <Loading /> : overview ? <><ScoreChart data={overview} range={range} onRange={changeRange} /><div className="chart-footnote">{!run.model_id && overview.train_end != null && <span><i className="legend-square training" />旧版历史训练段（含校准）</span>}<span><i className="legend-line threshold" />固定校准阈值</span><span>灰色点：不可评分 / 预热</span><span>显示 {overview.sampling.returned_points} / {counts.format(overview.sampling.total_points)} 样本 · 峰值保留采样</span></div></> : <Empty title="暂无分数数据">请重新加载检测结果。</Empty>}</section>
          <div className="range-heading" id="variable-analysis"><div className="section-title"><Layers3 size={17} /><h2>变量级分析</h2></div><span>当前范围 #{range.start}–#{range.end - 1} · {counts.format(range.end - range.start)} 样本</span></div>
          {detailError && <div className="error-banner" role="alert">{detailError}<button className="text-button" onClick={() => setReload((value) => value + 1)}>重试</button></div>}
          <div className="analysis-grid"><section className="card"><div className="card-heading"><div><h2>变量偏离热力图</h2><p>点击热力图选择变量；颜色为偏离强度，不是异常概率。</p></div></div>{heatmapLoading ? <Loading /> : heatmap ? <><ContributionChart data={heatmap} onFeature={setFeature} /><p className="chart-method">方法：{heatmap.method}</p></> : <Empty title="热力图未加载">重新加载以查看当前范围。</Empty>}</section><section className="card"><div className="card-heading"><div><h2>原始信号与参考</h2><p>{run.algorithm === 'isolation' ? '参考为历史拟合段稳健中位数，并非模型预测。' : '参考为模型重构或过去窗口预测，不表示真实反事实。'}</p></div></div><label className="feature-control">检查变量<select aria-label="检查变量" value={feature} onChange={(event) => setFeature(event.target.value)}>{dataset.features.map((name) => <option key={name}>{name}</option>)}</select></label>{seriesLoading ? <Loading /> : detailSeries ? <FeatureChart data={detailSeries} feature={feature} isolation={run.algorithm === 'isolation'} /> : <Empty title="变量曲线未加载">选择变量并重新加载。</Empty>}<p className="chart-method">缺失观测使用拟合段中位数填补，绘图显示检测所用值。</p></section></div>
          <div className="events-grid"><EventExplorer key={completedId} events={events} eventId={eventId} loading={resultLoading} onSelect={id => { setEventId(id); setReviewSaved(false); }} onBatchReview={batchReview} />
            <section className="card explanation-card"><div className="card-heading"><div><h2>事件解释与人工复核</h2><p>{selectedEvent ? `样本 #${selectedEvent.start}–#${selectedEvent.end - 1} · ${selectedEvent.anomaly_points} 个异常点` : '解释基于事件内累计变量偏离。'}</p></div><div className="event-navigation"><button className="icon-button" aria-label="上一个异常事件" disabled={!selectedEvent || events.indexOf(selectedEvent) <= 0} onClick={() => { const index = selectedEvent ? events.indexOf(selectedEvent) : -1; if (index > 0) setEventId(events[index - 1].id); }}><ChevronLeft size={16} /></button><button className="icon-button" aria-label="下一个异常事件" disabled={!selectedEvent || events.indexOf(selectedEvent) >= events.length - 1} onClick={() => { const index = selectedEvent ? events.indexOf(selectedEvent) : -1; if (index >= 0 && index + 1 < events.length) setEventId(events[index + 1].id); }}><ChevronRight size={16} /></button></div></div>{explanationError && <div className="error-banner" role="alert">{explanationError}</div>}{explanationLoading ? <Loading text="正在计算事件解释…" /> : explanation ? <><div className="explanation-method">{explanation.method}</div><ExplanationChart data={explanation} /><div className="explanation-notes">{explanation.notes.map((note, index) => <p key={index}>{note}</p>)}</div><div className="review-divider"><span>分析师复核</span><small>保存至本地数据库</small></div><label className="field">复核结论<select data-testid="review-status" value={reviewStatus} onChange={(event) => { setReviewStatus(event.target.value as ReviewStatus); setReviewSaved(false); }} disabled={reviewSaving || batchSaving}><option value="unreviewed">待复核</option><option value="confirmed">确认异常</option><option value="false_positive">误报</option></select></label><label className="field">分析备注<textarea data-testid="review-note" rows={3} value={reviewNote} maxLength={2000} placeholder="记录观察到的信号、业务背景与后续判断依据…" onChange={(event) => { setReviewNote(event.target.value); setReviewSaved(false); }} disabled={reviewSaving || batchSaving} /></label><div className="review-actions"><span className={reviewSaved ? 'saved-label' : 'muted'} aria-live="polite">{reviewSaved ? '已保存复核记录' : `${reviewNote.length} / 2000`}</span><button className="button primary compact" data-testid="review-save" disabled={reviewSaving || batchSaving} onClick={() => { void saveReview(); }}>{reviewSaving ? <LoaderCircle className="spin" size={15} /> : <Check size={15} />}{reviewSaving ? '保存中' : '保存复核'}</button></div></> : !explanationLoading && <Empty title="选择一个事件，展开分析">点击左侧事件列表。贡献描述同步偏离，不自动推断根因。</Empty>}</section></div>
          <section className="card evaluation-card"><div className="card-heading"><div><h2>标签评估</h2><p>仅评估 {counts.format(summary.scored_points)} 个可评分样本；排除预热与旧版训练样本，不做点调整。标签与人工复核是不同信息。</p></div><span className="badge neutral">可评分样本评估</span></div>{summary.metrics ? <div className="metric-grid">{([['精确率', summary.metrics.precision], ['召回率', summary.metrics.recall], ['F1', summary.metrics.f1], ['AUPRC', summary.metrics.auprc], ['事件召回率', summary.metrics.event_recall]] as const).map(([name, value]) => <div key={name}><span>{name}</span><strong>{value == null ? '—' : value.toFixed(3)}</strong></div>)}</div> : <div className="evaluation-empty"><AlertCircle size={17} /><span>{!summary.scored_points ? '当前批次全部不可评分 / 预热，不能计算标签指标，也不代表正常。' : dataset.has_labels ? '当前运行未返回可计算的标签指标。' : '数据未提供真值标签，无法计算精确率、召回率等指标。请通过原始信号和人工复核评估结果。'}</span></div>}</section>
        </>}
      </>}
      <footer className="page-footer"><span>ChronoLens · 时序异常检测与解释系统</span><span>独立训练 · 固定模型推理 · 人工判断</span></footer>
      </main>
    </div>{importOpen && <ImportDialog onClose={() => setImportOpen(false)} onImported={acceptDataset} />}
  </div>;
}


