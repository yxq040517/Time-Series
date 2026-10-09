import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { ArrowRight, Check, CircleHelp, GitCompareArrows, History } from 'lucide-react';
import { algorithmNames, runModelLabel, type Dataset, type Run } from '../types';
import { timeLabel } from './AnalysisCharts';

const counts = new Intl.NumberFormat('zh-CN');
function ratio(value: number | null | undefined): string { return value == null ? '—' : `${(value * 100).toFixed(2)}%`; }
function metric(value: number | null | undefined): string { return value == null ? '—' : value.toFixed(3); }

export function ComparisonPage({ dataset, runs, onOpenRun }: { dataset: Dataset; runs: Run[]; onOpenRun: (run: Run) => void }) {
  const completed = useMemo(() => runs.filter((run) => run.dataset_id === dataset.id && run.status === 'completed').sort((a, b) => b.created_at.localeCompare(a.created_at)), [dataset.id, runs]);
  const [selection, setSelection] = useState<{ datasetId: string; ids: string[]; initialized: boolean }>(() => ({ datasetId: dataset.id, ids: completed.slice(0, 2).map((run) => run.id), initialized: completed.length > 0 }));
  useEffect(() => {
    setSelection((current) => {
      if (current.datasetId !== dataset.id || (!current.initialized && completed.length)) return { datasetId: dataset.id, ids: completed.slice(0, 2).map((run) => run.id), initialized: completed.length > 0 };
      const ids = current.ids.filter((id) => completed.some((run) => run.id === id));
      return ids.length === current.ids.length ? current : { ...current, ids };
    });
  }, [dataset.id, completed]);
  const selectedIds = selection.datasetId === dataset.id ? selection.ids : completed.slice(0, 2).map((run) => run.id);
  const selected = completed.filter((run) => selectedIds.includes(run.id));
  const toggle = (runId: string, checked: boolean) => {
    setSelection((current) => {
      const ids = current.datasetId === dataset.id ? current.ids.filter((id) => completed.some((run) => run.id === id)) : [];
      return { datasetId: dataset.id, ids: checked ? (ids.includes(runId) || ids.length >= 3 ? ids : [...ids, runId]) : ids.filter((id) => id !== runId), initialized: true };
    });
  };
  const row = (label: string, values: ReactNode[], different = false) => <tr className={different ? 'comparison-difference' : undefined}><th scope="row">{label}{different && <small>配置不同</small>}</th>{values.map((value, index) => <td key={selected[index].id}>{value}</td>)}</tr>;
  const configRow = (label: string, values: string[]) => row(label, values, new Set(values).size > 1);
  const hasMetrics = selected.some((run) => run.summary?.metrics != null);

  if (!completed.length) return <section className="card empty-state comparison-empty"><GitCompareArrows size={30} /><strong>先完成一次检测，再开始比较</strong><p>这里会展示当前数据集的已完成任务。使用不同检测方法或配置运行后，可以比较异常事件、评估指标与耗时。</p></section>;
  return <div className="comparison-layout">
    <section className="card"><div className="card-heading"><div><h2>选择检测任务</h2><p>同一数据集最多选择 3 个已完成任务；默认选择最近的任务。</p></div><span className="badge teal">{completed.length} 个已完成任务</span></div>
      <div className="comparison-toolbar"><span>{dataset.name}</span><span className="muted" aria-live="polite">已选 {selected.length} / 3</span><button className="text-button" onClick={() => setSelection({ datasetId: dataset.id, ids: completed.slice(0, 2).map((run) => run.id), initialized: true })}>选择最近两次</button></div>
      <div className="comparison-run-grid">{completed.map((run) => {
        const checked = selectedIds.includes(run.id);
        return <label key={run.id} className={`comparison-run-option ${checked ? 'selected' : ''}`}><input type="checkbox" data-testid={`comparison-run-${run.id}`} checked={checked} disabled={!checked && selected.length >= 3} onChange={(event) => toggle(run.id, event.target.checked)} /><span><span className="comparison-run-title">{runModelLabel(run)}{checked && <Check size={15} />}</span><span className="comparison-run-meta">{algorithmNames[run.algorithm]} · <History size={13} />{timeLabel(run.created_at)}</span><span className="comparison-run-meta">{run.summary ? `${counts.format(run.summary.event_count)} 事件 · ${run.summary.scored_points ? ratio(run.summary.anomaly_ratio) : '不可评分'} · ${run.summary.scored_points} 可评分 / ${run.summary.warmup_points} 预热` : '该任务未提供统计摘要'}</span><span className="comparison-run-meta">任务 {run.id.slice(0, 12)}</span></span></label>;
      })}</div>
    </section>
    <div className="notice"><CircleHelp size={17} /><span>比较基于同一数据集的实际结果。不同算法的原始分数不可直接比较；异常数量少或耗时短，也不代表检测质量更好。请结合标签指标与人工复核判断。</span></div>
    {!selected.length ? <section className="card empty-state comparison-empty"><GitCompareArrows size={28} /><strong>选择任务，查看差异</strong><p>勾选上方任务；选择两个或三个任务，可以并排查看结果与配置。</p></section> : <section className="card"><div className="card-heading"><div><h2>检测结果与配置对比</h2><p>{selected.length === 1 ? '再选择一个任务即可查看差异。' : '浅色标记行表示所选任务的配置不同。'}</p></div><span className="badge neutral">{selected.length} 个任务</span></div>
      <div className="table-scroll"><table className="comparison-table" data-testid="comparison-table"><caption className="sr-only">当前数据集所选已完成任务的模型来源、检测结果、标签指标与事件选项比较</caption><thead><tr><th scope="col">对比维度</th>{selected.map((run) => <th scope="col" key={run.id}><strong>{runModelLabel(run)}</strong><small>{algorithmNames[run.algorithm]} · {timeLabel(run.created_at)} · {run.id.slice(0, 8)}</small></th>)}</tr></thead><tbody>
        {row('异常事件', selected.map((run) => <strong data-testid={`comparison-events-${run.id}`} data-value={run.summary?.event_count}>{run.summary ? counts.format(run.summary.event_count) : '—'}</strong>))}
        {row('异常样本', selected.map((run) => run.summary ? counts.format(run.summary.anomaly_points) : '—'))}
        {row('异常样本比例（可评分样本）', selected.map((run) => <span data-testid={`comparison-ratio-${run.id}`} data-value={run.summary?.scored_points ? run.summary.anomaly_ratio : undefined}>{ratio(run.summary?.scored_points ? run.summary.anomaly_ratio : null)}</span>))}
        {row('可评分样本', selected.map(run => run.summary ? counts.format(run.summary.scored_points) : '—'))}
        {row('不可评分 / 预热', selected.map(run => run.summary ? counts.format(run.summary.warmup_points) : '—'))}
        {row('检测用时', selected.map((run) => <span data-testid={`comparison-duration-${run.id}`} data-value={run.summary?.duration_ms}>{run.summary ? `${(run.summary.duration_ms / 1000).toFixed(2)} 秒` : '—'}</span>))}
        {row('精确率', selected.map((run) => metric(run.summary?.metrics?.precision)))}
        {row('召回率', selected.map((run) => metric(run.summary?.metrics?.recall)))}
        {row('F1', selected.map((run) => metric(run.summary?.metrics?.f1)))}
        {row('AUPRC', selected.map((run) => metric(run.summary?.metrics?.auprc)))}
        {row('事件召回率', selected.map((run) => metric(run.summary?.metrics?.event_recall)))}
        {configRow('实际模型版本', selected.map(runModelLabel))}
        {configRow('模型 ID', selected.map(run => run.model_id || '旧版无独立模型'))}
        {configRow('固定阈值', selected.map(run => run.summary?.threshold.toPrecision(5) || '—'))}
        {configRow('连续流 ID', selected.map(run => typeof run.config.stream_id === 'string' && run.config.stream_id ? run.config.stream_id : '独立批次'))}
        {configRow('最短事件长度', selected.map(run => typeof run.config.min_event_length === 'number' ? `${run.config.min_event_length} 样本` : '—'))}
        {configRow('事件合并间隔', selected.map(run => typeof run.config.merge_gap === 'number' ? `${run.config.merge_gap} 样本` : '—'))}
        {row('继续分析', selected.map((run) => <button className="text-button" onClick={() => onOpenRun(run)}>打开此任务<ArrowRight size={14} /></button>))}
      </tbody></table></div>
      {!hasMetrics && <div className="notice"><CircleHelp size={17} /><span>{dataset.has_labels ? '所选任务没有可计算的标签评估指标。' : '当前数据集未提供真值标签，无法计算精确率、召回率与 F1 等指标。'}「—」表示不可计算或未提供，不表示 0。</span></div>}
    </section>}
  </div>;
}
