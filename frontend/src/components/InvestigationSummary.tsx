import { useEffect, useState } from 'react';
import { AlertTriangle, ArrowRight, CheckCheck, ScanLine } from 'lucide-react';
import { api, isAbort } from '../api';
import type { AnomalyEvent, RunInsights } from '../types';

export function InvestigationSummary({ runId, events, loading, error }: { runId: string; events: AnomalyEvent[]; loading: boolean; error: string }) {
  const [insights, setInsights] = useState<RunInsights | null>(null);
  useEffect(() => {
    const controller = new AbortController(); setInsights(null);
    api.insights(runId, controller.signal).then(value => { if (!controller.signal.aborted) setInsights(value); }).catch(cause => { if (!isAbort(cause)) setInsights(null); });
    return () => controller.abort();
  }, [runId]);
  const reviewed = events.filter(item => item.status !== 'unreviewed').length;
  const high = events.filter(item => item.severity === 'high').length;
  const progress = events.length ? Math.round(reviewed / events.length * 100) : 0;
  if (loading || error) return <div className="investigation-summary summary-pending" role="status"><ScanLine size={22} /><span>{loading ? '正在整理事件与复核进度…' : '事件数据暂未加载，请重试检测结果。'}</span></div>;
  return <div className="investigation-summary">
    <div className="investigation-focus"><span className="focus-icon"><ScanLine size={25} /></span><div><span className="section-kicker">INVESTIGATION BRIEF</span><h3>{events.length ? `${events.length - reviewed} 个事件等待你的判断` : '本次检测没有聚合异常事件'}</h3><p>{high ? `${high} 个高严重度事件，建议优先结合原始信号检查。` : '结合信号与业务背景，完成异常事件复核。'}</p></div><a href="#event-review" className="brief-link">进入复核<ArrowRight size={16} /></a></div>
    <div className="review-progress"><div><span><CheckCheck size={16} />复核进度</span><strong>{reviewed}<small> / {events.length}</small></strong></div><progress value={progress} max={100} aria-label="事件复核完成百分比" /><p>{progress}% 已复核 · {events.filter(item => item.status === 'confirmed').length} 个确认异常</p></div>
    {insights && <div className="anomaly-density"><div><span><AlertTriangle size={15} />异常分布</span><small>训练后时段</small></div><div className="density-bars" role="img" aria-label="训练后各时段的异常样本比例">{insights.timeline.map((bucket, index) => <span key={bucket.start} title={`${bucket.label}：${bucket.anomaly_points} / ${bucket.total_points} 异常样本`} style={{ height: `${bucket.anomaly_points / Math.max(1, bucket.total_points) * 100}%`, background: bucket.anomaly_points ? '#f29a63' : '#dbe6e3' }}><i className="sr-only">时段 {index + 1}：{bucket.anomaly_points} 个异常样本</i></span>)}</div></div>}
  </div>;
}

