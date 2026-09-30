import { useMemo, useState } from 'react';
import { AlertCircle, Check, ChevronLeft, ChevronRight, Filter, LoaderCircle, Search, X } from 'lucide-react';
import type { AnomalyEvent, ReviewStatus } from '../types';
import { timeLabel } from './AnalysisCharts';
import { messageOf } from '../api';

const names: Record<ReviewStatus, string> = { unreviewed: '待复核', confirmed: '确认异常', false_positive: '误报' };
const severities = { high: '高', medium: '中', low: '低' };
const pageSize = 50;

export function EventExplorer({ events, eventId, loading, onSelect, onBatchReview }: {
  events: AnomalyEvent[]; eventId: string; loading: boolean;
  onSelect: (id: string) => void;
  onBatchReview: (ids: string[], status: ReviewStatus) => Promise<void>;
}) {
  const [query, setQuery] = useState('');
  const [severity, setSeverity] = useState('all');
  const [status, setStatus] = useState('all');
  const [sort, setSort] = useState('time');
  const [page, setPage] = useState(0);
  const [selected, setSelected] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [feedback, setFeedback] = useState('');
  const [error, setError] = useState('');
  const filtered = useMemo(() => events.filter(item =>
    (severity === 'all' || item.severity === severity) && (status === 'all' || item.status === status) &&
    `${item.top_feature} ${item.note} ${item.start_time} ${item.id} #${item.start}`.toLowerCase().includes(query.toLowerCase().trim())
  ).sort((a, b) => sort === 'peak' ? b.peak_score - a.peak_score : sort === 'length' ? b.length - a.length : a.start - b.start), [events, query, severity, status, sort]);
  const currentPage = Math.min(page, Math.max(0, Math.ceil(filtered.length / pageSize) - 1));
  const visible = filtered.slice(currentPage * pageSize, (currentPage + 1) * pageSize);
  const allChecked = visible.length > 0 && visible.every(item => selected.includes(item.id));
  const toggle = (id: string) => { setFeedback(''); setSelected(ids => ids.includes(id) ? ids.filter(item => item !== id) : ids.length < 500 ? [...ids, id] : ids); };
  const review = async (next: ReviewStatus) => {
    if (!selected.length || busy) return;
    const count = selected.length;
    setBusy(true); setFeedback(''); setError('');
    try { await onBatchReview(selected, next); setFeedback(`已将 ${count} 个事件标记为「${names[next]}」`); setSelected([]); }
    catch (cause) { setError(messageOf(cause)); }
    finally { setBusy(false); }
  };
  return <section className="card events-card" id="event-review">
    <div className="card-heading"><div><span className="section-kicker">EVENT EXPLORER</span><h2>异常事件<span className="heading-count">{events.length}</span></h2><p>定位值得关注的片段，保留每一次复核判断。</p></div><Filter size={20} className="muted" /></div>
    <div className="event-filters">
      <label className="search-field"><Search size={16} /><input data-testid="event-search" aria-label="搜索异常事件" placeholder="搜索变量、备注或样本序号" value={query} onChange={e => { setQuery(e.target.value); setPage(0); }} /></label>
      <div className="event-filter-row">
        <select data-testid="event-severity" aria-label="筛选严重程度" value={severity} onChange={e => { setSeverity(e.target.value); setPage(0); }}><option value="all">全部严重程度</option><option value="high">高严重度</option><option value="medium">中严重度</option><option value="low">低严重度</option></select>
        <select id="event-filter" aria-label="筛选复核状态" value={status} onChange={e => { setStatus(e.target.value); setPage(0); }}><option value="all">全部状态</option><option value="unreviewed">待复核</option><option value="confirmed">确认异常</option><option value="false_positive">误报</option></select>
        <select data-testid="event-sort" aria-label="事件排序" value={sort} onChange={e => { setSort(e.target.value); setPage(0); }}><option value="time">时间顺序</option><option value="peak">峰值优先</option><option value="length">持续长度</option></select>
      </div>
    </div>
    <div className="event-selection"><label><input data-testid="select-filtered-events" type="checkbox" checked={allChecked} disabled={busy || !visible.length} onChange={() => setSelected(ids => allChecked ? ids.filter(id => !visible.some(item => item.id === id)) : [...new Set([...ids, ...visible.map(item => item.id)])].slice(0, 500))} />选择本页</label><span>{selected.length ? `已选 ${selected.length} 个` : `${filtered.length} 个匹配事件`}</span>{selected.length > 0 && <button className="text-button" disabled={busy} onClick={() => setSelected([])}>清空</button>}</div>
    {selected.length > 0 && <div className="batch-actions"><span>{busy && <LoaderCircle size={15} className="spin" />}批量复核</span><button className="button primary compact" data-testid="batch-confirm" disabled={busy} onClick={() => void review('confirmed')}><Check size={14} />确认异常</button><button className="button secondary compact" data-testid="batch-false-positive" disabled={busy} onClick={() => void review('false_positive')}><X size={14} />标记误报</button><button className="text-button" disabled={busy} onClick={() => void review('unreviewed')}>待复核</button></div>}
    {feedback && <div className="batch-message" data-testid="batch-message" role="status"><Check size={16} />{feedback}</div>}
    {error && <div className="error-banner" role="alert"><AlertCircle size={16} />{error}</div>}
    {loading ? <div className="loading-state" role="status"><LoaderCircle size={22} className="spin" />正在读取事件…</div> : visible.length ? <div className="event-list">{visible.map((item, index) => <div className={`event-item ${eventId === item.id ? 'selected' : ''}`} key={item.id}>
      <label className="event-checkbox"><input type="checkbox" aria-label={`选择事件 ${item.id}`} checked={selected.includes(item.id)} disabled={busy || (selected.length >= 500 && !selected.includes(item.id))} onChange={() => toggle(item.id)} /></label>
      <button data-testid={`event-row-${item.id}`} className="event-row" aria-pressed={eventId === item.id} onClick={() => onSelect(item.id)}><span className={`event-index severity-${item.severity}`}>{String(currentPage * pageSize + index + 1).padStart(2, '0')}</span><span className="event-main"><strong>{timeLabel(item.start_time)}<span className={`severity-label ${item.severity}`}>{severities[item.severity]}</span></strong><span>#{item.start}–#{item.end - 1} · {item.length} 样本</span><span className="event-feature">{item.top_feature}</span></span><span className="event-meta"><span className={`badge ${item.status === 'confirmed' ? 'teal' : item.status === 'false_positive' ? 'neutral' : 'warm'}`}>{names[item.status]}</span><small>峰值 {item.peak_score.toPrecision(4)}</small></span><ChevronRight size={15} /></button>
    </div>)}</div> : <div className="empty-state"><Search size={27} /><strong>{events.length ? '没有匹配的异常事件' : '本次检测未发现符合聚合条件的事件'}</strong><p>{events.length ? '调整关键词或筛选条件，继续检查其他事件。' : '可结合分数曲线与最短事件设置继续分析。'}</p></div>}
    {filtered.length > pageSize && <div className="event-pagination"><span>{currentPage + 1} / {Math.ceil(filtered.length / pageSize)} 页</span><button className="icon-button" aria-label="上一页事件" disabled={currentPage === 0} onClick={() => setPage(currentPage - 1)}><ChevronLeft size={16} /></button><button className="icon-button" aria-label="下一页事件" disabled={(currentPage + 1) * pageSize >= filtered.length} onClick={() => setPage(currentPage + 1)}><ChevronRight size={16} /></button></div>}
  </section>;
}
