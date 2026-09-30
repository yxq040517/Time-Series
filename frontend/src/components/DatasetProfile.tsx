import { useEffect, useMemo, useState } from 'react';
import { Activity, AlertCircle, CheckCircle2, Clock3, Database, Layers3, LoaderCircle, RefreshCw, Search } from 'lucide-react';
import type { EChartsOption } from 'echarts';
import { api, isAbort, messageOf } from '../api';
import type { Dataset, DatasetProfile as DatasetProfileData } from '../types';
import { Chart } from './Chart';

const counts = new Intl.NumberFormat('zh-CN');
const numbers = new Intl.NumberFormat('zh-CN', { maximumSignificantDigits: 6 });
function statistic(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? '—' : numbers.format(value);
}

function CorrelationMatrix({ data }: { data: DatasetProfileData['correlation'] }) {
  const option = useMemo<EChartsOption>(() => {
    const cells: [number, number, number][] = [];
    data.values.forEach((row, y) => row.forEach((value, x) => {
      if (value !== null && Number.isFinite(value)) cells.push([x, y, value]);
    }));
    const labels = data.names.length <= 12;
    return {
      animation: false,
      textStyle: { fontFamily: 'Inter, "Microsoft YaHei", sans-serif', fontSize: 12, color: '#64748b' },
      grid: { left: Math.min(165, Math.max(90, ...data.names.map((name) => name.length * 8))), right: 24, top: 16, bottom: labels ? 104 : 70 },
      tooltip: {
        trigger: 'item', renderMode: 'richText', confine: true,
        formatter: (params: unknown) => {
          if (!params || typeof params !== 'object' || !('data' in params) || !Array.isArray(params.data)) return '';
          const [x, y, value] = params.data as [number, number, number];
          return `${data.names[y]} × ${data.names[x]}\nPearson 相关系数：${value.toFixed(3)}`;
        },
      },
      xAxis: { type: 'category', data: data.names, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: true }, axisLabel: { show: labels, rotate: 35, width: 96, overflow: 'truncate', interval: 0, fontSize: 11 } },
      yAxis: { type: 'category', data: data.names, inverse: true, axisLine: { show: false }, axisTick: { show: false }, splitArea: { show: true }, axisLabel: { width: 140, overflow: 'truncate', fontSize: 11, interval: data.names.length > 24 ? 'auto' : 0 } },
      visualMap: { min: -1, max: 1, orient: 'horizontal', left: 'center', bottom: 8, itemWidth: 12, itemHeight: 160, calculable: true, precision: 2, text: ['正相关', '负相关'], textStyle: { color: '#64748b', fontSize: 11 }, inRange: { color: ['#3d71b8', '#adc6e6', '#f1f4f7', '#a4d6cc', '#0e9788'] } },
      series: [{ type: 'heatmap', data: cells, label: { show: data.names.length <= 8, fontSize: 11, formatter: (params) => Array.isArray(params.value) ? Number(params.value[2]).toFixed(2) : '' }, emphasis: { itemStyle: { borderColor: '#223a52', borderWidth: 2 } }, progressive: 2000 }],
    };
  }, [data]);
  return <Chart option={option} height={Math.min(700, Math.max(340, data.names.length * 28 + 110))} label="观测变量的 Pearson 相关系数矩阵，空白单元表示不可计算；相关性不表示因果关系" />;
}

export function DatasetProfile({ dataset }: { dataset: Dataset }) {
  const [profile, setProfile] = useState<DatasetProfileData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [search, setSearch] = useState('');
  const [reload, setReload] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    setProfile(null); setLoading(true); setError(''); setSearch('');
    api.profile(dataset.id, controller.signal).then((value) => {
      if (!controller.signal.aborted) setProfile(value);
    }).catch((cause: unknown) => {
      if (!controller.signal.aborted && !isAbort(cause)) setError(messageOf(cause));
    }).finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [dataset.id, reload]);
  const current = profile?.dataset_id === dataset.id ? profile : null;
  const filtered = useMemo(() => current?.features.filter((item) => item.name.toLocaleLowerCase().includes(search.trim().toLocaleLowerCase())) || [], [current, search]);
  const missing = current?.features.reduce((total, item) => total + item.missing, 0) || 0;
  const cells = (current?.rows || 0) * (current?.features.length || 0);

  if (error) return <section className="card"><div className="card-heading"><h2>数据诊断暂时无法读取</h2></div><div className="error-banner" role="alert"><AlertCircle size={18} /><span>{error}</span><button className="button secondary compact" onClick={() => setReload((value) => value + 1)}><RefreshCw size={15} />重试诊断</button></div></section>;
  if (loading || !current) return <section className="card"><div className="loading-state" role="status"><LoaderCircle size={22} className="spin" /><span>正在计算数据统计与采样质量…</span></div></section>;
  if (!current.features.length) return <section className="card empty-state"><Database size={28} /><strong>没有可诊断的数值变量</strong><p>请导入包含数值变量的时序数据。</p></section>;
  return <div className="profile-layout">
    <div className="kpi-grid profile-summary">
      <div className="kpi-card"><span>观测样本<Database size={17} /></span><strong>{counts.format(current.rows)}<small>行</small></strong><p>使用已导入的实际数据统计</p></div>
      <div className="kpi-card"><span>数值变量<Layers3 size={17} /></span><strong data-testid="profile-feature-count">{counts.format(current.features.length)}<small>个</small></strong><p>非数值列不参与相关性分析</p></div>
      <div className="kpi-card"><span>缺失观测<AlertCircle size={17} /></span><strong>{counts.format(missing)}<small>格</small></strong><p>占全部数值观测 {cells ? (missing / cells * 100).toFixed(2) : '0.00'}%</p></div>
      <div className="kpi-card"><span>采样依据<Clock3 size={17} /></span><strong className="fact-text">{current.sampling.mode === 'timestamp' ? '时间戳' : '样本序号'}</strong><p>{current.sampling.mode === 'timestamp' ? '检查相邻时间戳间隔' : '未提供可计算的时间间隔'}</p></div>
    </div>
    <div className="profile-quality-grid">
      <section className="card"><div className="card-heading"><div className="section-title"><Activity size={18} /><h2>采样质量</h2></div><span className="badge neutral">{current.sampling.mode === 'timestamp' ? '时间序列' : '按序观测'}</span></div>
        <div className="profile-cadence"><div><span>典型采样间隔</span><strong>{statistic(current.sampling.median_interval_seconds)}<small>{current.sampling.median_interval_seconds === null ? '' : ' 秒'}</small></strong></div><div><span>不规则间隔</span><strong>{counts.format(current.sampling.irregular_intervals)}</strong></div><div><span>重复时间间隔</span><strong>{counts.format(current.sampling.duplicate_intervals)}</strong></div></div>
        <p className="profile-correlation-note">{current.sampling.mode === 'timestamp' ? '典型间隔为相邻时间戳的正间隔中位数。不规则或重复时间可能影响按样本窗口计算的时序模型。' : '当前使用样本序号，无法从数据推断真实采样频率；检测窗口按样本数计算。'}</p>
      </section>
      <section className="card"><div className="card-heading"><div className="section-title"><CheckCircle2 size={18} /><h2>诊断提示</h2></div><span className={`badge ${current.warnings.length ? 'warm' : 'teal'}`}>{current.warnings.length ? `${current.warnings.length} 条提示` : '质量检查通过'}</span></div>
        {current.warnings.length ? <ul className="profile-warnings">{current.warnings.map((warning, index) => <li key={index}>{warning}</li>)}</ul> : <p className="profile-correlation-note">未发现当前质量规则关注的问题。请继续结合业务背景，确认训练区间是否能代表正常状态。</p>}
      </section>
    </div>
    <section className="card"><div className="card-heading"><div><h2>变量统计</h2><p>基于原始非缺失观测计算，未使用检测时的缺失值填补结果。</p></div><span className="badge neutral">{dataset.name}</span></div>
      <div className="profile-toolbar"><label className="profile-search"><Search size={17} /><input data-testid="profile-search" aria-label="搜索变量统计" placeholder="搜索变量名称…" value={search} onChange={(event) => setSearch(event.target.value)} /></label><span className="muted" aria-live="polite">显示 {filtered.length} / {current.features.length} 个变量</span></div>
      {filtered.length ? <div className="table-scroll"><table className="profile-table" data-testid="profile-table"><caption className="sr-only">各数值变量的有效观测数、缺失比例及分布统计</caption><thead><tr><th scope="col">变量</th><th scope="col">有效观测</th><th scope="col">缺失比例</th><th scope="col">最小值</th><th scope="col">最大值</th><th scope="col">均值</th><th scope="col">标准差</th><th scope="col">P25</th><th scope="col">中位数</th><th scope="col">P75</th></tr></thead><tbody>{filtered.map((item) => <tr key={item.name}><th scope="row" className="profile-variable">{item.name}{item.std === 0 && <small>恒定变量</small>}</th><td>{counts.format(item.count)}</td><td>{(item.missing_ratio * 100).toFixed(2)}%<small>{counts.format(item.missing)} 格</small></td><td>{statistic(item.min)}</td><td>{statistic(item.max)}</td><td>{statistic(item.mean)}</td><td>{statistic(item.std)}</td><td>{statistic(item.p25)}</td><td>{statistic(item.median)}</td><td>{statistic(item.p75)}</td></tr>)}</tbody></table></div> : <div className="empty-state"><Search size={25} /><strong>没有匹配的变量</strong><p>调整关键词或清空搜索查看全部变量。</p><button className="text-button" onClick={() => setSearch('')}>清空搜索</button></div>}
    </section>
    <section className="card"><div className="card-heading"><div><h2>变量相关矩阵</h2><p>Pearson 系数描述线性共同变化，范围 −1 到 1。</p></div><span className="badge neutral">{current.correlation.names.length} 个变量</span></div>
      {current.correlation.names.length >= 2 ? <CorrelationMatrix data={current.correlation} /> : <div className="empty-state"><Layers3 size={25} /><strong>变量数量不足</strong><p>至少需要两个变量才能查看变量间相关性。</p></div>}
      <p className="profile-correlation-note">空白单元表示有效数据不足或变量恒定，相关系数不可计算。缺失值按成对有效观测处理；相关性不表示因果关系。</p>
    </section>
  </div>;
}
