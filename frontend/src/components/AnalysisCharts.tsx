import { useMemo } from 'react';
import type { EChartsOption } from 'echarts';
import type { Explanation, Heatmap, SampleRange, Series } from '../types';
import { Chart } from './Chart';

const teal = '#119785';
const indigo = '#5369cc';
const anomaly = '#e47752';
const axisLabel = { color: '#64748b', fontSize: 12 };
const common = {
  animation: false,
  textStyle: { fontFamily: 'Inter, "Microsoft YaHei", sans-serif', color: '#64748b', fontSize: 12 },
  tooltip: { trigger: 'axis', renderMode: 'richText', confine: true, backgroundColor: '#ffffff', borderColor: '#dce5e8', borderWidth: 1, borderRadius: 10, padding: 12, textStyle: { color: '#334155', fontSize: 13, lineHeight: 22 } },
  grid: { left: 62, right: 24, top: 36, bottom: 50 },
} satisfies EChartsOption;
const integer = new Intl.NumberFormat('zh-CN');
export function timeLabel(value: string): string {
  if (/^\d+$/.test(value)) return `样本 #${value}`;
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
}
export function ScoreChart({ data, range, onRange }: { data: Series; range: SampleRange; onRange: (range: SampleRange) => void }) {
  const option = useMemo<EChartsOption>(() => {
    const startPosition = Math.max(0, data.indices.findIndex((index) => index >= range.start));
    const endPosition = Math.max(startPosition, data.indices.reduce((last, index, position) => index < range.end ? position : last, -1));
    const trainingLast = data.indices.reduce((last, index, position) => index < data.train_end ? position : last, -1);
    return {
      ...common, grid: { left: 62, right: 24, top: 36, bottom: 76 },
      legend: { top: 0, right: 20, icon: 'roundRect', itemWidth: 14, itemHeight: 9, textStyle: { color: '#526477', fontSize: 12 } },
      xAxis: { type: 'category', data: data.indices.map(String), boundaryGap: false, axisLine: { lineStyle: { color: '#dfe7ee' } }, axisTick: { show: false }, axisLabel: { ...axisLabel, formatter: (value: string) => `#${integer.format(Number(value))}` } },
      yAxis: { type: 'value', name: '异常分数', nameTextStyle: { color: '#64748b', align: 'left', fontSize: 12 }, axisLabel, splitLine: { lineStyle: { color: '#edf1f5' } } },
      tooltip: { ...common.tooltip, trigger: 'axis', formatter: (params: unknown) => {
        const rows = Array.isArray(params) ? params : [];
        const item: unknown = rows[0];
        if (!item || typeof item !== 'object' || !('dataIndex' in item) || typeof item.dataIndex !== 'number') return '';
        const position = item.dataIndex;
        return `${timeLabel(data.timestamps[position])} · 样本 #${data.indices[position]}\n异常分数：${data.scores[position].toPrecision(5)}\n校准阈值：${data.threshold.toPrecision(5)}\n检测判断：${data.flags[position] ? '异常点' : '正常点'}${data.labels ? `\n真值标签：${data.labels[position] ? '异常' : '正常'}` : ''}`;
      } },
      dataZoom: [
        { type: 'inside', startValue: startPosition, endValue: endPosition, filterMode: 'none', throttle: 180 },
        { type: 'slider', bottom: 12, height: 24, startValue: startPosition, endValue: endPosition, borderColor: '#dce5e8', fillerColor: 'rgba(17,151,133,.14)', handleStyle: { color: teal }, textStyle: { color: '#64748b', fontSize: 12 }, filterMode: 'none', throttle: 180 },
      ],
      series: [
        { name: '异常分数', type: 'line', data: data.scores, symbol: 'none', lineStyle: { color: teal, width: 2.2 }, itemStyle: { color: teal }, areaStyle: { color: 'rgba(17,151,133,.08)' },
          markLine: { symbol: 'none', label: { formatter: '校准阈值', color: '#b85a36', fontSize: 12, position: 'insideEndTop' }, lineStyle: { color: anomaly, type: 'dashed', width: 1.5 }, data: [{ yAxis: data.threshold }] },
          markArea: trainingLast >= 0 ? { silent: true, itemStyle: { color: 'rgba(83,105,204,.08)' }, label: { show: true, color: '#5369aa', fontSize: 12, position: 'insideTopLeft' }, data: [[{ name: '历史训练区间', xAxis: '0' }, { xAxis: String(data.indices[trainingLast]) }]] } : undefined,
        },
        { name: '超阈值点', type: 'scatter', symbolSize: 6, itemStyle: { color: anomaly }, data: data.scores.map((score, position) => data.flags[position] ? score : null), z: 4 },
      ],
    };
  }, [data, range]);
  const onZoom = (event: unknown) => {
    if (!event || typeof event !== 'object') return;
    let entry = event;
    if ('batch' in event && Array.isArray(event.batch) && event.batch[0] && typeof event.batch[0] === 'object') entry = event.batch[0];
    const begin = 'startValue' in entry && typeof entry.startValue === 'number' ? entry.startValue : 'start' in entry && typeof entry.start === 'number' ? Math.floor(entry.start / 100 * (data.indices.length - 1)) : 0;
    const finish = 'endValue' in entry && typeof entry.endValue === 'number' ? entry.endValue : 'end' in entry && typeof entry.end === 'number' ? Math.ceil(entry.end / 100 * (data.indices.length - 1)) : data.indices.length - 1;
    const start = data.indices[Math.max(0, Math.min(data.indices.length - 1, Math.round(begin)))];
    const end = data.indices[Math.max(0, Math.min(data.indices.length - 1, Math.round(finish)))] + 1;
    if (start !== undefined && end > start) onRange({ start, end });
  };
  return <Chart option={option} height={300} label="异常分数、阈值与历史训练区间，拖动下方滑块联动分析范围" onEvents={{ datazoom: onZoom }} />;
}
export function ContributionChart({ data, onFeature }: { data: Heatmap; onFeature: (name: string) => void }) {
  const option = useMemo<EChartsOption>(() => {
    const cells: [number, number, number][] = [];
    let maximum = 0;
    data.values.forEach((row, featureIndex) => row.forEach((value, position) => { cells.push([position, featureIndex, value]); maximum = Math.max(maximum, value); }));
    return {
      ...common, grid: { left: Math.min(180, Math.max(100, ...data.features.map((name) => name.length * 9))), right: 28, top: 12, bottom: 65 },
      tooltip: { ...common.tooltip, trigger: 'item', formatter: (params: unknown) => {
        if (!params || typeof params !== 'object' || !('data' in params) || !Array.isArray(params.data)) return '';
        const [position, feature, value] = params.data;
        return `${data.features[Number(feature)]}\n${timeLabel(data.timestamps[Number(position)])}\n样本 #${data.indices[Number(position)]}\n偏离贡献：${Number(value).toPrecision(4)}`;
      } },
      xAxis: { type: 'category', data: data.indices.map(String), axisTick: { show: false }, axisLine: { show: false }, splitArea: { show: true }, axisLabel: { ...axisLabel, formatter: (value: string) => `#${integer.format(Number(value))}` } },
      yAxis: { type: 'category', data: data.features, inverse: true, axisLine: { show: false }, axisTick: { show: false }, axisLabel: { ...axisLabel, width: 155, overflow: 'truncate', color: '#526477' } },
      visualMap: { min: 0, max: maximum || 1, orient: 'horizontal', left: 'center', bottom: 0, itemWidth: 12, itemHeight: 145, calculable: true, precision: 2, text: ['较高偏离', '较低偏离'], textStyle: { fontSize: 12, color: '#64748b' }, inRange: { color: ['#eef5f3', '#c6e3dc', '#87cbbc', '#3caf99', '#176c68'] } },
      series: [{ type: 'heatmap', data: cells, progressive: 2000, emphasis: { itemStyle: { borderColor: '#15273c', borderWidth: 1 } } }],
    };
  }, [data]);
  return <Chart option={option} height={Math.min(570, Math.max(240, data.features.length * 29 + 100))} label="各变量偏离贡献热力图，点击选择变量" onEvents={{ click: (params: unknown) => {
    if (params && typeof params === 'object' && 'data' in params && Array.isArray(params.data)) {
      const feature = data.features[Number(params.data[1])];
      if (feature) onFeature(feature);
    }
  } }} />;
}
export function FeatureChart({ data, feature, isolation }: { data: Series; feature: string; isolation: boolean }) {
  const option = useMemo<EChartsOption>(() => ({
    ...common, legend: { top: 0, right: 10, icon: 'roundRect', itemWidth: 14, itemHeight: 9, textStyle: { color: '#526477', fontSize: 12 } },
    xAxis: { type: 'category', data: data.indices.map(String), boundaryGap: false, axisTick: { show: false }, axisLine: { lineStyle: { color: '#e1e8ee' } }, axisLabel: { ...axisLabel, formatter: (value: string) => `#${integer.format(Number(value))}` } },
    yAxis: { type: 'value', scale: true, axisLabel, splitLine: { lineStyle: { color: '#edf1f5' } } },
    tooltip: { ...common.tooltip, trigger: 'axis', formatter: (params: unknown) => {
      const first: unknown = Array.isArray(params) ? params[0] : null;
      if (!first || typeof first !== 'object' || !('dataIndex' in first) || typeof first.dataIndex !== 'number') return '';
      const index = first.dataIndex;
      return `${feature}\n${timeLabel(data.timestamps[index])} · #${data.indices[index]}\n原始值：${data.values[feature]?.[index]?.toPrecision(5) ?? '无数据'}\n${isolation ? '稳健中位数参考' : '模型参考'}：${data.reference[feature]?.[index]?.toPrecision(5) ?? '无数据'}`;
    } },
    series: [
      { name: '原始观测', type: 'line', data: data.values[feature] || [], symbol: 'none', lineStyle: { color: teal, width: 2 }, itemStyle: { color: teal } },
      { name: isolation ? '稳健中位数' : '模型参考', type: 'line', data: data.reference[feature] || [], symbol: 'none', lineStyle: { color: indigo, width: 2, type: 'dashed' }, itemStyle: { color: indigo } },
    ],
  }), [data, feature, isolation]);
  return <Chart option={option} height={260} label={`${feature} 原始观测与${isolation ? '稳健中位数' : '模型参考'}曲线`} />;
}
export function ExplanationChart({ data }: { data: Explanation }) {
  const top = data.top_features.slice(0, 8).reverse();
  const option = useMemo<EChartsOption>(() => ({
    ...common, grid: { left: 110, right: 58, top: 12, bottom: 24 },
    xAxis: { type: 'value', max: 1, axisLabel: { ...axisLabel, formatter: (value: number) => `${Math.round(value * 100)}%` }, splitLine: { lineStyle: { color: '#edf1f5' } } },
    yAxis: { type: 'category', data: top.map((item) => item.name), axisTick: { show: false }, axisLine: { show: false }, axisLabel: { ...axisLabel, width: 96, overflow: 'truncate', color: '#526477' } },
    tooltip: { ...common.tooltip, trigger: 'item', formatter: (params: unknown) => {
      if (!params || typeof params !== 'object' || !('dataIndex' in params) || typeof params.dataIndex !== 'number') return '';
      const item = top[params.dataIndex];
      return `${item.name}\n事件内累计偏离：${item.contribution.toPrecision(5)}\n偏离贡献占比：${(item.share * 100).toFixed(1)}%\n占比不是异常概率，也不表示因果关系。`;
    } },
    series: [{ type: 'bar', data: top.map((item) => item.share), barWidth: 16, itemStyle: { color: teal, borderRadius: [0, 4, 4, 0] }, label: { show: true, position: 'right', color: '#526477', fontSize: 12, formatter: (params) => `${(Number(params.value) * 100).toFixed(1)}%` } }],
  }), [data]);
  return <Chart option={option} height={Math.max(150, top.length * 30 + 45)} label="事件内变量偏离贡献占比，不代表因果关系" />;
}
