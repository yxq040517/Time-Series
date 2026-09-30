import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { BarChart, HeatmapChart, LineChart, ScatterChart } from 'echarts/charts';
import { DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapContinuousComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { ECharts, EChartsOption } from 'echarts';

echarts.use([BarChart, HeatmapChart, LineChart, ScatterChart, DataZoomComponent, GridComponent, LegendComponent, MarkAreaComponent, MarkLineComponent, TooltipComponent, VisualMapContinuousComponent, CanvasRenderer]);

interface ChartProps {
  option: EChartsOption;
  height?: number;
  label: string;
  onEvents?: Record<string, (value: unknown) => void>;
}
export function Chart({ option, height = 280, label, onEvents }: ChartProps) {
  const element = useRef<HTMLDivElement>(null);
  const chart = useRef<ECharts | null>(null);
  const latestEvents = useRef(onEvents);
  latestEvents.current = onEvents;
  const eventNames = Object.keys(onEvents || {}).sort().join(',');
  useEffect(() => {
    if (!element.current) return;
    const instance = echarts.init(element.current, undefined, { renderer: 'canvas' });
    chart.current = instance;
    const resize = () => { if (!instance.isDisposed()) instance.resize(); };
    const observer = new ResizeObserver(resize);
    observer.observe(element.current);
    window.addEventListener('resize', resize);
    return () => {
      observer.disconnect(); window.removeEventListener('resize', resize);
      chart.current = null; instance.dispose();
    };
  }, []);
  useEffect(() => { chart.current?.setOption(option, { notMerge: true }); }, [option]);
  useEffect(() => {
    const instance = chart.current;
    if (!instance || !eventNames) return;
    const handlers = eventNames.split(',').map((name) => {
      const handler = (value: unknown) => latestEvents.current?.[name]?.(value);
      instance.on(name, handler);
      return { name, handler };
    });
    return () => { if (!instance.isDisposed()) handlers.forEach(({ name, handler }) => instance.off(name, handler)); };
  }, [eventNames]);
  return <div ref={element} className="echart" style={{ height }} role="img" aria-label={label} />;
}
