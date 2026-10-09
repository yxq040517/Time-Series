import type { AnomalyEvent, Dataset, DatasetProfile, RunInsights, Explanation, Heatmap, Preview, ReviewStatus, Run, DetectionConfig, ModelInfo, TrainingConfig, TrainingJob, SampleRange, Series } from './types';

async function errorMessage(response: Response): Promise<string> {
  let message = `请求失败（HTTP ${response.status}）`;
  try {
    const body: unknown = await response.json();
    if (body && typeof body === 'object' && 'detail' in body) {
      const detail = body.detail;
      if (typeof detail === 'string') message = detail;
      else if (Array.isArray(detail)) message = detail.map((item: unknown) => {
        if (item && typeof item === 'object' && 'msg' in item) {
          const location = 'loc' in item && Array.isArray(item.loc) ? item.loc.slice(1).map(String).join(' / ') : '参数';
          return `${location || '参数'}：${String(item.msg)}`;
        }
        return String(item);
      }).join('；');
    }
  } catch { /* Non-JSON errors keep their HTTP status. */ }
  return message;
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try { response = await fetch(`/api${path}`, init); }
  catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new Error('无法连接本地服务，请检查服务是否已启动。');
  }
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json() as Promise<T>;
}
const json = (value: unknown): RequestInit => ({ method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(value) });
const runPath = (id: string) => `/runs/${encodeURIComponent(id)}`;
export const api = {
  profile: (id: string, signal?: AbortSignal) => request<DatasetProfile>(`/datasets/${encodeURIComponent(id)}/profile`, { signal }),
  insights: (id: string, signal?: AbortSignal) => request<RunInsights>(`${runPath(id)}/insights`, { signal }),
  batchReview: (id: string, event_ids: string[], status: ReviewStatus) => request<{ items: AnomalyEvent[] }>(`${runPath(id)}/events/review`, json({ event_ids, status })),
  datasets: (signal?: AbortSignal) => request<{ items: Dataset[] }>('/datasets', { signal }),
  models: (signal?: AbortSignal) => request<{ items: ModelInfo[] }>('/models', { signal }),
  trainings: (signal?: AbortSignal) => request<{ items: TrainingJob[] }>('/trainings', { signal }),
  training: (id: string, signal?: AbortSignal) => request<TrainingJob>(`/trainings/${encodeURIComponent(id)}`, { signal }),
  createTraining: (config: TrainingConfig, signal?: AbortSignal) => request<TrainingJob>('/trainings', { ...json(config), signal }),
  publishModel: (id: string, signal?: AbortSignal) => request<ModelInfo>(`/models/${encodeURIComponent(id)}/publish`, { ...json({}), signal }),
  disableModel: (id: string, signal?: AbortSignal) => request<ModelInfo>(`/models/${encodeURIComponent(id)}/disable`, { ...json({}), signal }),
  demo: () => request<Dataset>('/datasets/demo', json({ seed: 42 })),
  upload: (form: FormData) => request<Dataset>('/datasets/upload', { method: 'POST', body: form }),
  preview: (id: string, signal?: AbortSignal) => request<Preview>(`/datasets/${encodeURIComponent(id)}/preview?limit=8`, { signal }),
  runs: (datasetId: string, signal?: AbortSignal) => request<{ items: Run[] }>(`/runs?${new URLSearchParams({ dataset_id: datasetId })}`, { signal }),
  createRun: (config: DetectionConfig) => request<Run>('/runs', json(config)),
  run: (id: string, signal?: AbortSignal) => request<Run>(runPath(id), { signal }),
  series: (id: string, range: SampleRange, features: string[], signal?: AbortSignal) => {
    const query = new URLSearchParams({ start: String(range.start), end: String(range.end), max_points: '1600' });
    if (features.length) query.set('features', features.join(','));
    return request<Series>(`${runPath(id)}/series?${query}`, { signal });
  },
  heatmap: (id: string, range: SampleRange, signal?: AbortSignal) => request<Heatmap>(`${runPath(id)}/heatmap?${new URLSearchParams({ start: String(range.start), end: String(range.end), bins: '120' })}`, { signal }),
  events: (id: string, signal?: AbortSignal) => request<{ items: AnomalyEvent[] }>(`${runPath(id)}/events`, { signal }),
  explanation: (id: string, eventId: string, signal?: AbortSignal) => request<Explanation>(`${runPath(id)}/events/${encodeURIComponent(eventId)}/explanation`, { signal }),
  review: (id: string, eventId: string, status: ReviewStatus, note: string) => request<AnomalyEvent>(`${runPath(id)}/events/${encodeURIComponent(eventId)}`, { ...json({ status, note }), method: 'PATCH' }),
};
export function isAbort(error: unknown): boolean { return error instanceof DOMException && error.name === 'AbortError'; }
export function messageOf(error: unknown): string { return error instanceof Error ? error.message : '发生未知错误，请重试。'; }
export async function download(path: string, fallback: string): Promise<void> {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  const blob = await response.blob();
  const disposition = response.headers.get('Content-Disposition') || '';
  const encoded = /filename\*=UTF-8''([^;]+)/i.exec(disposition)?.[1];
  const plain = /filename="?([^";]+)"?/i.exec(disposition)?.[1];
  let name = plain || fallback;
  if (encoded) { try { name = decodeURIComponent(encoded); } catch { name = fallback; } }
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url; anchor.download = name; document.body.appendChild(anchor); anchor.click(); anchor.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

