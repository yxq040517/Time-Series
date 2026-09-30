import { useEffect, useRef, useState, type FormEvent, type DragEvent } from 'react';
import { FileUp, X, AlertCircle, LoaderCircle, CheckCircle2, Columns3, FileSpreadsheet } from 'lucide-react';
import { api, messageOf } from '../api';
import type { Dataset } from '../types';
import { readCsvPreview, type CsvHeaderPreview } from './csvPreview';

const MAX_FILE_SIZE = 25 * 1024 * 1024;
const fileSize = (size: number) => size >= 1024 * 1024 ? `${(size / 1024 / 1024).toFixed(2)} MiB` : `${(size / 1024).toFixed(1)} KiB`;

export function ImportDialog({ onClose, onImported }: { onClose: () => void; onImported: (dataset: Dataset) => void }) {
  const dialog = useRef<HTMLDialogElement>(null);
  const filePicker = useRef<HTMLInputElement>(null);
  const dragDepth = useRef(0);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [preview, setPreview] = useState<CsvHeaderPreview | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [previewError, setPreviewError] = useState('');
  const [timestampColumn, setTimestampColumn] = useState('');
  const [labelColumn, setLabelColumn] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  useEffect(() => {
    const node = dialog.current;
    const previous = document.activeElement;
    node?.showModal();
    return () => { node?.close(); if (previous instanceof HTMLElement) previous.focus(); };
  }, []);
  useEffect(() => {
    if (!file || file.size > MAX_FILE_SIZE) return;
    let active = true;
    setPreviewLoading(true);
    readCsvPreview(file).then(value => { if (active) setPreview(value); })
      .catch(cause => { if (active) setPreviewError(messageOf(cause)); })
      .finally(() => { if (active) setPreviewLoading(false); });
    return () => { active = false; };
  }, [file]);
  const chooseFile = (next: File | null) => {
    setFile(next); setPreview(null); setPreviewError(''); setPreviewLoading(false);
    setTimestampColumn(''); setLabelColumn('');
    setError(next && next.size > MAX_FILE_SIZE ? '文件超过 25 MiB，请缩小后导入。' : '');
  };
  const drop = (event: DragEvent<HTMLLabelElement>) => {
    event.preventDefault(); dragDepth.current = 0; setDragging(false);
    if (busy) return;
    const next = event.dataTransfer.files[0];
    if (next) chooseFile(next);
  };
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!file) { setError('请先选择 CSV 文件。'); return; }
    if (file.size > MAX_FILE_SIZE) { setError('文件超过 25 MiB，请缩小后导入。'); return; }
    const form = new FormData(event.currentTarget);
    form.set('file', file);
    for (const key of ['name', 'timestamp_column', 'label_column']) {
      const value = form.get(key);
      if (typeof value === 'string' && !value.trim()) form.delete(key);
      else if (typeof value === 'string' && key === 'name') form.set(key, value.trim());
    }
    setBusy(true); setError('');
    try { onImported(await api.upload(form)); }
    catch (cause) { setError(messageOf(cause)); }
    finally { setBusy(false); }
  };
  return <dialog ref={dialog} className="import-dialog" aria-labelledby="import-title" onCancel={(event) => { if (busy) event.preventDefault(); else onClose(); }}>
    <form onSubmit={(event) => { void submit(event); }}>
      <div className="dialog-heading"><div><span className="eyebrow">数据工作台</span><h2 id="import-title">导入时序数据</h2></div><button type="button" className="icon-button" aria-label="关闭导入窗口" onClick={onClose} disabled={busy}><X size={20} /></button></div>
      <p className="muted">支持 UTF-8 / GB18030 编码；120–100,000 行、2–64 个数值变量。数据仅在本地处理。</p>
      <div className="import-steps" aria-label="导入流程"><span className="import-step-active"><b>1</b>选择数据</span><span className={preview ? 'import-step-active' : ''}><b>2</b>映射字段</span><span><b>3</b>完整校验</span></div>
      <label className={`file-drop import-dropzone ${file ? 'has-file' : ''} ${dragging ? 'import-dragging' : ''}`} htmlFor="csv-file"
        onDragEnter={event => { event.preventDefault(); if (!busy) { dragDepth.current += 1; setDragging(true); } }}
        onDragLeave={event => { event.preventDefault(); dragDepth.current = Math.max(0, dragDepth.current - 1); if (dragDepth.current === 0) setDragging(false); }}
        onDragOver={event => { event.preventDefault(); event.dataTransfer.dropEffect = busy ? 'none' : 'copy'; }} onDrop={drop}>
        {file ? <FileSpreadsheet size={32} /> : <FileUp size={32} />}<strong>{dragging ? '松开以添加文件' : file ? file.name : '拖放 CSV 到这里，或点击选择'}</strong><span>{file ? `${fileSize(file.size)} · 点击重新选择` : '最大 25 MiB · 第一行为列名'}</span>
      </label>
      <input ref={filePicker} id="csv-file" data-testid="file-input" type="file" accept=".csv,text/csv" name="file" disabled={busy} onChange={event => chooseFile(event.target.files?.[0] || null)} />
      {file && <div className="import-file-meta"><span>{fileSize(file.size)}{preview ? ` · ${preview.encoding} · ${preview.columns.length} 列` : ''}</span><button type="button" className="text-button" onClick={() => { chooseFile(null); if (filePicker.current) filePicker.current.value = ''; }} disabled={busy}><X size={13} />移除文件</button></div>}
      {previewLoading && <div className="import-preview-loading" role="status"><LoaderCircle className="spin" size={15} />正在读取 CSV 表头…</div>}
      {preview && <div className="import-preview" data-testid="csv-header-preview"><div className="import-preview-heading"><span><Columns3 size={15} />表头预览</span><span className="badge teal"><CheckCircle2 size={11} />已读取</span></div><div className="import-columns">{preview.columns.map((column, index) => <span key={`${index}-${column}`} className={`import-column ${column === (timestampColumn || preview.timestampColumn) ? 'import-column-time' : column === (labelColumn || preview.labelColumn) ? 'import-column-label' : ''}`} title={column}>{column || '空列名'}</span>)}</div><p>仅读取文件开头，行数、数据类型与缺失值将在导入时完整校验。</p></div>}
      {previewError && <div className="import-preview-error" role="status"><AlertCircle size={15} /><span>{previewError} 仍可提交，由服务端完整校验。</span></div>}
      <label className="field">数据集名称 <span className="optional">可选</span><input name="name" placeholder="留空使用文件名称" maxLength={200} disabled={busy} /></label>
      <div className="form-grid"><label className="field">时间列名 <span className="optional">可选</span><select name="timestamp_column" aria-label="时间列名" value={timestampColumn} onChange={event => setTimestampColumn(event.target.value)} disabled={busy || previewLoading}><option value="">{preview?.timestampColumn ? `自动识别 · ${preview.timestampColumn}` : '自动识别 timestamp / 时间 等'}</option>{preview?.columns.map((column, index) => <option key={`${index}-${column}`} value={column}>{column || '空列名'}</option>)}</select></label><label className="field">标签列名 <span className="optional">可选</span><select name="label_column" aria-label="标签列名" value={labelColumn} onChange={event => setLabelColumn(event.target.value)} disabled={busy || previewLoading}><option value="">{preview?.labelColumn ? `自动识别 · ${preview.labelColumn}` : '自动识别 label / 标签 等'}</option>{preview?.columns.map((column, index) => <option key={`${index}-${column}`} value={column}>{column || '空列名'}</option>)}</select></label></div>
      <div className="notice"><AlertCircle size={16} /><span>无时间列时使用样本序号。标签必须为 0/1，仅用于训练之后的评估，不参与阈值校准。缺失值使用历史拟合段中位数填补。</span></div>
      {error && <div className="error-banner" role="alert">{error}</div>}
      <div className="dialog-actions"><button type="button" className="button secondary" onClick={onClose} disabled={busy}>取消</button><button type="submit" className="button primary" data-testid="upload-submit" disabled={busy || !file || file.size > MAX_FILE_SIZE}>{busy ? <LoaderCircle className="spin" size={16} /> : <FileUp size={16} />}{busy ? '正在解析与校验' : '校验并导入'}</button></div>
    </form>
  </dialog>;
}
