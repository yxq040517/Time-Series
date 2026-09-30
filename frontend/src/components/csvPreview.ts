export interface CsvHeaderPreview {
  columns: string[];
  encoding: 'UTF-8' | 'GB18030';
  timestampColumn: string | null;
  labelColumn: string | null;
}

const PREVIEW_BYTES = 128 * 1024;
const TIME_NAMES = ['timestamp', 'time', 'datetime', 'date', '时间'];
const LABEL_NAMES = ['is_anomaly', 'label', 'anomaly', '标签'];

function headerOf(text: string, truncated: boolean): string[] {
  const columns: string[] = [];
  let current = '';
  let quoted = false;
  let finishedQuote = false;
  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (quoted) {
      if (character === '"') {
        if (text[index + 1] === '"') { current += '"'; index += 1; }
        else { quoted = false; finishedQuote = true; }
      } else current += character;
      continue;
    }
    if (character === ',' || character === '\n' || character === '\r') {
      columns.push(current); current = ''; finishedQuote = false;
      if (character !== ',') return columns;
    } else if (character === '"' && current.length === 0 && !finishedQuote) quoted = true;
    else {
      if (finishedQuote) throw new Error('CSV 表头引号格式无效，请检查列名。');
      current += character;
    }
  }
  if (quoted) throw new Error('CSV 表头引号尚未闭合，无法预览列名。');
  if (truncated) throw new Error('CSV 表头过长，无法在前 128 KiB 中读取；请直接导入进行完整校验。');
  if (text.length === 0) throw new Error('文件为空，无法预览 CSV 表头。');
  columns.push(current);
  return columns;
}

export async function readCsvPreview(file: Pick<Blob, 'size' | 'slice'>): Promise<CsvHeaderPreview> {
  const bytes = await file.slice(0, PREVIEW_BYTES).arrayBuffer();
  const truncated = file.size > PREVIEW_BYTES;
  let text: string;
  let encoding: CsvHeaderPreview['encoding'] = 'UTF-8';
  try { text = new TextDecoder('utf-8', { fatal: true }).decode(bytes, { stream: truncated }); }
  catch {
    try { text = new TextDecoder('gb18030', { fatal: true }).decode(bytes, { stream: truncated }); encoding = 'GB18030'; }
    catch { throw new Error('无法读取表头编码，请使用 UTF-8 或 GB18030 编码的 CSV。'); }
  }
  const columns = headerOf(text.replace(/^\uFEFF/, ''), truncated);
  const byName = new Map(columns.map(column => [column.trim().toLowerCase(), column]));
  return {
    columns, encoding,
    timestampColumn: TIME_NAMES.map(name => byName.get(name)).find(Boolean) ?? null,
    labelColumn: LABEL_NAMES.map(name => byName.get(name)).find(Boolean) ?? null,
  };
}
