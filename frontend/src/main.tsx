import { StrictMode, Component, type ErrorInfo, type ReactNode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles.css';

class RenderBoundary extends Component<{ children: ReactNode }, { error: string }> {
  state = { error: '' };
  static getDerivedStateFromError(error: Error) { return { error: error.message || '界面渲染出现错误' }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('ChronoLens render error', error, info.componentStack); }
  render() {
    if (this.state.error) return <main className="content"><section className="card ready-card"><div><h1>分析界面暂时无法显示</h1><p role="alert">{this.state.error}</p><button className="button primary" onClick={() => window.location.reload()}>重新加载工作空间</button></div></section></main>;
    return this.props.children;
  }
}
const root = document.getElementById('root');
if (!root) throw new Error('页面缺少应用挂载节点 #root。');
createRoot(root).render(<StrictMode><RenderBoundary><App /></RenderBoundary></StrictMode>);
