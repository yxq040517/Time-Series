# ChronoLens · 时序异常检测与解释系统

基于 **FastAPI、React、TypeScript 与 ECharts** 的本地多变量时序分析工作台。支持 CSV 导入、真实异常检测、变量偏离解释、人工复核与离线报告。

![ChronoLens 分析工作台](docs/images/workbench.png)

## 功能

- **三种检测方法**：PCA 重构、时序 Ridge 自回归、IsolationForest。
- **交互分析**：异常分数与历史阈值、缩放联动、变量热力图、原始信号与模型参考。
- **数据诊断**：变量分布、缺失情况、采样间隔、相关矩阵与变量搜索。
- **模型对比**：同一数据集最多比较三次已完成任务的结果、标签指标与配置。
- **事件复核**：关键词、严重度和状态筛选，排序、分页、批量复核与备注持久化。
- **导入与配置**：CSV 拖放、UTF-8／GB18030 表头预览、时间和标签列映射、参数预设与历史配置复用。
- **本地持久化**：SQLite 保存记录，压缩 NPZ 保存数据；导出标注 CSV 与无需外部资源的 HTML 报告。

## 快速运行（Windows）

安装 **Python 3.10、3.11 或 3.12**，并在安装时启用 PATH，然后克隆仓库：

```powershell
git clone https://github.com/yxq040517/Time-Series.git
cd Time-Series
.\start.bat
```

也可以下载仓库 ZIP、解压后双击 `start.bat`。首次运行会创建 `.venv` 并安装 Python 依赖，需要网络；仓库包含 `frontend/dist`，普通运行无需 Node.js。

浏览器会自动打开 **http://127.0.0.1:8767**。保持启动窗口开启，按 `Ctrl+C` 停止该次启动的服务。其他启动方式：

```powershell
.\start.bat --no-browser
.\start.bat --port 8768
```

数据保存在 `backend/data/`，该目录不进入 Git。迁移到新电脑时由启动器重新创建虚拟环境。

## 体验流程

1. 点击“生成演示”，或导入 `examples/server_metrics.csv`／自己的 CSV。
2. 设置方法和训练比例，点击“开始异常检测”。
3. 查看分数、变量分析和事件解释，结合原始信号复核。
4. 保存结论与备注，或批量处理选中的事件。
5. 在“数据诊断”和“模型对比”中继续检查数据与检测结果，最后导出报告。

CSV 需包含 **120–100,000 行、2–64 个数值变量**，最大 **25 MiB**。时间列须升序；标签列可选，值为 0 或 1。无时间列时使用样本序号。

![真实变量统计与采样质量](docs/images/data-diagnostics.png)

## 开发与测试

前端开发和重建使用 **Node.js 24**：

```powershell
cd frontend
npm ci
npm run dev
# 开发界面 http://127.0.0.1:5173，API 代理到 8767
npm run build
```

后端与单元测试，在仓库根目录运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -p no:cacheprovider -q
node scripts/test_configuration_import.mjs
```

浏览器验收需先启动服务并安装前端依赖及 Playwright 浏览器：

```powershell
cd frontend
npx playwright install chromium
cd ..
node scripts/verify_ui.mjs
node scripts/verify_upgrade.mjs
node scripts/verify_diagnostics.mjs
node scripts/test_review_races.mjs
```

验收脚本会在所连接服务中创建合成测试数据和检测任务。可通过 `CHRONOLENS_URL` 指向单独的验收服务，通过 `CHRONOLENS_DATA_DIR` 为后端指定独立数据目录。

本次验证：**33 项 Python 测试、6 项 Node 行为测试、TypeScript 检查、生产构建及浏览器流程通过**；浏览器覆盖桌面、390px 手机和 768px 平板布局，以及请求延迟、错误重试和复核状态一致性。

## 项目结构

```text
backend/             FastAPI、检测算法、统计诊断、SQLite/NPZ 存储与测试
frontend/src/        React/TypeScript 界面与 ECharts 分析组件
frontend/dist/       已构建的生产界面
examples/            合成 CSV 与数据字典
scripts/             启动器、示例生成及浏览器验收
docs/                改版方案、实施记录与界面截图
start.bat            Windows 启动入口
开始使用.txt          简明运行说明
使用说明.html         完整离线说明
```

## 分析约定

拟合、缺失值处理和阈值校准仅使用历史训练段，后续标签用于评估。变量贡献描述模型偏离，**不等于因果根因**；合成数据上的指标不代表生产准确率。批量复核默认只更新结论，未保存备注需单独提交。

默认服务只监听本机地址，面向本地分析与研究演示。完整参数、数据格式和 API 说明见 [使用说明.html](使用说明.html)。
