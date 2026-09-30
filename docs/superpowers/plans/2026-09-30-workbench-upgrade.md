# ChronoLens Workbench Implementation Plan

> Implement task by task with parallel agents and root integration; unavailable superpowers execution skills are replaced by the current collaboration tools.

**Goal:** Improve the visual quality and investigation workflow while adding data diagnostics, model comparison, and atomic batch review.
**Architecture:** Keep React/FastAPI and existing detection/persistence contracts. New statistics live in a focused backend module; separate frontend components own diagnostic and comparison views.
**Tech Stack:** React 19, TypeScript, ECharts 6, Vite 7, FastAPI, NumPy, SQLite, pytest, Playwright.

## Global Constraints
- Keep existing backend/data and detection calibration behavior.
- Statistics represent observed data; unavailable statistics and labels are explicitly empty.
- Comparisons use completed tasks from the same dataset and do not compare raw scores between algorithms.
- CSV upload must preserve existing analysis on failure.
- Batch review validates all event IDs before any write and preserves notes unless explicitly submitted.

## Task 1: Backend diagnosis, insights, batch review, and performance
Files: backend/diagnostics.py, backend/app.py, backend/schemas.py, backend/storage.py; backend/tests/test_diagnostics.py and test_api.py.
- [x] Add tests asserting observed variable statistics, missing values, constant-feature null correlation, timestamp cadence, and index time handling.
- [x] Add TestClient tests for profile/insights success and 404, batch review rollback and note preservation, invalid statuses, cached array immutability and rewrite invalidation.
- [x] Run `.venv/Scripts/python -m pytest -p no:cacheprovider -q` and verify new assertions fail for the missing behavior.
- [x] Implement GET datasets/{id}/profile, GET runs/{id}/insights, POST runs/{id}/events/review; move upload parsing into run_in_threadpool; add bounded read-only array cache.
- [x] Run all backend tests and verify original detection tests remain green.

## Task 2: Data diagnostics and model comparison
Files: frontend/src/components/DatasetProfile.tsx, ComparisonPage.tsx; root owns types.ts and api.ts.
Consumes: dataset: Dataset, runs: Run[], profile: DatasetProfile from api.profile(id, signal).
Produces: DatasetProfile({dataset}) and ComparisonPage({dataset,runs,onOpenRun}).
- [x] Add real browser assertions that diagnosis shows actual variable count/statistics, comparison selects completed tasks from current dataset and shows their actual summary values.
- [x] Observe missing-page failure before implementation.
- [x] Build focused components with abort-safe requests, loading/retry/empty states, semantic tables and responsive charts.
- [x] Integrate into navigation; verify TypeScript and browser checks.

## Task 3: Configuration presets and import interaction
Files: RunConfiguration.tsx, ImportDialog.tsx, csvPreview.ts; tests/frontend CSV preview tests.
- [x] Add unit tests for quoted CSV headers, BOM, CRLF and incomplete previews; run and observe missing behavior.
- [x] Add configuration scenario presets while keeping current schema bounds and default values.
- [x] Add drag/drop, header preview and optional column mapping, keeping final server validation and failed-upload retention.
- [x] Run targeted tests and TypeScript check.

## Task 4: Workbench visual and review flow
Files: frontend/src/App.tsx, styles.css, EventExplorer.tsx, InvestigationSummary.tsx; scripts/verify_upgrade.mjs.
- [x] Write browser assertions for readable navigation, result-first layout, search/severity/sort, checked selection and batch persistence, mobile overflow.
- [x] Run baseline browser script and observe missing navigation or controls.
- [x] Rebuild typography, color, spacing, cards, nav and responsive layouts. Collapse setup for completed runs, add section navigation and review progress.
- [x] Add search, severity and sorting with selection and atomic batch API; update local selected explanation after batch review.
- [x] Run npm run build, original verify_ui.mjs and upgrade browser tests; inspect desktop/mobile screenshots.

## Task 5: Delivery
- [x] Build production frontend/dist and run on a new local port using actual production code.
- [x] Update Chinese usage instructions and changelog with implemented functionality and verification evidence.
- [x] Save final screenshots and verification notes to this chat outputs, open the running app, and provide concise launch instructions.


Release evidence: 33 pytest + 6 Node tests, existing and new Playwright flows, race regression, release capture all passed. New production service is on 8767; original 8765 process was preserved because automatic approval review rejected the stop-process action. Launcher and dev proxy now use 8767.

