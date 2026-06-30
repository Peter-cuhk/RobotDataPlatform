# Robot Data Studio Community Demo Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a 75-110 second English-subtitled, no-BGM community demo video for Robot Data Studio.

**Architecture:** Record real browser interactions from the current local app into scene-based clips, then compose those clips into a polished 1920x1080 MP4. Use HyperFrames if final rendering dependencies are available; otherwise use a deterministic Python/FFmpeg composition script so the user still receives the requested video.

**Tech Stack:** Playwright + Chrome for capture, FastAPI/Uvicorn and Vite for local app runtime, FFmpeg for encoding, Python/Pillow for fallback composition, HyperFrames source/spec where practical.

---

### Task 1: Prepare Runtime

**Files:**
- Read: `package.json`
- Read: `apps/api/main.py`
- Read: `scripts/record_promo.js`
- Create: `output/community-video/`

- [ ] **Step 1: Check whether ports 8000 and 5173 already serve the app**

Run:

```bash
lsof -iTCP:8000 -sTCP:LISTEN -n -P || true
lsof -iTCP:5173 -sTCP:LISTEN -n -P || true
```

- [ ] **Step 2: Start missing services**

Run backend if port 8000 is missing:

```bash
.venv/bin/uvicorn apps.api.main:app --host 127.0.0.1 --port 8000
```

Run frontend if port 5173 is missing:

```bash
pnpm dev:web
```

- [ ] **Step 3: Verify app health**

Run:

```bash
curl -s http://127.0.0.1:8000/formats
curl -I http://127.0.0.1:5173/
```

Expected: API returns JSON and frontend returns HTTP 200.

### Task 2: Record Scene-Based Clips

**Files:**
- Create: `scripts/record_community_video.js`
- Create: `output/community-video/raw/*.webm`
- Create: `output/community-video/stills/*.png`

- [ ] **Step 1: Write a Playwright recording script**

The script should launch Chrome, use viewport 1440x900, import `data/samples/aloha_static_coffee`, run cleaning, open filter detail, replay in Rerun, export selected or passed episodes, and save scene clips/stills.

- [ ] **Step 2: Run the recording script**

Run:

```bash
node scripts/record_community_video.js
```

Expected: raw clips and stills appear under `output/community-video/`.

- [ ] **Step 3: Verify captures exist**

Run:

```bash
find output/community-video -maxdepth 3 -type f | sort
```

Expected: at least one full workflow browser recording and stills for import, quality, review, Rerun, and export.

### Task 3: Compose Final Video

**Files:**
- Create: `scripts/make_community_video.py`
- Create: `output/community-video/final/robot-data-studio-community-demo.mp4`
- Create: `output/community-video/final/thumbnail.png`

- [ ] **Step 1: Write composition script**

The script should make a 90-100 second 1920x1080 MP4 with English subtitles, no background music, title/end cards, real product footage, zoom emphasis, and smooth pacing.

- [ ] **Step 2: Render the video**

Run:

```bash
.venv/bin/python scripts/make_community_video.py
```

Expected: final MP4 and thumbnail are written.

- [ ] **Step 3: Verify media**

Run:

```bash
ls -lh output/community-video/final
.venv/bin/python - <<'PY'
from pathlib import Path
for path in Path("output/community-video/final").glob("*"):
    print(path, path.stat().st_size)
PY
```

Expected: MP4 is non-empty and larger than 1 MB; thumbnail is non-empty.

### Task 4: Final QA

**Files:**
- Read: `output/community-video/final/robot-data-studio-community-demo.mp4`
- Read: `output/community-video/final/thumbnail.png`

- [ ] **Step 1: Extract QA frames**

Run:

```bash
.venv/bin/python scripts/make_community_video.py --qa-only
```

Expected: QA stills are generated from several timestamps.

- [ ] **Step 2: Inspect QA frames**

Check that subtitles are readable, product UI is visible, no BGM/audio track is required, and the final video is within 75-110 seconds.

- [ ] **Step 3: Report delivery paths**

Return the MP4 path, thumbnail path, and any caveats about runtime dependencies.
