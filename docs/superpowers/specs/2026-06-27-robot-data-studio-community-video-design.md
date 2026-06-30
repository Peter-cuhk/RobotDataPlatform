# Robot Data Studio Community Demo Video Specification

Date: 2026-06-27
Audience: open-source users, robotics developers, robot learning researchers, and data engineers.
Target length: 90 seconds, acceptable range 75-110 seconds.
Audio: no background music. Prefer silent video with English subtitles; optional spoken English narration can be added later from the same script.

## Goal

Create a demo-first open-source community video that shows Robot Data Studio as a local-first workspace for turning raw robot episodes into trusted training data.

The video should feel like a real product workflow, not a slide deck. The screen recording should carry the story. Motion graphics should only clarify: title cards, subtitles, mouse guidance, local zooms, and clean transitions.

## Core Message

Primary tagline:

> From raw robot episodes to trusted training data.

Secondary positioning:

> A local-first open-source workspace for robot dataset quality.

The video should communicate three advantages:

- Local-first: source datasets stay on the user's machine and are not copied or modified by default.
- Explainable quality workflow: scores, findings, thresholds, and manual review live together.
- Training-format bridge: clean episodes can be exported to formats such as ACT HDF5, robomimic HDF5, UMI Zarr, and LeRobot.

## Required Tooling

Available and recommended:

- HyperFrames: available through `npx hyperframes`. Use it for the final composition, subtitles, transitions, zoom regions, and MP4 render.
- Chrome: available at `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome`.
- Playwright: available through `/tmp/rds-video-tools/node_modules/playwright`, matching the existing promo capture scripts.
- Existing capture scripts: `scripts/record_promo.js` and `scripts/record_promo_tail.js` can be used as references, but should be rewritten for scene-based recording rather than screenshot assembly.
- Canva: available, but optional. Use only for thumbnails or social preview images.
- Figma: available, but optional. Use only if a reusable visual system or title-card mockup is needed.

Readiness notes:

- `npx hyperframes doctor` reports HyperFrames 0.7.14 and Chrome are ready.
- System `ffmpeg` and `ffprobe` are not on PATH. The project virtual environment has a bundled FFmpeg binary at `.venv/lib/python3.12/site-packages/imageio_ffmpeg/binaries/ffmpeg-macos-aarch64-v7.1`. For HyperFrames render, either install system FFmpeg with `brew install ffmpeg` or make wrapper scripts/symlinks available on PATH for `ffmpeg` and `ffprobe`.
- Whisper, Kokoro TTS, MusicGen, and Docker are not needed for the no-BGM subtitle-first version.

## Visual Style

Use a quiet technical style that matches the current Robot Data Studio UI:

- Dark UI, high contrast, restrained blue and green accents.
- Typography should be crisp and developer-oriented.
- Keep subtitles readable and short.
- Avoid marketing-heavy animation, stock footage, fake dashboards, or generic AI imagery.
- Avoid screenshot-only montage. Every product section should be based on real browser recording.

## Video Structure

### Scene 1: Problem Hook, 0-8s

Visuals:
Fast cuts from real UI states: dataset path, episode list, cleaning summary, quality report, Rerun replay, export result.

Subtitle:
`Robot datasets are hard to trust before training.`

Purpose:
Set the pain point quickly for robotics developers.

### Scene 2: Product Definition, 8-18s

Visuals:
Open Robot Data Studio. Show the header, local status, and dataset path input.

Subtitle:
`Robot Data Studio is a local-first workspace for inspecting, cleaning, replaying, and converting robot datasets.`

Purpose:
Explain what the project is in one sentence.

### Scene 3: Local Import, 18-32s

Visuals:
Fill the dataset path, select auto-detect, click Import dataset, then hold on metadata: format, episodes, frames, fps.

Subtitle:
`Import a local dataset without copying or modifying the source files.`

Purpose:
Make the local-first and non-destructive behavior concrete.

### Scene 4: Quality Pipeline, 32-50s

Visuals:
Click Run cleaning Pipeline. Show the cleaning summary chart and Pass / Review / Exclude counters.

Subtitle:
`Run quality checks across episodes and turn raw data into reviewable evidence.`

Purpose:
Show that the product is more than a viewer.

### Scene 5: Explainable Review, 50-65s

Visuals:
Open a filter detail view. Use a local zoom on findings, issue reasons, and threshold/chart evidence.

Subtitle:
`Every decision stays explainable: scores, findings, thresholds, and manual review in one place.`

Purpose:
Highlight trust and reviewability for open-source users.

### Scene 6: Rerun Replay, 65-80s

Visuals:
Click Replay in Rerun. Show synchronized video, state, action, and timeline.

Subtitle:
`Replay video, state, action, and timelines together with Rerun.`

Purpose:
Connect Robot Data Studio to a known visualization strength while showing it inside a larger workflow.

### Scene 7: Export, 80-92s

Visuals:
Choose export scope and format. Click Export. Hold on the exported result.

Subtitle:
`Export clean episodes to training formats like ACT HDF5, robomimic HDF5, UMI Zarr, and LeRobot.`

Purpose:
Close the workflow from inspection to training-ready artifacts.

### Scene 8: Open-Source Close, 92-100s

Visuals:
Return to the clean product view or a simple title card.

Subtitle:
`Open source. Local-first. Built for robot learning datasets before training.`

End card:
`From raw robot episodes to trusted training data.`

Purpose:
Leave viewers with a memorable one-line value proposition.

## Capture Plan

Record separate browser clips instead of one long take:

- `01-hook-scan.webm`: quick UI scan across the main screens.
- `02-import.webm`: dataset path and import flow.
- `03-quality.webm`: cleaning pipeline and summary.
- `04-review.webm`: filter detail and explainable findings.
- `05-rerun.webm`: Rerun replay.
- `06-export.webm`: export flow.

Recommended recording settings:

- Browser viewport: 1440x900 for capture.
- Final video: 1920x1080.
- Use deterministic pauses and mouse movements.
- Capture actual interaction states, not static screenshots.
- Add zoom and crop in HyperFrames rather than recording the browser at extreme zoom.

## Editing Plan

Use HyperFrames for:

- Full composition timeline.
- English subtitles synced to each scene.
- Short title cards at the beginning and end.
- Smooth transitions between scenes.
- Local zoom overlays for quality findings, counters, Rerun viewer, and export status.
- Optional cursor highlight when clicking important controls.

Avoid:

- Background music.
- Long full-screen static screenshots.
- Overly decorative animation.
- Explaining every button.
- Showing incomplete or planned features as if they are finished.

## Success Criteria

The final video is successful if an open-source viewer can answer these questions within 90 seconds:

- What problem does Robot Data Studio solve?
- Can it run locally on my data?
- What does the core workflow look like?
- How does it help me trust data before training?
- What can I export after review?

The final render should include:

- One 75-110 second MP4.
- One silent subtitle-first version.
- The source HyperFrames project.
- Reusable recorded browser clips.
- A thumbnail frame or still image for GitHub/social sharing.

## Recommended Next Step

Create a dedicated video workspace under `output/community-video/`, rewrite the Playwright capture script into scene-based clips, scaffold a HyperFrames project, and build the final composition from those clips with English subtitles.
