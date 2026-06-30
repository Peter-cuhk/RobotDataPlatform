# Robot Data Studio README First-User Design

## Goal

Rewrite the project README as a Chinese quick-start guide for first-time users. A new user
should be able to understand what Robot Data Studio does, install its dependencies, start
the frontend and backend, and complete one dataset workflow without reading the source code.

## Audience

The primary audience is a first-time local user of Robot Data Studio. Contributor-oriented
material remains available near the end, but it must not interrupt the main onboarding path.

## Structure

The README will use this order:

1. Product introduction and local-first data policy.
2. Core capabilities and supported import/export formats.
3. System requirements.
4. Installation commands, including optional sample data.
5. Backend and frontend startup commands.
6. A first-use walkthrough: import, select a run scope, configure checks, run the pipeline,
   read the cleaning report, inspect episodes, replay in Rerun, make manual decisions, and
   export clean data.
7. Artifact locations, VLM configuration, common problems, development checks, and known
   limitations.

## Content Rules

- Describe only behavior supported by the current codebase.
- Prefer copy-pasteable commands from the repository root.
- Clearly separate required setup from optional VLM, FFmpeg, and kinematics setup.
- Explain that source datasets are not modified and generated files live in
  `.rds-artifacts/`.
- Use current UI concepts, including enabled checks, selected/all episode runs, filter detail
  views, the cleaning report dashboard, downloadable report JSON, Rerun replay, and export
  scopes.
- Keep API details brief because the README is user-first rather than API-first.
- Preserve troubleshooting that directly helps local installation and startup.

## Validation

- Compare installation and startup commands with `pyproject.toml`, `package.json`, and
  `pnpm-workspace.yaml`.
- Confirm documented features and UI labels against the current frontend and API.
- Run Markdown-oriented static checks available in the repository, plus command-level smoke
  checks that do not require starting long-running servers.
- Scan the final README for stale claims, placeholder text, broken local links, and ambiguous
  required/optional dependencies.

## Out of Scope

- Changing application behavior or dependencies.
- Adding screenshots or externally hosted assets.
- Publishing deployment instructions for a production server.
- Documenting every API endpoint or internal package.
