# Robot Data Studio Design Specification

The validated product and architecture specification is maintained in the workspace root at `PRD.md`.

Design decisions approved on 2026-06-23:

- Local-first browser UI plus local Python API.
- Manipulation and embodied-learning Episode data for MVP.
- LeRobot v2/v3, ACT/robomimic HDF5, and UMI/Zarr profiles.
- Visual HDF5 schema mapping.
- Rerun as a replaceable viewer adapter.
- Canonical Episode View owned by the platform.
- Non-destructive, reproducible Pipeline.
- Deterministic quality rules plus optional pluggable VLM.
- Coordinate-system conversion with explicit frame and pose semantics.
- Initial dataset target below 10 GB, with streaming interfaces preserved for future scaling.
- React/TypeScript frontend and FastAPI/Python modular monolith.

The full requirements, interfaces, acceptance criteria, reuse strategy, risks, and roadmap are in `PRD.md`.
