# Progress

## Status
- **Current milestone:** M0 setup
- **GPU spend:** see `docs/GPU_LEDGER.md`

## Done
- M0.1 environment: Modal profile connected, no stray apps, Docker running, no ANTHROPIC_API_KEY,
  gh logged in. Modal workspace had $0 usage this month at start.
- M0.2 repo skeleton, plan renamed to PLAN.md, `.gitattributes` LF.

## Next step
- Build CPU docker image, smoke-test Modal image build (CPU download task), then L4 smoke test.

## Restart checklist
1. `python -m modal app list` and stop anything running.
2. `git status`; commit leftovers as `wip: recovered after interruption`.
3. Read this file and continue from "Next step".
