# Frozen Reference Implementations

`bin/memory` and `bin/execute` are **frozen reference implementations** from
before the `memory_system` migration.

They are no longer used by the formal evaluation path:

- `run_libero_eval.py` uses `memory_system.execute`.
- New offline construction lives in `memory_system/offline`.
- New online execution lives in `memory_system/execute`.

These old files are kept only for:

- historical reference;
- old/new consistency tests;
- debugging and comparison with the migrated implementation.

Do **not** add new features here. New changes should go into
`memory_system/`.
