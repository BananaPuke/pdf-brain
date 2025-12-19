---
"pdf-brain": major
---

# 🔄 Database Migration: PGLite → libSQL

**BREAKING CHANGE**: Complete rewrite of the database layer.

## What Changed

Replaced PGLite (WASM Postgres + pgvector) with libSQL for vector storage. This is a **breaking change** - existing PGLite databases are not compatible. Use the migration script to convert your data.

## Why

PGLite with pgvector was causing crashes under heavy embedding load:

- WASM memory limits (~2GB ceiling)
- WAL file accumulation (found 2.7GB orphaned files)
- Required complex daemon/socket architecture to work around single-connection limit
- ~15MB bundle size for pgvector WASM

## Benefits

- **Native vector support** - libSQL's `F32_BLOB(N)` type, no extensions needed
- **Rock-solid WAL** - SQLite's battle-tested WAL mode
- **Simpler architecture** - No daemon, no socket server, no write queue
- **Smaller bundle** - ~200KB vs ~15MB
- **Concurrent access** - libSQL handles it natively

## Migration

If you have existing data in PGLite format:

```bash
# Export from old PGLite database
bun run scripts/migration/pglite-to-libsql.ts

# Or re-ingest your documents
pdf-brain ingest ~/your/pdf/directory
```

## Removed

- Daemon service (`pdf-brain daemon start/stop/status`)
- Socket-based database client
- Write queue architecture
- PGLite dependencies (@electric-sql/pglite, pglite-socket, pglite-tools)
