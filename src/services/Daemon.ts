/**
 * PGlite Daemon Service
 *
 * Solves PGlite's single-connection limitation by running a daemon process
 * that owns the PGlite instance and exposes it via Unix socket.
 *
 * Why: When multiple CLI invocations create their own PGlite instances,
 * they corrupt the database. The daemon ensures ONE connection.
 *
 * References:
 * - https://pglite.dev/docs/pglite-socket
 * - Semantic memory: 48610ac6-d52f-4505-8b06-9df2fad353aa (multi-connection corruption bug)
 * - Semantic memory: 4d167832-70e4-46b0-85ba-170e5826b9c8 (CHECKPOINT pattern)
 */

import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";
import { PGLiteSocketServer } from "@electric-sql/pglite-socket";
import { dirname, join } from "node:path";
import { existsSync, mkdirSync, rmSync } from "node:fs";

export interface DaemonConfig {
  /**
   * Unix socket directory for IPC (e.g., ~/.pdf-library)
   * Socket file will be created as .s.PGSQL.5432 in this directory
   * following PostgreSQL Unix socket convention
   */
  socketPath: string;
  /** PID file path for lifecycle management (e.g., ~/.pdf-library/daemon.pid) */
  pidPath: string;
  /** PGlite database directory path */
  dbPath: string;
}

/** Active daemon instance (singleton within process) */
let activeDaemon: {
  server: PGLiteSocketServer;
  db: PGlite;
  config: DaemonConfig;
} | null = null;

/**
 * Start the PGlite daemon
 *
 * 1. Creates PGlite instance with pgvector
 * 2. Starts Unix socket server
 * 3. Writes PID file
 * 4. Sets up graceful shutdown handlers
 *
 * @throws Error if daemon already running
 */
export async function startDaemon(config: DaemonConfig): Promise<void> {
  // Check if already running
  if (await isDaemonRunning(config)) {
    throw new Error("Daemon already running");
  }

  // Ensure directories exist
  const socketDir = config.socketPath; // Now this is the directory, not file path
  const pidDir = dirname(config.pidPath);
  const dbDir = dirname(config.dbPath);

  for (const dir of [socketDir, pidDir, dbDir]) {
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }
  }

  // Initialize PGlite with pgvector extension
  const pgDataDir = config.dbPath.replace(".db", "");
  const db = new PGlite(pgDataDir, { extensions: { vector } });
  await db.waitReady;

  // Initialize schema (must happen after waitReady)
  // Embedding dimension for mxbai-embed-large
  const EMBEDDING_DIM = 1024;

  await db.exec("CREATE EXTENSION IF NOT EXISTS vector;");

  // Documents table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      path TEXT NOT NULL UNIQUE,
      added_at TIMESTAMPTZ NOT NULL,
      page_count INTEGER NOT NULL,
      size_bytes INTEGER NOT NULL,
      tags JSONB DEFAULT '[]',
      metadata JSONB DEFAULT '{}'
    )
  `);

  // Chunks table
  await db.exec(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      page INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL
    )
  `);

  // Embeddings table with vector column
  await db.exec(`
    CREATE TABLE IF NOT EXISTS embeddings (
      chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
      embedding vector(${EMBEDDING_DIM}) NOT NULL
    )
  `);

  // Create HNSW index for fast approximate nearest neighbor search
  await db.exec(`
    CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx 
    ON embeddings 
    USING hnsw (embedding vector_cosine_ops)
  `);

  // Full-text search index
  await db.exec(`
    CREATE INDEX IF NOT EXISTS chunks_content_idx 
    ON chunks 
    USING gin (to_tsvector('english', content))
  `);

  // Other indexes
  await db.exec(`CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)`);
  await db.exec(`CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(path)`);

  // Start Unix socket server with PostgreSQL naming convention
  // Creates socket at: ${socketDir}/.s.PGSQL.5432
  const socketFile = join(socketDir, ".s.PGSQL.5432");
  const server = new PGLiteSocketServer({
    db,
    path: socketFile,
  });
  await server.start();

  // Write PID file
  await Bun.write(config.pidPath, `${process.pid}`);

  // Store active daemon
  activeDaemon = { server, db, config };

  // Graceful shutdown handlers
  const shutdown = async () => {
    await stopDaemon(config);
    process.exit(0);
  };

  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

/**
 * Stop the PGlite daemon
 *
 * 1. Runs CHECKPOINT to flush WAL
 * 2. Closes socket server
 * 3. Closes PGlite instance
 * 4. Removes PID file and socket
 */
export async function stopDaemon(config: DaemonConfig): Promise<void> {
  const socketFile = join(config.socketPath, ".s.PGSQL.5432");

  if (!activeDaemon) {
    // Not running in this process - clean up files if they exist
    if (existsSync(socketFile)) {
      rmSync(socketFile, { force: true });
    }
    if (existsSync(config.pidPath)) {
      rmSync(config.pidPath, { force: true });
    }
    return;
  }

  try {
    // Run CHECKPOINT to flush WAL before closing
    // This prevents WAL accumulation and ensures clean shutdown
    await activeDaemon.db.exec("CHECKPOINT");
  } catch (e) {
    // Log but don't fail - we still want to clean up
    console.error("CHECKPOINT failed during shutdown:", e);
  }

  // Close socket server
  await activeDaemon.server.stop();

  // Close PGlite
  await activeDaemon.db.close();

  // Remove PID file
  if (existsSync(config.pidPath)) {
    rmSync(config.pidPath, { force: true });
  }

  // Remove socket file (already declared at function start)
  if (existsSync(socketFile)) {
    rmSync(socketFile, { force: true });
  }

  // Clear active daemon
  activeDaemon = null;
}

/**
 * Check if daemon is running
 *
 * Verifies:
 * 1. Socket file exists
 * 2. PID file exists
 * 3. Process with that PID is alive
 *
 * @returns true if daemon is running, false otherwise
 */
export async function isDaemonRunning(config: DaemonConfig): Promise<boolean> {
  // Check socket file exists (PostgreSQL naming convention)
  const socketFile = join(config.socketPath, ".s.PGSQL.5432");
  if (!existsSync(socketFile)) {
    return false;
  }

  // Check PID file exists
  if (!existsSync(config.pidPath)) {
    return false;
  }

  // Read PID
  const pidContent = await Bun.file(config.pidPath).text();
  const pid = parseInt(pidContent.trim(), 10);

  if (Number.isNaN(pid) || pid <= 0) {
    return false;
  }

  // Check if process is alive
  try {
    // Sending signal 0 doesn't kill the process, just checks if it exists
    process.kill(pid, 0);
    return true;
  } catch {
    // Process doesn't exist
    return false;
  }
}

/**
 * Result of ensuring daemon is running
 *
 * Represents the outcome of attempting to start the daemon.
 * Used by DatabaseLive to decide routing strategy.
 */
export interface EnsureDaemonResult {
  /** Whether daemon is available for use */
  success: boolean;
  /** Connection mode: 'daemon' if daemon running, 'fallback' if using direct PGlite */
  mode: "daemon" | "fallback";
  /** Error message if daemon failed to start (only present when mode='fallback') */
  error?: string;
}

/**
 * Ensure daemon is running - auto-start if not running
 *
 * **DAEMON-FIRST ARCHITECTURE**: This is the entry point for making daemon
 * the default mode. Every database operation goes through this function.
 *
 * Flow:
 * 1. Check if daemon already running (fast path - no startup cost)
 * 2. If not running, attempt to start daemon in background
 * 3. If start succeeds → return success with mode='daemon'
 * 4. If start fails → return fallback mode (caller uses direct PGlite)
 *
 * **Graceful Degradation**: This function NEVER throws. If daemon can't start,
 * it returns a fallback result, allowing the caller to fall back to direct PGlite.
 * This ensures the application always works, even if daemon is unavailable.
 *
 * **Why auto-start?**
 * - Solves PGlite's single-connection limitation by default
 * - No manual `pdf-brain daemon start` required
 * - Multi-process safety out of the box
 * - Graceful fallback preserves single-process simplicity when needed
 *
 * @param config - Daemon configuration (socket path, PID path, DB path)
 * @returns Result indicating daemon status and mode
 *
 * @example
 * ```typescript
 * const result = await ensureDaemonRunning(config);
 * if (result.success) {
 *   // Use DatabaseClient (Unix socket connection)
 *   console.log("Using daemon mode - multi-process safe");
 * } else {
 *   // Use direct PGlite (fallback)
 *   console.warn(`Daemon failed: ${result.error}`);
 *   console.log("Using direct PGlite - single process only");
 * }
 * ```
 */
export async function ensureDaemonRunning(
  config: DaemonConfig
): Promise<EnsureDaemonResult> {
  // Fast path: check if already running
  if (await isDaemonRunning(config)) {
    return { success: true, mode: "daemon" };
  }

  // Daemon not running - try to start it
  try {
    await startDaemon(config);
    return { success: true, mode: "daemon" };
  } catch (error) {
    // Daemon failed to start - return fallback mode
    // This is graceful degradation, not a failure
    // Caller can fall back to direct PGlite
    return {
      success: false,
      mode: "fallback",
      error: error instanceof Error ? error.message : String(error),
    };
  }
}
