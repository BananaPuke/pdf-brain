#!/usr/bin/env bun
/**
 * Migrate data from PGlite to LibSQL
 *
 * This script migrates an existing PGlite database to LibSQL format.
 * Handles schema translation, vector format conversion (pgvector → F32_BLOB),
 * and data migration.
 *
 * Prerequisites:
 *   - Existing PGlite database at specified path
 *   - Bun runtime (for @libsql/client)
 *
 * Usage:
 *   bun run scripts/migration/pglite-to-libsql.ts [pglite-db-path] [libsql-db-path]
 *
 * Example:
 *   bun run scripts/migration/pglite-to-libsql.ts \
 *     ~/Documents/.pdf-library/library \
 *     ~/Documents/.pdf-library/library-libsql.db
 *
 * Vector Format Conversion:
 *   - PGlite: vector(1024) column with '[1.2,3.4,...]' text format
 *   - LibSQL: F32_BLOB(1024) column with JSON.stringify([1.2,3.4,...])
 */

import { PGlite } from "@electric-sql/pglite";
import { vector } from "@electric-sql/pglite/vector";
import { createClient } from "@libsql/client";
import { existsSync, mkdirSync } from "fs";
import { dirname } from "path";

// Embedding dimension for mxbai-embed-large
const EMBEDDING_DIM = 1024;

const args = process.argv.slice(2);
const pglitePath =
  args[0] ||
  `${process.env.HOME}/Documents/.pdf-library/library`.replace(".db", "");
const libsqlPath =
  args[1] || `${process.env.HOME}/Documents/.pdf-library/library-libsql.db`;

interface MigrationStats {
  documents: number;
  chunks: number;
  embeddings: number;
  skippedEmbeddings: number;
  errors: string[];
}

/**
 * Parse pgvector text format to number array
 * PGlite returns vectors as '[1.2,3.4,5.6,...]'
 */
function parseVector(vectorText: string): number[] {
  // Remove brackets and parse
  const cleaned = vectorText.replace(/^\[|\]$/g, "");
  return cleaned.split(",").map((v) => parseFloat(v.trim()));
}

/**
 * Initialize LibSQL schema
 */
async function initLibSQLSchema(client: ReturnType<typeof createClient>) {
  console.log("Creating LibSQL schema...");

  // Documents table
  await client.execute(`
    CREATE TABLE IF NOT EXISTS documents (
      id TEXT PRIMARY KEY,
      title TEXT NOT NULL,
      path TEXT NOT NULL UNIQUE,
      added_at TEXT NOT NULL,
      page_count INTEGER NOT NULL,
      size_bytes INTEGER NOT NULL,
      tags TEXT DEFAULT '[]',
      metadata TEXT DEFAULT '{}'
    )
  `);

  // Chunks table
  await client.execute(`
    CREATE TABLE IF NOT EXISTS chunks (
      id TEXT PRIMARY KEY,
      doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
      page INTEGER NOT NULL,
      chunk_index INTEGER NOT NULL,
      content TEXT NOT NULL
    )
  `);

  // Embeddings table with F32_BLOB
  await client.execute(`
    CREATE TABLE IF NOT EXISTS embeddings (
      chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
      embedding F32_BLOB(${EMBEDDING_DIM}) NOT NULL
    )
  `);

  // Create indexes
  await client.execute(
    `CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id)`
  );
  await client.execute(
    `CREATE INDEX IF NOT EXISTS idx_docs_path ON documents(path)`
  );

  console.log("✓ Schema created");
}

/**
 * Migrate documents from PGlite to LibSQL
 */
async function migrateDocuments(
  pgDb: PGlite,
  libsqlClient: ReturnType<typeof createClient>,
  stats: MigrationStats
) {
  console.log("\nMigrating documents...");

  const result = await pgDb.query("SELECT * FROM documents ORDER BY id");
  const docs = result.rows;

  console.log(`  Found ${docs.length} documents`);

  for (const doc of docs) {
    try {
      const docRow = doc as {
        id: string;
        title: string;
        path: string;
        added_at: string;
        page_count: number;
        size_bytes: number;
        tags: unknown;
        metadata: unknown;
      };

      await libsqlClient.execute({
        sql: `INSERT INTO documents (id, title, path, added_at, page_count, size_bytes, tags, metadata)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          docRow.id,
          docRow.title,
          docRow.path,
          docRow.added_at, // Already ISO string in PGlite
          docRow.page_count,
          docRow.size_bytes,
          JSON.stringify(docRow.tags), // JSONB → TEXT
          JSON.stringify(docRow.metadata || {}), // JSONB → TEXT
        ],
      });
      stats.documents++;
    } catch (e) {
      const error = `Document ${(doc as { id: string }).id}: ${e}`;
      stats.errors.push(error);
      console.error(`  ✗ ${error}`);
    }
  }

  console.log(`  ✓ Migrated ${stats.documents} documents`);
}

/**
 * Migrate chunks from PGlite to LibSQL
 */
async function migrateChunks(
  pgDb: PGlite,
  libsqlClient: ReturnType<typeof createClient>,
  stats: MigrationStats
) {
  console.log("\nMigrating chunks...");

  const countResult = await pgDb.query("SELECT COUNT(*) as c FROM chunks");
  const totalChunks = parseInt((countResult.rows[0] as any).c);

  console.log(`  Found ${totalChunks} chunks`);

  const batchSize = 100;
  let migrated = 0;

  for (let offset = 0; offset < totalChunks; offset += batchSize) {
    const batch = await pgDb.query(
      `SELECT * FROM chunks ORDER BY id LIMIT ${batchSize} OFFSET ${offset}`
    );

    // Use libSQL batch for transaction
    const statements = batch.rows.map((chunk) => {
      const chunkRow = chunk as {
        id: string;
        doc_id: string;
        page: number;
        chunk_index: number;
        content: string;
      };
      return {
        sql: "INSERT INTO chunks (id, doc_id, page, chunk_index, content) VALUES (?, ?, ?, ?, ?)",
        args: [
          chunkRow.id,
          chunkRow.doc_id,
          chunkRow.page,
          chunkRow.chunk_index,
          chunkRow.content,
        ],
      };
    });

    try {
      await libsqlClient.batch(statements, "write");
      migrated += batch.rows.length;
      stats.chunks += batch.rows.length;
    } catch (e) {
      const error = `Chunk batch at offset ${offset}: ${e}`;
      stats.errors.push(error);
      console.error(`  ✗ ${error}`);
    }

    if (migrated % 1000 === 0 || migrated === totalChunks) {
      console.log(`  Progress: ${migrated}/${totalChunks}`);
    }
  }

  console.log(`  ✓ Migrated ${stats.chunks} chunks`);
}

/**
 * Migrate embeddings from PGlite to LibSQL
 *
 * KEY CONVERSION:
 * - PGlite: vector(1024) → returns as '[1.2,3.4,...]' text
 * - LibSQL: F32_BLOB(1024) → requires JSON.stringify([1.2,3.4,...])
 */
async function migrateEmbeddings(
  pgDb: PGlite,
  libsqlClient: ReturnType<typeof createClient>,
  stats: MigrationStats
) {
  console.log("\nMigrating embeddings...");

  try {
    const countResult = await pgDb.query(
      "SELECT COUNT(*) as c FROM embeddings"
    );
    const totalEmb = parseInt((countResult.rows[0] as any).c);

    console.log(`  Found ${totalEmb} embeddings`);

    const batchSize = 50; // Smaller batches for large embedding data
    let migrated = 0;

    for (let offset = 0; offset < totalEmb; offset += batchSize) {
      const batch = await pgDb.query(
        `SELECT chunk_id, embedding::text as embedding 
         FROM embeddings 
         ORDER BY chunk_id 
         LIMIT ${batchSize} OFFSET ${offset}`
      );

      const statements: Array<{ sql: string; args: [string, string] }> = [];

      for (const row of batch.rows) {
        try {
          const embRow = row as { chunk_id: string; embedding: string };

          // Parse pgvector text format → number array
          const vectorArray = parseVector(embRow.embedding);

          // Validate dimension
          if (vectorArray.length !== EMBEDDING_DIM) {
            stats.skippedEmbeddings++;
            stats.errors.push(
              `Embedding ${embRow.chunk_id}: wrong dimension ${vectorArray.length}, expected ${EMBEDDING_DIM}`
            );
            continue;
          }

          // Convert to LibSQL format: JSON.stringify for F32_BLOB
          statements.push({
            sql: "INSERT INTO embeddings (chunk_id, embedding) VALUES (?, vector(?))",
            args: [embRow.chunk_id, JSON.stringify(vectorArray)],
          });
        } catch (e) {
          const embRow = row as { chunk_id: string };
          stats.skippedEmbeddings++;
          stats.errors.push(`Embedding ${embRow.chunk_id}: ${e}`);
        }
      }

      if (statements.length > 0) {
        try {
          await libsqlClient.batch(statements, "write");
          migrated += statements.length;
          stats.embeddings += statements.length;
        } catch (e) {
          const error = `Embedding batch at offset ${offset}: ${e}`;
          stats.errors.push(error);
          console.error(`  ✗ ${error}`);
        }
      }

      if (
        migrated % 500 === 0 ||
        migrated + stats.skippedEmbeddings >= totalEmb
      ) {
        console.log(
          `  Progress: ${migrated}/${totalEmb} (${stats.skippedEmbeddings} skipped)`
        );
      }
    }

    console.log(`  ✓ Migrated ${stats.embeddings} embeddings`);
    if (stats.skippedEmbeddings > 0) {
      console.log(
        `  ⚠ Skipped ${stats.skippedEmbeddings} embeddings (see errors)`
      );
    }
  } catch (e) {
    console.error(`  ✗ Failed to migrate embeddings: ${e}`);
    console.log("  (Embeddings can be regenerated after migration)");
    stats.errors.push(`Embeddings migration failed: ${e}`);
  }
}

/**
 * Verify migration succeeded
 */
async function verifyMigration(
  pgDb: PGlite,
  libsqlClient: ReturnType<typeof createClient>,
  stats: MigrationStats
): Promise<boolean> {
  console.log("\nVerifying migration...");

  try {
    // Check counts
    const pgDocs = await pgDb.query("SELECT COUNT(*) as c FROM documents");
    const pgChunks = await pgDb.query("SELECT COUNT(*) as c FROM chunks");
    const pgEmb = await pgDb.query("SELECT COUNT(*) as c FROM embeddings");

    const libsqlDocs = await libsqlClient.execute(
      "SELECT COUNT(*) as c FROM documents"
    );
    const libsqlChunks = await libsqlClient.execute(
      "SELECT COUNT(*) as c FROM chunks"
    );
    const libsqlEmb = await libsqlClient.execute(
      "SELECT COUNT(*) as c FROM embeddings"
    );

    const pgDocsCount = parseInt((pgDocs.rows[0] as { c: string }).c);
    const pgChunksCount = parseInt((pgChunks.rows[0] as { c: string }).c);
    const pgEmbCount = parseInt((pgEmb.rows[0] as { c: string }).c);

    const libsqlDocsCount = Number(
      (libsqlDocs.rows[0] as unknown as { c: number }).c
    );
    const libsqlChunksCount = Number(
      (libsqlChunks.rows[0] as unknown as { c: number }).c
    );
    const libsqlEmbCount = Number(
      (libsqlEmb.rows[0] as unknown as { c: number }).c
    );

    console.log("\nComparison:");
    console.log(`  Documents: ${pgDocsCount} → ${libsqlDocsCount}`);
    console.log(`  Chunks: ${pgChunksCount} → ${libsqlChunksCount}`);
    console.log(
      `  Embeddings: ${pgEmbCount} → ${libsqlEmbCount} (${stats.skippedEmbeddings} skipped)`
    );

    const docsMatch = pgDocsCount === libsqlDocsCount;
    const chunksMatch = pgChunksCount === libsqlChunksCount;
    const embMatch = pgEmbCount === libsqlEmbCount + stats.skippedEmbeddings;

    if (docsMatch && chunksMatch && embMatch) {
      console.log("\n✓ Verification passed - all counts match!");
      return true;
    } else {
      console.log("\n✗ Verification failed - count mismatch");
      if (!docsMatch) console.log("  - Documents don't match");
      if (!chunksMatch) console.log("  - Chunks don't match");
      if (!embMatch) console.log("  - Embeddings don't match");
      return false;
    }
  } catch (e) {
    console.error(`\n✗ Verification error: ${e}`);
    return false;
  }
}

/**
 * Main migration workflow
 */
async function main() {
  console.log("=== PGlite to LibSQL Migration ===\n");
  console.log(`PGlite DB: ${pglitePath}`);
  console.log(`LibSQL DB: ${libsqlPath}`);

  // Check PGlite database exists
  if (!existsSync(pglitePath)) {
    console.error(`\nError: PGlite database not found at ${pglitePath}`);
    console.error("Run with correct path or use default location.");
    process.exit(1);
  }

  const stats: MigrationStats = {
    documents: 0,
    chunks: 0,
    embeddings: 0,
    skippedEmbeddings: 0,
    errors: [],
  };

  // Ensure LibSQL directory exists
  const libsqlDir = dirname(libsqlPath);
  if (!existsSync(libsqlDir)) {
    mkdirSync(libsqlDir, { recursive: true });
  }

  console.log("\nOpening PGlite database...");
  const pgDb = new PGlite(pglitePath, { extensions: { vector } });
  await pgDb.waitReady;
  console.log("✓ PGlite ready");

  console.log("\nOpening LibSQL database...");
  const libsqlClient = createClient({
    url: `file:${libsqlPath}`,
  });
  console.log("✓ LibSQL ready");

  try {
    // Initialize LibSQL schema
    await initLibSQLSchema(libsqlClient);

    // Migrate data
    await migrateDocuments(pgDb, libsqlClient, stats);
    await migrateChunks(pgDb, libsqlClient, stats);
    await migrateEmbeddings(pgDb, libsqlClient, stats);

    // Verify migration
    const verified = await verifyMigration(pgDb, libsqlClient, stats);

    // Summary
    console.log("\n=== Migration Summary ===");
    console.log(`Documents migrated: ${stats.documents}`);
    console.log(`Chunks migrated: ${stats.chunks}`);
    console.log(`Embeddings migrated: ${stats.embeddings}`);
    if (stats.skippedEmbeddings > 0) {
      console.log(`Embeddings skipped: ${stats.skippedEmbeddings}`);
    }
    console.log(`Errors: ${stats.errors.length}`);

    if (stats.errors.length > 0) {
      console.log("\nErrors:");
      stats.errors.slice(0, 10).forEach((err) => console.log(`  - ${err}`));
      if (stats.errors.length > 10) {
        console.log(`  ... and ${stats.errors.length - 10} more`);
      }
    }

    if (verified) {
      console.log("\n✓ Migration completed successfully!");
      console.log(`\nLibSQL database created at: ${libsqlPath}`);
      process.exit(0);
    } else {
      console.log("\n⚠ Migration completed with verification warnings");
      process.exit(1);
    }
  } catch (e) {
    console.error(`\nMigration failed: ${e}`);
    process.exit(1);
  } finally {
    await pgDb.close();
    libsqlClient.close();
  }
}

main().catch((e) => {
  console.error("Fatal error:", e);
  process.exit(1);
});
