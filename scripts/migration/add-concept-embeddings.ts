#!/usr/bin/env bun
/**
 * Generate embeddings for all concepts in the taxonomy
 *
 * This script:
 * 1. Adds the concept_embeddings table if it doesn't exist
 * 2. Generates embeddings for all concepts using Ollama mxbai-embed-large
 * 3. Stores embeddings in the same vector space as document chunks
 *
 * Prerequisites:
 *   - Ollama running: ollama serve
 *   - Embedding model: ollama pull mxbai-embed-large
 *
 * Usage:
 *   bun run scripts/migration/add-concept-embeddings.ts [db-path]
 *
 * Environment:
 *   OLLAMA_HOST - Ollama server URL (default: http://localhost:11434)
 */

import { createClient } from "@libsql/client";
import { join } from "path";

const OLLAMA_HOST = process.env.OLLAMA_HOST || "http://localhost:11434";
const OLLAMA_MODEL = "mxbai-embed-large"; // Hardcoded - must match document embeddings
const EMBEDDING_DIM = 1024; // mxbai-embed-large dimension
const BATCH_SIZE = 5; // Process 5 concepts at a time

const args = process.argv.slice(2);
const dbPath =
  args[0] ||
  `file:${join(process.env.HOME!, "Documents/.pdf-library/library.db")}`;

interface EmbeddingResponse {
  embedding: number[];
}

interface Concept {
  id: string;
  pref_label: string;
  definition: string | null;
}

async function getEmbedding(text: string): Promise<number[]> {
  const response = await fetch(`${OLLAMA_HOST}/api/embeddings`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: OLLAMA_MODEL, prompt: text }),
  });

  if (!response.ok) {
    throw new Error(`Ollama error: ${response.status} ${response.statusText}`);
  }

  const data = (await response.json()) as EmbeddingResponse;

  // Validate dimension
  if (data.embedding.length !== EMBEDDING_DIM) {
    throw new Error(
      `Expected ${EMBEDDING_DIM} dimensions, got ${data.embedding.length}`
    );
  }

  return data.embedding;
}

async function checkOllama(): Promise<boolean> {
  try {
    const response = await fetch(`${OLLAMA_HOST}/api/tags`);
    if (!response.ok) return false;

    const data = await response.json();
    const hasModel = data.models.some(
      (m: { name: string }) =>
        m.name === OLLAMA_MODEL || m.name.startsWith(`${OLLAMA_MODEL}:`)
    );

    if (!hasModel) {
      console.error(`Model ${OLLAMA_MODEL} not found.`);
      console.error(`Run: ollama pull ${OLLAMA_MODEL}`);
      return false;
    }

    // Test embedding generation
    const testEmb = await getEmbedding("test");
    console.log(
      `Ollama ready! Model: ${OLLAMA_MODEL}, Dimension: ${testEmb.length}`
    );
    return true;
  } catch {
    return false;
  }
}

async function main() {
  console.log("=== Concept Embeddings Migration ===\n");
  console.log(`Database: ${dbPath}`);
  console.log(`Ollama: ${OLLAMA_HOST}`);
  console.log(`Model: ${OLLAMA_MODEL}\n`);

  // Check Ollama
  console.log("Checking Ollama...");
  if (!(await checkOllama())) {
    console.error("\nOllama not available. Make sure it's running:");
    console.error("  ollama serve");
    console.error(`  ollama pull ${OLLAMA_MODEL}`);
    process.exit(1);
  }

  // Connect to database
  console.log("\nConnecting to database...");
  const client = createClient({ url: dbPath });

  // Step 1: Create concept_embeddings table if it doesn't exist
  console.log("Creating concept_embeddings table...");
  await client.execute(`
    CREATE TABLE IF NOT EXISTS concept_embeddings (
      concept_id TEXT PRIMARY KEY REFERENCES concepts(id) ON DELETE CASCADE,
      embedding F32_BLOB(${EMBEDDING_DIM}) NOT NULL
    )
  `);

  // Create vector index
  await client.execute(`
    CREATE INDEX IF NOT EXISTS concept_embeddings_idx 
    ON concept_embeddings(libsql_vector_idx(embedding, 'compress_neighbors=float8'))
  `);

  console.log("Schema ready!");

  // Count concepts needing embeddings
  const countResult = await client.execute(`
    SELECT COUNT(*) as count FROM concepts c
    LEFT JOIN concept_embeddings e ON e.concept_id = c.id
    WHERE e.concept_id IS NULL
  `);
  const total = Number((countResult.rows[0] as any).count || 0);

  console.log(`\nConcepts needing embeddings: ${total}`);

  if (total === 0) {
    console.log("All concepts have embeddings!");
    client.close();
    return;
  }

  // Estimate time
  const estimatedMinutes = Math.ceil(total / 15 / 60); // ~15 embeddings/sec
  console.log(`Estimated time: ~${estimatedMinutes} minutes\n`);

  let processed = 0;
  let errors = 0;
  const startTime = Date.now();

  while (processed < total) {
    // Get batch of concepts without embeddings
    const batch = await client.execute(`
      SELECT c.id, c.pref_label, c.definition FROM concepts c
      LEFT JOIN concept_embeddings e ON e.concept_id = c.id
      WHERE e.concept_id IS NULL
      LIMIT ${BATCH_SIZE}
    `);

    if (batch.rows.length === 0) break;

    // Generate embeddings for batch
    for (const row of batch.rows) {
      const concept = row as unknown as Concept;
      try {
        // Create text: "prefLabel: definition" or just "prefLabel"
        const text = concept.definition
          ? `${concept.pref_label}: ${concept.definition}`
          : concept.pref_label;

        const embedding = await getEmbedding(text);

        // Store using vector32() function
        await client.execute({
          sql: `INSERT INTO concept_embeddings (concept_id, embedding)
               VALUES (?, vector32(?))
               ON CONFLICT (concept_id) DO UPDATE SET
                 embedding = excluded.embedding`,
          args: [concept.id, JSON.stringify(embedding)],
        });

        processed++;
      } catch (e) {
        errors++;
        console.error(`Error on concept ${concept.id}: ${e}`);
      }
    }

    // Progress update every 10 concepts or at completion
    if (processed % 10 === 0 || processed === total) {
      const elapsed = (Date.now() - startTime) / 1000;
      const rate = processed / elapsed;
      const remaining = (total - processed) / rate;
      const pct = ((processed / total) * 100).toFixed(1);

      console.log(
        `Progress: ${processed}/${total} (${pct}%) | ` +
          `${rate.toFixed(1)}/s | ` +
          `ETA: ${Math.ceil(remaining / 60)}min`
      );
    }
  }

  // Final stats
  const embResult = await client.execute(
    "SELECT COUNT(*) as count FROM concept_embeddings"
  );
  const finalCount = Number((embResult.rows[0] as any).count || 0);

  console.log("\n=== Complete ===");
  console.log(`Embeddings generated: ${processed}`);
  console.log(`Total concept embeddings: ${finalCount}`);
  console.log(`Errors: ${errors}`);
  console.log(
    `Time: ${((Date.now() - startTime) / 1000 / 60).toFixed(1)} minutes`
  );

  client.close();
}

main().catch((e) => {
  console.error("Migration failed:", e);
  process.exit(1);
});
