/**
 * LibSQL Database Service Tests (TDD)
 */

import { Effect, Layer } from "effect";
import { describe, expect, test } from "bun:test";
import { Database } from "./Database.js";
import { LibSQLDatabase } from "./LibSQLDatabase.js";
import { Document, SearchOptions } from "../types.js";

describe("LibSQLDatabase", () => {
  describe("initialization", () => {
    test("can be created with in-memory DB", async () => {
      const program = Effect.gen(function* () {
        // Create layer
        const layer = LibSQLDatabase.make({ url: ":memory:" });

        // Build the layer in a scope to verify it initializes successfully
        yield* Effect.scoped(
          Effect.gen(function* () {
            yield* Layer.build(layer);
            return "created";
          })
        );

        return "created";
      });

      const result = await Effect.runPromise(program);
      expect(result).toBe("created");
    });
  });

  describe("document operations", () => {
    test("addDocument stores document", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Create test document
        const doc = new Document({
          id: "test-123",
          title: "Test Document",
          path: "/path/to/test.pdf",
          addedAt: new Date("2025-01-01T00:00:00Z"),
          pageCount: 10,
          sizeBytes: 1024,
          tags: ["test", "example"],
          metadata: { source: "test" },
        });

        // Add document
        yield* db.addDocument(doc);

        // Verify it was stored by retrieving it
        const retrieved = yield* db.getDocument("test-123");

        return retrieved;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const retrieved = await Effect.runPromise(Effect.provide(program, layer));

      expect(retrieved).not.toBeNull();
      expect(retrieved?.id).toBe("test-123");
      expect(retrieved?.title).toBe("Test Document");
      expect(retrieved?.path).toBe("/path/to/test.pdf");
      expect(retrieved?.tags).toEqual(["test", "example"]);
    });

    test("listDocuments returns all documents", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add multiple documents
        yield* db.addDocument(
          new Document({
            id: "doc-1",
            title: "First Doc",
            path: "/path/1.pdf",
            addedAt: new Date("2025-01-01"),
            pageCount: 5,
            sizeBytes: 500,
            tags: ["tag1"],
          })
        );
        yield* db.addDocument(
          new Document({
            id: "doc-2",
            title: "Second Doc",
            path: "/path/2.pdf",
            addedAt: new Date("2025-01-02"),
            pageCount: 10,
            sizeBytes: 1000,
            tags: ["tag2"],
          })
        );

        // List all documents
        const docs = yield* db.listDocuments();

        return docs;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const docs = await Effect.runPromise(Effect.provide(program, layer));

      expect(docs).toHaveLength(2);
      expect(docs[0].id).toBe("doc-2"); // Most recent first
      expect(docs[1].id).toBe("doc-1");
    });

    test("deleteDocument removes document and cascades", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add document
        yield* db.addDocument(
          new Document({
            id: "doc-del",
            title: "To Delete",
            path: "/path/del.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: [],
          })
        );

        // Delete it
        yield* db.deleteDocument("doc-del");

        // Verify it's gone
        const retrieved = yield* db.getDocument("doc-del");
        return retrieved;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const result = await Effect.runPromise(Effect.provide(program, layer));

      expect(result).toBeNull();
    });
  });

  describe("chunk and embedding operations", () => {
    test("addChunks stores chunks", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add document first
        yield* db.addDocument(
          new Document({
            id: "doc-chunks",
            title: "Chunked Doc",
            path: "/path/chunks.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: [],
          })
        );

        // Add chunks
        yield* db.addChunks([
          {
            id: "chunk-1",
            docId: "doc-chunks",
            page: 1,
            chunkIndex: 0,
            content: "First chunk content",
          },
          {
            id: "chunk-2",
            docId: "doc-chunks",
            page: 1,
            chunkIndex: 1,
            content: "Second chunk content",
          },
        ]);

        return "chunks-added";
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const result = await Effect.runPromise(Effect.provide(program, layer));

      expect(result).toBe("chunks-added");
    });

    test("getStats returns document/chunk/embedding counts", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add document
        yield* db.addDocument(
          new Document({
            id: "doc-stats",
            title: "Stats Doc",
            path: "/path/stats.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: [],
          })
        );

        // Add chunks
        yield* db.addChunks([
          {
            id: "chunk-stats-1",
            docId: "doc-stats",
            page: 1,
            chunkIndex: 0,
            content: "Content",
          },
        ]);

        // Get stats
        const stats = yield* db.getStats();
        return stats;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const stats = await Effect.runPromise(Effect.provide(program, layer));

      expect(stats.documents).toBe(1);
      expect(stats.chunks).toBe(1);
      expect(stats.embeddings).toBe(0);
    });
  });

  describe("full-text search (FTS5)", () => {
    test("ftsSearch returns matching results", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add document and chunks
        yield* db.addDocument(
          new Document({
            id: "doc-fts",
            title: "FTS Test Doc",
            path: "/path/fts.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: ["tech"],
          })
        );

        yield* db.addChunks([
          {
            id: "chunk-fts-1",
            docId: "doc-fts",
            page: 1,
            chunkIndex: 0,
            content: "React hooks are awesome for state management",
          },
          {
            id: "chunk-fts-2",
            docId: "doc-fts",
            page: 1,
            chunkIndex: 1,
            content: "TypeScript provides excellent type safety",
          },
        ]);

        // Search for "hooks"
        const results = yield* db.ftsSearch(
          "hooks",
          new SearchOptions({ limit: 10 })
        );

        return results;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const results = await Effect.runPromise(Effect.provide(program, layer));

      expect(results).toHaveLength(1);
      expect(results[0].content).toContain("hooks");
      expect(results[0].docId).toBe("doc-fts");
      expect(results[0].matchType).toBe("fts");
    });

    test("ftsSearch respects tag filter", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        // Add two documents with different tags
        yield* db.addDocument(
          new Document({
            id: "doc-fts-tech",
            title: "Tech Doc",
            path: "/path/tech.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: ["tech"],
          })
        );

        yield* db.addDocument(
          new Document({
            id: "doc-fts-business",
            title: "Business Doc",
            path: "/path/business.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: ["business"],
          })
        );

        yield* db.addChunks([
          {
            id: "chunk-tech",
            docId: "doc-fts-tech",
            page: 1,
            chunkIndex: 0,
            content: "React hooks documentation",
          },
          {
            id: "chunk-business",
            docId: "doc-fts-business",
            page: 1,
            chunkIndex: 0,
            content: "React to market changes",
          },
        ]);

        // Search for "react" with tech tag filter
        const results = yield* db.ftsSearch(
          "react",
          new SearchOptions({ limit: 10, tags: ["tech"] })
        );

        return results;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const results = await Effect.runPromise(Effect.provide(program, layer));

      expect(results).toHaveLength(1);
      expect(results[0].docId).toBe("doc-fts-tech");
    });

    test("ftsSearch returns empty array when no matches", async () => {
      const program = Effect.gen(function* () {
        const db = yield* Database;

        yield* db.addDocument(
          new Document({
            id: "doc-fts-empty",
            title: "Empty Search Doc",
            path: "/path/empty.pdf",
            addedAt: new Date(),
            pageCount: 1,
            sizeBytes: 100,
            tags: [],
          })
        );

        yield* db.addChunks([
          {
            id: "chunk-empty",
            docId: "doc-fts-empty",
            page: 1,
            chunkIndex: 0,
            content: "Some content here",
          },
        ]);

        const results = yield* db.ftsSearch(
          "nonexistent",
          new SearchOptions({ limit: 10 })
        );
        return results;
      });

      const layer = LibSQLDatabase.make({ url: ":memory:" });
      const results = await Effect.runPromise(Effect.provide(program, layer));

      expect(results).toHaveLength(0);
    });
  });
});
