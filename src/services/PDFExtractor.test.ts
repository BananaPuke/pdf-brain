/**
 * PDFExtractor Unit Tests
 */

import { describe, expect, test } from "bun:test";
import { sanitizeText } from "./PDFExtractor.js";

// ============================================================================
// sanitizeText() Tests
// ============================================================================

describe("sanitizeText", () => {
  test("strips null bytes from text", () => {
    const input = "Hello\x00World\x00!";
    const result = sanitizeText(input);
    expect(result).toBe("HelloWorld!");
  });

  test("strips multiple consecutive null bytes", () => {
    const input = "Text\x00\x00\x00with\x00\x00nulls";
    const result = sanitizeText(input);
    expect(result).toBe("Textwithnulls");
  });

  test("handles text with no null bytes", () => {
    const input = "Clean text";
    const result = sanitizeText(input);
    expect(result).toBe("Clean text");
  });

  test("handles empty string", () => {
    const input = "";
    const result = sanitizeText(input);
    expect(result).toBe("");
  });

  test("handles string with only null bytes", () => {
    const input = "\x00\x00\x00";
    const result = sanitizeText(input);
    expect(result).toBe("");
  });

  test("preserves unicode characters", () => {
    const input = "café\x00naïve\x00résumé";
    const result = sanitizeText(input);
    expect(result).toBe("cafénaïverésumé");
  });

  test("strips null bytes before other processing", () => {
    // Verify that null bytes are removed early in the pipeline
    const input = "Text\x00with\x00null\x00bytes";
    const result = sanitizeText(input);
    // Should not contain null bytes
    expect(result).not.toContain("\x00");
    // Should preserve the rest
    expect(result).toBe("Textwithnullbytes");
  });
});
