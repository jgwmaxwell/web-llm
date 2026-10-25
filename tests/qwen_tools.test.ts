import { describe, expect, test } from "@jest/globals";
import { prepareQwenTools, parseQwenTools } from "../src/qwen_tools";
import type {
  ChatCompletionRequest,
  ChatCompletionTool,
} from "../src/openai_api_protocols/chat_completion";

const tools: ChatCompletionTool[] = [
  {
    type: "function",
    function: {
      name: "lookup",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string" },
          limit: { type: "integer" },
          exact: { type: "boolean" },
          options: { type: "object" },
          ids: { type: "array" },
        },
        required: ["query"],
      },
    },
  },
];
const output = `<tool_call>\n<function=lookup>\n<parameter=query>\n  Job.name\nwith newline  \n</parameter>\n<parameter=limit>\n5\n</parameter>\n</function>\n</tool_call>`;
const base: ChatCompletionRequest = {
  messages: [
    { role: "system", content: "Keep edits small." },
    { role: "user", content: "Find jobs" },
  ],
  tools,
};

describe("Qwen native tools", () => {
  test("accepts normal answers, typed arguments and multiple calls", () => {
    expect(parseQwenTools("Hello", base)).toEqual({ content: "Hello" });
    expect(parseQwenTools("<think>\n\n</think>\n\nHello", base)).toEqual({
      content: "Hello",
    });
    expect(
      parseQwenTools("<think>\n\n</think>\n\n" + output, base).content,
    ).toBeNull();
    const parsed = parseQwenTools(
      "I will look that up.\n" + output + "\n" + output,
      base,
    );
    expect(parsed.content).toBe("I will look that up.");
    expect(parsed.tool_calls).toHaveLength(2);
    expect(parsed.tool_calls![0].id).not.toBe(parsed.tool_calls![1].id);
    expect(JSON.parse(parsed.tool_calls![0].function.arguments)).toEqual({
      query: "  Job.name\nwith newline  ",
      limit: 5,
    });
  });
  test.each([
    output.slice(0, -5),
    output + "trailing prose",
    output.replace("lookup", "execute"),
    output.replace("parameter=query", "parameter=unknown"),
    output.replace("\n5\n", "\nno\n"),
    output.replace("parameter=query", "parameter=limit"),
    "<tool_call",
    "<function=lookup>",
    "<tool_call><function=lookup></function></tool_call>",
  ])(
    "rejects malformed, missing, duplicate or unadvertised arguments: %s",
    (raw) => {
      expect(() => parseQwenTools(raw, base)).toThrow();
    },
  );
  test("enforces tool choice and never forces JSON mode", () => {
    expect(() =>
      parseQwenTools(output, { ...base, tool_choice: "none" }),
    ).toThrow();
    expect(() =>
      parseQwenTools("Hello", { ...base, tool_choice: "required" }),
    ).toThrow();
    expect(() =>
      parseQwenTools(output, {
        ...base,
        tool_choice: { type: "function", function: { name: "other" } },
      }),
    ).toThrow();
    expect(prepareQwenTools(base).response_format).toBeUndefined();
    expect(
      prepareQwenTools({ ...base, tool_choice: "none" }).messages[0].content,
    ).not.toContain("<tools>");
    expect(() =>
      prepareQwenTools({ ...base, extra_body: { enable_thinking: true } }),
    ).toThrow();
  });
  test("roundtrips assistant calls and grouped tool responses without mutating reusable input", () => {
    const calls = parseQwenTools(output + output, base).tool_calls!;
    const request: ChatCompletionRequest = {
      ...base,
      messages: [
        ...base.messages,
        { role: "assistant", content: null, tool_calls: calls },
        {
          role: "tool",
          tool_call_id: calls[0].id,
          content: "Job.name: String",
        },
        { role: "tool", tool_call_id: calls[1].id, content: "Job.id: ID!" },
      ],
    };
    const before = JSON.stringify(request);
    const wire = prepareQwenTools(request);
    expect(wire.messages[0].content).toContain(
      "</IMPORTANT>\n\nKeep edits small.",
    );
    expect(wire.messages[2].content).toContain(
      "<think>\n\n</think>\n\n<tool_call>",
    );
    expect(wire.messages[3]).toEqual({
      role: "user",
      content:
        "<tool_response>\nJob.name: String\n</tool_response>\n<tool_response>\nJob.id: ID!\n</tool_response>",
    });
    expect(wire.messages).toHaveLength(4);
    expect(JSON.stringify(request)).toBe(before);
    expect(prepareQwenTools(request)).toEqual(wire);
    expect(() =>
      prepareQwenTools({ ...request, messages: request.messages.slice(0, -1) }),
    ).toThrow();
    expect(() =>
      prepareQwenTools({
        ...base,
        messages: [
          ...base.messages,
          { role: "tool", tool_call_id: "orphan", content: "x" },
        ],
      }),
    ).toThrow();
  });
});
