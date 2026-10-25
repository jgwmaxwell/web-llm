import type {
  ChatCompletionRequest,
  ChatCompletionMessageParam,
  ChatCompletionMessageToolCall,
  ChatCompletionTool,
} from "./openai_api_protocols/chat_completion";

// Deliberately narrow until other compiled Qwen profiles have been evaluated.
export const qwenToolModelIds = ["Qwen3.5-4B-q4f16_1-MLC"];
export const isQwenToolModel = (model: string): boolean =>
  qwenToolModelIds.includes(model);

const instructions = `# Tools

You have access to the following functions:

<tools>
{tools}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>`;

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function identifier(value: string): string {
  if (!/^[a-zA-Z_][a-zA-Z0-9_-]*$/.test(value))
    throw new Error("Invalid Qwen tool or parameter name");
  return value;
}
function encodeCall(call: ChatCompletionMessageToolCall): string {
  const args: unknown = JSON.parse(call.function.arguments);
  if (!object(args)) throw new Error("Tool arguments must be an object");
  const parameters = Object.entries(args).map(([name, value]) => {
    const text = typeof value === "string" ? value : JSON.stringify(value);
    if (/<\/?(?:tool_call|function|parameter)(?:[=>]|$)/.test(text))
      throw new Error("Reserved Qwen delimiters in tool arguments");
    return `<parameter=${identifier(name)}>\n${text}\n</parameter>\n`;
  });
  return `<tool_call>\n<function=${identifier(call.function.name)}>\n${parameters.join("")}</function>\n</tool_call>`;
}

/** Build a private wire request. Never mutate the caller's reusable history. */
export function prepareQwenTools(
  request: ChatCompletionRequest,
): ChatCompletionRequest {
  if (request.response_format)
    throw new Error("Qwen tools cannot use a separate response_format");
  if (request.extra_body?.enable_thinking === true)
    throw new Error("Qwen tools currently support non-thinking mode only");
  const tools = request.tools ?? [];
  const names = tools.map((tool) => identifier(tool.function.name));
  if (new Set(names).size !== names.length)
    throw new Error("Duplicate tool name");
  const choice = request.tool_choice ?? "auto";
  if (typeof choice === "object" && !names.includes(choice.function.name))
    throw new Error("Requested tool is not available");
  if (tools.length === 0 && choice !== "auto" && choice !== "none")
    throw new Error("Tool choice requires tools");
  let system =
    tools.length && choice !== "none"
      ? instructions.replace(
          "{tools}",
          tools.map((tool) => JSON.stringify(tool)).join("\n"),
        )
      : "";
  if (choice === "required")
    system += "\nYou must call at least one provided function.";
  if (typeof choice === "object")
    system += `\nYou must call ${choice.function.name}.`;
  if (choice === "none")
    system += "Do not call any functions. Respond with ordinary text.";
  const messages: ChatCompletionMessageParam[] = [];
  const pending = new Set<string>();
  let lastUser = -1;
  request.messages.forEach((message, i) => {
    if (message.role === "user") lastUser = i;
  });
  if (lastUser < 0) throw new Error("Qwen tools require a user query");
  request.messages.forEach((message, i) => {
    if (message.role !== "tool" && pending.size)
      throw new Error("Missing tool responses");
    if (message.role === "system") {
      if (i !== 0) throw new Error("System message must be first");
      system += (system ? "\n\n" : "") + message.content.trim();
    } else if (message.role === "assistant") {
      let content = (message.content ?? "").trim();
      if (content.includes("</think>"))
        content = content
          .split("</think>")
          .slice(1)
          .join("</think>")
          .trimStart();
      for (const call of message.tool_calls ?? []) {
        if (!call.id || pending.has(call.id))
          throw new Error("Duplicate or missing tool call ID");
        pending.add(call.id);
        content += (content.trim() ? "\n\n" : "") + encodeCall(call);
      }
      if (i > lastUser) content = "<think>\n\n</think>\n\n" + content;
      messages.push({ role: "assistant", content });
    } else if (message.role === "tool") {
      if (!pending.delete(message.tool_call_id))
        throw new Error("Unmatched tool response");
      const text = `<tool_response>\n${message.content.trim()}\n</tool_response>`;
      const previous = messages[messages.length - 1];
      if (
        i > 0 &&
        request.messages[i - 1].role === "tool" &&
        previous?.role === "user"
      )
        previous.content += "\n" + text;
      else messages.push({ role: "user", content: text });
    } else if (message.role === "user" && typeof message.content === "string") {
      messages.push({ role: "user", content: message.content.trim() });
    } else throw new Error("Qwen tools currently support text messages only");
  });
  if (pending.size) throw new Error("Missing tool responses");
  if (system) messages.unshift({ role: "system", content: system });
  return {
    ...request,
    messages,
    extra_body: { ...request.extra_body, enable_thinking: false },
  };
}

function parameterValue(raw: string, schema: unknown): unknown {
  if (!object(schema)) throw new Error("Missing parameter schema");
  // Qwen's template emits string arguments verbatim, structured values as JSON.
  const text = raw.replace(/^\r?\n/, "").replace(/\r?\n$/, "");
  if (schema.type === "string") return text;
  const value: unknown = JSON.parse(text);
  if (
    (schema.type === "integer" && !Number.isInteger(value)) ||
    (schema.type === "number" && typeof value !== "number") ||
    (schema.type === "boolean" && typeof value !== "boolean") ||
    (schema.type === "object" && !object(value)) ||
    (schema.type === "array" && !Array.isArray(value)) ||
    (schema.type === "null" && value !== null)
  )
    throw new Error("Invalid tool argument type");
  return value;
}

/** No calls are exposed until the entire stopped response has been checked. */
export function parseQwenTools(
  output: string,
  request: Pick<ChatCompletionRequest, "tools" | "tool_choice">,
): {
  content: string | null;
  tool_calls?: ChatCompletionMessageToolCall[];
} {
  // The compiled pipeline includes its empty non-thinking prefix in getMessage().
  output = output.replace(/^\s*<think>\s*<\/think>\s*/, "");
  const text = output.trim();
  const start = text.indexOf("<tool_call>");
  const choice = request.tool_choice ?? "auto";
  if (start < 0) {
    if (/<\/?(?:tool_call|function|parameter)(?:[=>]|$)/.test(text))
      throw new Error("Malformed Qwen tool call");
    if (choice === "required" || typeof choice === "object")
      throw new Error("Required tool call missing");
    return { content: output };
  }
  if (choice === "none") throw new Error("Tool call disallowed by tool_choice");
  const prefix = text.slice(0, start).trim();
  if (/<\/?(?:tool_call|function|parameter)(?:[=>]|$)/.test(prefix))
    throw new Error("Malformed Qwen tool prefix");
  const calls: ChatCompletionMessageToolCall[] = [];
  let remaining = text.slice(start);
  while (remaining) {
    const match =
      /^<tool_call>\s*<function=([\w-]+)>\s*([\s\S]*?)<\/function>\s*<\/tool_call>/.exec(
        remaining,
      );
    if (!match) throw new Error("Malformed Qwen tool call");
    const name = match[1];
    const tool: ChatCompletionTool | undefined = request.tools?.find(
      (t) => t.function.name === name,
    );
    if (!tool || (typeof choice === "object" && choice.function.name !== name))
      throw new Error("Unexpected tool name");
    const schema: unknown = tool.function.parameters;
    if (!object(schema))
      throw new Error("Tool needs an object parameter schema");
    const properties = object(schema.properties) ? schema.properties : {};
    const args: Record<string, unknown> = Object.create(null);
    let parameters = match[2].trimStart();
    while (parameters) {
      const p = /^<parameter=([\w-]+)>([\s\S]*?)<\/parameter>/.exec(parameters);
      if (!p || !Object.hasOwn(properties, p[1]) || Object.hasOwn(args, p[1]))
        throw new Error("Invalid or duplicate tool parameter");
      if (/<\/?(?:tool_call|function|parameter)(?:[=>]|$)/.test(p[2]))
        throw new Error("Reserved Qwen delimiters in parameter");
      args[p[1]] = parameterValue(p[2], properties[p[1]]);
      parameters = parameters.slice(p[0].length).trimStart();
    }
    if (
      Array.isArray(schema.required) &&
      schema.required.some(
        (key) => typeof key !== "string" || !Object.hasOwn(args, key),
      )
    )
      throw new Error("Missing required tool argument");
    calls.push({
      id: `call_${crypto.randomUUID()}`,
      type: "function",
      function: { name, arguments: JSON.stringify(args) },
    });
    remaining = remaining.slice(match[0].length).trimStart();
  }
  return { content: prefix || null, tool_calls: calls };
}
