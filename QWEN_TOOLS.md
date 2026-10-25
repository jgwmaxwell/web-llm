# Qwen3.5 tool protocol experiment

Based on upstream v0.2.85, for Qwen3.5-4B-q4f16_1-MLC only. The model weights
and WASM remain upstream artifacts. No second model or custom compilation.

Pass ordinary OpenAI-shaped `tools`, and `tool_choice` (`auto`, `none`, `required`,
or a named function), to `engine.chat.completions.create`. The adapter uses the
Qwen3.5 tool prompt and XML-like call syntax, returning structured `tool_calls`.
Feed the assistant message and matching `role: tool` results back for the next
turn. Consecutive results are grouped into Qwen's user-role tool-response block.
Caller-owned request/history objects are preserved.

Scope and limitations:

- Text-only, non-thinking mode. Thinking explicitly enabled is rejected.
- Streaming is buffered until completion. Normal answers return content; calls
  return complete structured arguments with indices in the final delta.
- Length/abort never expose partial calls. Malformed calls reject the request.
- Unknown tools/parameters, duplicate parameters and missing required arguments
  are rejected. Primitive parameter types are decoded using their declared type.
  The host must still perform full JSON Schema and authorization validation.
- Required/named tool choice is prompted and checked, not constrained decoding.
- Delimiter sequences in argument strings are rejected, not guessed or escaped
  using a format the model was not trained on.
- No changes to host discovery, tool execution, user approval or workbench logic.

Reference template: https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja

Validation: `pnpm exec jest --runInBand --coverage`, `pnpm build`.
Use `pnpm exec eslint src/qwen_tools.ts src/engine.ts src/config.ts
src/openai_api_protocols/chat_completion.ts tests/qwen_tools.test.ts
tests/engine_integration.test.ts` for changed files.
