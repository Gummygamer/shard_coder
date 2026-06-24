# ShardCoder

ShardCoder is a local-first coding agent for small-context LLMs (such as
Gemma 4) that edits code by decomposing tasks into compact, retrievable,
testable code shards. It is designed to compensate for the limitations of
small models — short context windows, weaker long-horizon reasoning, and
sensitivity to prompt noise — through retrieval, summarisation, narrow
prompts, deterministic tools, validators, and iterative refinement.

It runs entirely against any OpenAI-compatible local model server such as
Ollama, llama.cpp, LM Studio or vLLM. Web search is disabled by default and
external documentation, when used, is always compressed into compact notes
before being shown to the model.

## Why small models?

Small local models are private, cheap and fast, but require a different
agent architecture. Frontier-style "give the model the whole repo and let
it think" prompts overflow their context and produce unreliable patches.
ShardCoder treats the model as a focused reasoning component:

- tools collect facts (scanner, runner, git status, retrieval)
- retrieval picks only the relevant snippets
- summaries compress files into structured metadata
- prompts are narrow, schema-driven, and repeat the goal
- outputs are validated (JSON, unified diffs)
- patches are small and applied carefully
- tests are run after every meaningful change
- recent work is compressed into rolling memory

## Install

```bash
git clone <this repo>
cd shardcoder
pip install -e ".[dev]"
```

## Run a local OpenAI-compatible server

ShardCoder talks to any server that exposes the `chat/completions`
endpoint. Common options:

- **Ollama** — `ollama serve`, then `ollama pull gemma:4b`
  (the OpenAI-compatible URL is `http://localhost:11434/v1`).
- **llama.cpp** — `./server -m <model>.gguf --port 8080`
  (URL: `http://localhost:8080/v1`).
- **LM Studio** — start the local server in the UI
  (URL: `http://localhost:1234/v1`).
- **vLLM** — `vllm serve <model>` (URL: `http://localhost:8000/v1`).

Copy `shardcoder.toml.example` to `shardcoder.toml` in your project and
edit `[llm]` to point at the server you started.

## Configure LM Studio

```toml
[llm]
base_url = "http://localhost:1234/v1"
model = "auto"
max_context_tokens = 8192
max_output_tokens = 1024
temperature = 0.1
```

In LM Studio, load a model, open the local server tab, and start the server.
The default `model = "auto"` asks `/v1/models` and uses the first loaded
model advertised by the server. If you want to pin a model explicitly, replace
`auto` with the exact model id shown by LM Studio.

Check the connection before running an agent task:

```bash
shardcoder llm-check
```

ShardCoder doesn't bake in model-specific behaviour — any OpenAI-compatible
local model will work.

## Index a repository

```bash
shardcoder index .
```

This walks the tree (skipping `.git`, `node_modules`, build directories,
and binaries), extracts rough symbols using regex (no native compilers
required), and stores file metadata + content hashes in a local SQLite
database under `.shardcoder/`.

## Summarise a repository

```bash
shardcoder summarize .
```

For each indexed file, the local model is asked for a compact JSON
summary (purpose, symbols, side effects, likely edit points). Files with
unchanged content hashes are skipped.

## Ask questions

```bash
shardcoder ask "explain the auth flow"
```

Builds a context pack and asks the local model. Cheap and read-only.

## Plan a task

```bash
shardcoder plan "add rate limiting to the public API"
```

Asks the local model for a strict JSON plan that decomposes the task into
small subtasks with associated files, validation commands and budget.

## Edit code

```bash
shardcoder edit "fix the failing login test"
shardcoder edit "fix the FastAPI lifespan warning" --web
```

The MVP executes the first subtask of the plan, builds a focused context
pack, asks the model for a unified diff, validates and (if allowed)
applies it, then runs the configured validation command. On failure it
retrieves a focused context for the failure and asks for a repair patch,
up to `max_iterations`.

## Dry-run mode

`dry_run = true` in `[agent]` (the default) prints the diff and validation
plan but never writes to disk and never runs the test command. Combine
with `--auto-apply=false` for the safest experience.

## Safety checks

- The agent refuses to operate on a dirty git tree unless `--allow-dirty`
  is passed.
- Diffs that touch ignored directories, vendored paths or binary files are
  rejected before they touch the filesystem.
- Suspicious shell commands (`rm -rf`, `sudo`, piping to `sh`, force git
  resets, deploy/publish commands) are blocked unless explicitly allowed
  in the safety configuration.
- Patches that include prose around the diff or fail to match context
  lines are rejected.
- The model is instructed to return `NO_PATCH: <reason>` when it lacks
  context, instead of hallucinating.

## Optional web / external documentation retrieval

External documentation is opt-in:

```bash
shardcoder edit "fix the FastAPI lifespan warning" --web
```

Web search is disabled by default (`web.enabled = false`). When enabled:

1. The planner decides whether external docs are actually needed.
2. ShardCoder generates 1–3 precise queries.
3. At most `max_fetched_pages` pages are retrieved.
4. Each page is **compressed** into a structured note (`source_title`,
   `source_url`, `date_accessed`, `relevance`, `facts`).
5. Notes are ranked (official docs preferred) and the top notes are
   included in the context pack under a strict `max_web_context_tokens`
   budget.
6. Raw HTML or full pages **never** enter the model context.

The MVP ships with a safe stub backend (`backend = "disabled"`) — it
returns a clear error rather than performing real network requests. Plug
in your own backend by implementing `external_context.web_search.WebBackend`.

## Limitations

- Symbol extraction is regex-based; tree-sitter is not bundled.
- Token counts are approximate (heuristic ~4 chars per token).
- Embedding-based retrieval is not bundled — keyword/symbol/path/import
  retrieval only.
- Only the first subtask of a plan is executed in the MVP.
- The bundled web backend is a safe stub; you must wire up your own
  search/fetch backend to actually retrieve external pages.

## Roadmap

- Tree-sitter symbol extraction.
- Optional embedding store for semantic retrieval.
- Full multi-subtask plan execution with checkpoints.
- A default `WebBackend` that uses a configurable search API.
- Per-language adapters (Go, Rust, TS) for richer summaries.
