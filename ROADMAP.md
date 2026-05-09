# ShardCoder roadmap

This document is the single source of truth for **what is built**, **what
is stubbed**, and **what to do next**. It is written so that a fresh
session — with no prior context — can pick up development immediately.

Work top-to-bottom; each section lists concrete file paths, anchors, and
acceptance criteria.

---

## 0. Orient yourself

```bash
cd /home/darabat/coding/shard_coder
.venv/bin/pytest               # baseline: 38/38 pass (33 MVP + 5 M1)
.venv/bin/shardcoder --help    # 7 commands
ls src/shardcoder/             # one subpackage per concern
```

Read in this order:

1. `README.md` — user-facing summary.
2. `src/shardcoder/agent.py` — orchestration; everything else hangs off it.
3. `src/shardcoder/cli.py` — exposed surface.
4. `src/shardcoder/config.py` — every setting, defaulted in one place.
5. `tests/` — these encode the contracts; do not regress them.

---

## 1. MVP status (done)

All of the following are implemented and tested:

| Module                              | Status | Notes |
| ----------------------------------- | ------ | ----- |
| `config.py`                         | Done   | TOML + env (`SHARDCODER_<SECTION>__<KEY>`) + CLI overrides. |
| `tokens/budget.py`                  | Done   | Heuristic 4-char/token, priority-based pruning, reserved sections. |
| `llm/openai_compatible.py`          | Done   | Synchronous httpx client; clear `LLMError` messages. |
| `indexing/scanner.py`               | Done   | os.walk, ignore dirs, binary detection, content-hash. |
| `indexing/symbols.py`               | Done   | Regex extractors for Python and JS/TS; generic fallback. |
| `summarization/store.py`            | Done   | SQLite (files / symbols / imports / summaries). |
| `summarization/summarizer.py`       | Done   | LLM summary with deterministic fallback. |
| `retrieval/{ranking,retriever,context_pack}.py` | Done | Hybrid retrieval + budget-aware pack assembly. |
| `planning/planner.py`               | Done   | Strict-JSON plan, single-subtask fallback. |
| `prompting/{templates,schemas}.py`  | Done   | All prompts + Pydantic validators. |
| `editing/{diff_utils,patcher}.py`   | Done   | Pure-Python parser; rejects prose, ignored paths, binaries, oversized deletions. |
| `execution/runner.py`               | Done   | subprocess + timeout + failing-path detection. |
| `memory/memory.py`                  | Done   | Sliding window with archive folding. |
| `safety/commands.py`                | Done   | Pattern-based dangerous-command detector. |
| `external_context/*`                | Done   | `DisabledBackend` stub; pluggable `WebBackend`; compression pipeline. |
| `agent.py`                          | Done   | Index → plan → context → patch → validate → repair, looped over every subtask in the plan with NO_PATCH / consecutive-failure stop guards. |
| `cli.py`                            | Done   | `index`, `summarize`, `ask`, `plan`, `edit`, `run-tests`, `status`. |

Tests: `tests/test_budget.py`, `test_scanner.py`, `test_context_pack.py`,
`test_patch_validation.py`, `test_external_context.py`, `test_agent_loop.py`.

---

## 2. Known stubs and limitations

These are deliberate gaps. **Read these before extending the system** so
you do not assume something works when it does not.

1. **Token estimation is heuristic** (`tokens/budget.py:_CHARS_PER_TOKEN`).
   Off by ±20% for code-heavy text. Acceptable for budgeting; not
   acceptable for tight context windows on large prompts.
2. **Symbol extraction is regex-based** (`indexing/symbols.py`). Misses:
   nested classes whose `class` keyword is in a string, decorated function
   ranges in oddly-formatted files, JS export-default-anonymous, Go/Rust/
   Java/Kotlin entirely (we record the file but no symbols).
3. **No embeddings** — retrieval is keyword/symbol/path/import only
   (`retrieval/retriever.py`).
4. ~~Plans are executed one subtask at a time~~ — **done in M1.**
   `agent.run_edit` walks every subtask, folding outcomes into
   `SlidingMemory` between subtasks. Stops early on `NO_PATCH` or two
   consecutive validation failures. CLI exposes `--max-subtasks N`.
5. **`WebBackend` ships as `DisabledBackend`**
   (`external_context/web_search.py`). Calling `search`/`fetch` raises
   `WebSearchUnavailable`. No real network code exists.
6. **Patch deletion path is conservative**
   (`editing/patcher.py:apply_validated_patch`). Currently it overwrites
   the file with `""` and only `unlink`s in narrow conditions; verify
   before relying on it for deletes.
7. **No formatter / linter integration** — `[validation]` exposes
   `lint_command` and `format_command`, but `agent._run_validation` only
   runs `test_command`.
8. **`shardcoder ask`** internally calls `agent._build_context_pack` (a
   private). It works but should move to a small public method on Agent.
9. **No telemetry on context pack** beyond `used_tokens` / `pruned`. We
   never record per-section sizes for tuning.
10. **No persistent run log** — every `edit` invocation starts with empty
    `SlidingMemory`. The class is ready for persistence; nothing wires it
    to disk.

---

## 3. Next milestones

Each milestone is sized to fit one focused session. Acceptance is a
green test plus the relevant CLI demo.

### M1 — Multi-subtask plan execution ✅ done

Shipped:
- `agent.run_edit` iterates `plan.subtasks`, folding each outcome into
  memory between iterations.
- New `AgentReport.stop_reason` field is set when the loop stops early.
- Early-stop guards: `NO_PATCH` from any subtask, or two consecutive
  hard validation failures.
- `cli.py:cmd_edit` exposes `--max-subtasks N` (min 1; default: all
  subtasks). When the cap truncates the plan, a warning is added.
- `_render_report` now walks every outcome and prints the stop reason.
- `tests/test_agent_loop.py` covers: two-subtask happy path, NO_PATCH
  stop, two-consecutive-failure stop, max-subtasks cap, single-failure
  recovery (does not stop).

### M2 — Real `WebBackend`

**Why**: `--web` is plumbing-only without a backend.

- Add `external_context/backends/searx.py` (or similar — pick one HTTP
  search API and a `readability-lxml`-style extractor).
- Wire it via `register_backend("searx", SearxBackend(...))`.
- Honour `[web].require_user_approval` and `max_fetched_pages`.
- Test with a recorded fixture (`tests/fixtures/searx_response.json`)
  via `respx` or a hand-rolled `httpx.MockTransport`.
- Update `README.md` "Optional web" section with the new backend slug.

### M3 — Tree-sitter symbols

**Why**: better symbols → better retrieval and summaries for free.

- Optional dep: `tree-sitter`, `tree-sitter-languages`.
- Replace `indexing/symbols.extract_python` /
  `extract_js_like` with tree-sitter when the dep is present; keep regex
  as fallback so the install stays slim.
- Add Go and Rust extractors.
- Existing `tests/test_scanner.py` should still pass; add language-
  specific cases.

### M4 — Formatter + linter in the loop

**Why**: small models often emit valid-but-ugly diffs.

- After a successful apply, run `[validation].format_command` (silent on
  success) and `[validation].lint_command` (treat non-zero like a test
  failure to trigger repair).
- Add `--no-format` / `--no-lint` flags.
- Test: in `test_agent_loop.py`, assert format command is invoked when
  configured.

### M5 — Persistent memory

**Why**: agents that forget every previous run cannot improve.

- Persist `SlidingMemory` to `.shardcoder/memory.jsonl`.
- Load on `Agent.__init__`; append on every `_update_memory` call.
- Add `shardcoder forget` to wipe it.
- New test: `tests/test_memory_persistence.py` round-trips entries.

### M6 — Embeddings (optional dep)

**Why**: keyword retrieval misses paraphrases.

- Add an `embeddings` extra: `sqlite-vec`, plus a small embedding
  model exposed by the same OpenAI-compatible server.
- Compute per-symbol or per-summary embeddings during `summarize`.
- Add a hybrid score: `0.6 * keyword + 0.4 * cosine`.
- Behaviour should degrade cleanly when the embedding model returns 404.

### M7 — Telemetry + tuning aids

**Why**: we can't tune budgets we don't measure.

- Add `agent.last_run_telemetry` with per-section sizes, prune list,
  iteration count, NO_PATCH reasons.
- `shardcoder status --telemetry` prints the last run's table.
- No new tests required, but add a smoke test that telemetry is
  populated.

### M8 — Distribution

**Why**: local users want `pip install shardcoder` to just work.

- Decide on the package name and release.
- Confirm minimum Python (3.11 is OK; pin in `pyproject.toml`).
- Add `Makefile` or `tox.ini` for `test`, `lint`, `format`, `release`.
- Wire `ruff` and `mypy` into a `pre-commit` config; CI on push.

---

## 4. Bug backlog (small, do alongside features)

Resolved (this milestone):
- ✅ `editing/patcher.py:apply_validated_patch` — deletion path now
  tracks `is_deleted_file` per pending write instead of re-deriving via
  `path.relative_to(path.anchor)`. Regression test:
  `test_patch_validation.py::test_diff_deleting_file_unlinks_it`.
- ✅ `cli.py:cmd_ask` no longer reaches into `_build_context_pack`; it
  calls the new public `Agent.build_context_for(task)`.

Still open:
- `agent._ask_for_patch` does not pass `stop=["NO_PATCH:"]`; consider
  adding stop tokens to keep small models from continuing past valid diff
  output.
- `summarization/summarizer._safe_json_parse` would benefit from
  reusing `prompting.schemas.extract_json` (currently duplicated).
- `cli.py:cmd_edit` writes `overrides["agent"] = {}` even when not
  needed, then pops it. Slightly clearer to build the dict only when
  there are real overrides.

---

## 5. Invariants — do not break these

If you change anything in the patch or context-pack code, make sure:

1. **Raw web pages never enter the context pack.** Only `CompactNote`
   instances may be rendered in `WEB NOTES:` (covered by
   `tests/test_external_context.py::test_raw_pages_are_never_passed_through`).
2. **Patches with prose around the diff are rejected**
   (`test_patch_validation.py::test_diff_with_prose_is_rejected`).
3. **Patches touching ignored / binary paths are rejected**
   (`test_diff_targeting_ignored_path_is_rejected`,
   `test_diff_targeting_binary_extension_is_rejected`).
4. **Reserved budget sections are never pruned**
   (`test_budget.py::test_critical_task_instructions_remain_when_overflowing`).
5. **Web search is disabled by default** (`config.WebConfig.enabled = False`).
6. **`apply_validated_patch(..., dry_run=True)` does not touch disk**
   (`test_patch_validation.py::test_dry_run_does_not_modify_files`).
7. **The agent refuses dirty git trees unless `--allow-dirty` is set**
   (`agent.run_edit` early return).

---

## 6. How to add a new module without breaking tests

1. Add the module under `src/shardcoder/<area>/<name>.py`.
2. Re-export public names from `src/shardcoder/<area>/__init__.py`.
3. Wire it into `agent.py` only after it is testable in isolation.
4. Add a `tests/test_<name>.py` that covers the happy path **and** one
   failure mode.
5. Run `.venv/bin/pytest`. Do not commit if anything regresses.
6. If the module is user-facing, add a CLI option in `cli.py` and a
   short README paragraph.

---

## 7. Quick demo a fresh session can run

```bash
# From inside a freshly cloned ShardCoder checkout:
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest                                       # 33 passed

# Try it on a sample repo without a real LLM:
mkdir -p /tmp/scdemo/pkg
printf 'def greet(name):\n    return "hi " + name\n' > /tmp/scdemo/app.py
printf 'def add(a, b):\n    return a + b\n'         > /tmp/scdemo/pkg/util.py
.venv/bin/shardcoder index /tmp/scdemo
.venv/bin/shardcoder summarize /tmp/scdemo --no-llm
.venv/bin/shardcoder status --repo /tmp/scdemo
```

The above is the **smoke test for any change you make** — if it stops
working, you have regressed something foundational.
