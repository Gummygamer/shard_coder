"""ShardCoder CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .agent import Agent, AgentReport
from .config import ConfigError, load_config
from .execution.runner import run_command, summarize_result
from .llm.client import ChatMessage, LLMError
from .llm.openai_compatible import OpenAICompatibleClient

app = typer.Typer(
    name="shardcoder",
    help="ShardCoder: a local-first coding agent for small-context LLMs.",
    no_args_is_help=True,
)
console = Console()


def _build_agent(
    repo: Path,
    config_path: Optional[Path],
    overrides: dict | None = None,
    use_llm: bool = True,
) -> Agent:
    try:
        config = load_config(
            config_path=str(config_path) if config_path else None,
            cli_overrides=overrides,
        )
    except ConfigError as exc:
        console.print(f"[red]Configuration error:[/red] {exc}")
        raise typer.Exit(2)
    llm = OpenAICompatibleClient(config.llm) if use_llm else None
    return Agent(repo_root=repo, config=config, llm=llm)


@app.command("index")
def cmd_index(
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Index a repository (rough symbols + content hashes)."""
    agent = _build_agent(repo, config, use_llm=False)
    added, skipped = agent.index(only_changed=True)
    console.print(f"Indexed: [green]{added}[/green] new/updated, [dim]{skipped} unchanged[/dim]")
    if verbose:
        for row in agent.store.all_files()[:50]:
            console.print(f"  {row['path']} [{row['language']}]")


@app.command("summarize")
def cmd_summarize(
    repo: Path = typer.Argument(Path("."), exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Use deterministic summaries only."),
) -> None:
    """Summarise files via the local model (skips unchanged files)."""
    agent = _build_agent(repo, config, use_llm=not no_llm)
    updated, skipped = agent.summarize(use_llm=not no_llm)
    console.print(
        f"Summarised [green]{updated}[/green] files, [dim]{skipped} unchanged[/dim]"
    )


@app.command("ask")
def cmd_ask(
    question: str = typer.Argument(...),
    repo: Path = typer.Option(Path("."), "--repo", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
    max_context: Optional[int] = typer.Option(None, "--max-context"),
) -> None:
    """Ask the local model a question with retrieved context."""
    overrides = (
        {"llm": {"max_context_tokens": max_context}} if max_context else None
    )
    agent = _build_agent(repo, config, overrides=overrides)
    pack = agent.build_context_for(question)
    if agent.llm is None:
        console.print("[red]No LLM configured[/red]")
        raise typer.Exit(2)
    try:
        result = agent.llm.chat(
            [
                ChatMessage(
                    role="system",
                    content="You answer questions about the codebase using only the provided context.",
                ),
                ChatMessage(
                    role="user",
                    content=f"Question: {question}\n\nContext:\n{pack.render()}",
                ),
            ],
            temperature=0.0,
            max_tokens=600,
        )
    except LLMError as exc:
        console.print(f"[red]LLM error:[/red] {exc}")
        raise typer.Exit(1)
    console.print(Panel(result.text, title=question))


@app.command("plan")
def cmd_plan(
    task: str = typer.Argument(...),
    repo: Path = typer.Option(Path("."), "--repo", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Show the JSON plan the model would execute for *task*."""
    agent = _build_agent(repo, config)
    result = agent.plan(task)
    table = Table(title=f"Plan ({'fallback' if result.fallback_used else 'model'})")
    table.add_column("ID")
    table.add_column("Goal")
    table.add_column("Files")
    table.add_column("Scope")
    table.add_column("Validation")
    table.add_column("Web?")
    for st in result.plan.subtasks:
        table.add_row(
            st.id,
            st.goal,
            ", ".join(st.likely_files) or "-",
            st.edit_scope,
            st.validation or "-",
            "yes" if st.needs_external_docs else "no",
        )
    console.print(table)


@app.command("edit")
def cmd_edit(
    task: str = typer.Argument(...),
    repo: Path = typer.Option(Path("."), "--repo", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    auto_apply: bool = typer.Option(False, "--auto-apply"),
    no_tests: bool = typer.Option(False, "--no-tests"),
    allow_dirty: bool = typer.Option(False, "--allow-dirty"),
    web: bool = typer.Option(False, "--web", help="Enable external doc retrieval."),
    no_web: bool = typer.Option(False, "--no-web"),
    require_web_approval: bool = typer.Option(
        False, "--require-web-approval"
    ),
    no_require_web_approval: bool = typer.Option(
        False, "--no-require-web-approval"
    ),
    max_subtasks: Optional[int] = typer.Option(
        None,
        "--max-subtasks",
        min=1,
        help="Cap how many planner subtasks are attempted (default: all).",
    ),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    """Run the edit workflow for *task*."""
    overrides: dict = {"agent": {}, "web": {}}
    if dry_run:
        overrides["agent"]["dry_run"] = True
    if auto_apply:
        overrides["agent"]["auto_apply"] = True
        overrides["agent"]["dry_run"] = False
    if no_tests:
        overrides["agent"]["auto_run_tests"] = False
    if no_web:
        overrides["web"]["enabled"] = False
    elif web:
        overrides["web"]["enabled"] = True
    if require_web_approval:
        overrides["web"]["require_user_approval"] = True
    if no_require_web_approval:
        overrides["web"]["require_user_approval"] = False
    if not overrides["agent"]:
        overrides.pop("agent")
    if not overrides["web"]:
        overrides.pop("web")

    agent = _build_agent(repo, config, overrides=overrides or None)

    def _approve(url: str) -> bool:
        if not agent.config.web.require_user_approval:
            return True
        return typer.confirm(f"Fetch {url}?", default=False)

    report = agent.run_edit(
        task,
        web=web or agent.config.web.enabled,
        allow_dirty=allow_dirty,
        approve_fetch=_approve,
        max_subtasks=max_subtasks,
    )
    _render_report(report, verbose=verbose)


@app.command("run-tests")
def cmd_run_tests(
    repo: Path = typer.Option(Path("."), "--repo", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Run the configured validation command."""
    agent = _build_agent(repo, config, use_llm=False)
    cmd = agent.config.validation.test_command
    if not cmd:
        console.print("[yellow]No test command configured[/yellow]")
        raise typer.Exit(2)
    result = run_command(cmd, cwd=repo, timeout=agent.config.llm.timeout_seconds * 2 or 300)
    console.print(summarize_result(result))


@app.command("status")
def cmd_status(
    repo: Path = typer.Option(Path("."), "--repo", exists=True, file_okay=False),
    config: Optional[Path] = typer.Option(None, "--config"),
) -> None:
    """Show indexed file count and configuration summary."""
    agent = _build_agent(repo, config, use_llm=False)
    files = agent.store.all_files()
    summaries = agent.store.all_summaries()
    table = Table(title="ShardCoder status")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Repository", str(repo.resolve()))
    table.add_row("Indexed files", str(len(files)))
    table.add_row("Summarised files", str(len(summaries)))
    table.add_row("Local model URL", agent.config.llm.base_url)
    table.add_row("Default model", agent.config.llm.model)
    table.add_row("Web backend", agent.config.web.backend)
    table.add_row("Web enabled", str(agent.config.web.enabled))
    console.print(table)


def _render_report(report: AgentReport, *, verbose: bool = False) -> None:
    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]warning:[/yellow] {w}")

    if not report.outcomes:
        console.print("[red]No subtasks executed.[/red]")
        return

    console.print(
        Panel(
            f"[bold]Task:[/bold] {report.task}\n"
            f"Subtasks executed: {len(report.outcomes)}\n"
            f"Files changed: {', '.join(report.files_changed) or '(none)'}\n"
            f"External docs used: {report.used_external_docs}\n"
            f"Sources: {', '.join(report.external_sources) or '(none)'}\n"
            f"Stop reason: {report.stop_reason or '(plan completed)'}",
            title="ShardCoder edit report",
        )
    )

    for idx, outcome in enumerate(report.outcomes, start=1):
        console.print(
            Panel(
                f"[bold]Subtask {idx}/{len(report.outcomes)}:[/bold] "
                f"{outcome.subtask.goal}\n"
                f"Iterations: {outcome.iterations}\n"
                f"Files changed: "
                f"{', '.join(outcome.apply_result.changed_files) if outcome.apply_result else '(none)'}",
                title=outcome.subtask.id,
            )
        )

        if outcome.no_patch_reason:
            console.print(
                f"[yellow]Model returned NO_PATCH:[/yellow] {outcome.no_patch_reason}"
            )

        if outcome.validation and outcome.validation.errors:
            for err in outcome.validation.errors:
                console.print(f"[red]patch validation:[/red] {err}")
        if outcome.validation and outcome.validation.warnings:
            for w in outcome.validation.warnings:
                console.print(f"[yellow]patch warning:[/yellow] {w}")

        if outcome.command_result:
            status = (
                "[green]OK[/green]"
                if outcome.command_result.ok
                else "[red]FAIL[/red]"
            )
            console.print(f"Validation: {status}")
            if verbose:
                console.print(summarize_result(outcome.command_result))

        if verbose and outcome.context_pack:
            console.print(
                Panel(
                    outcome.context_pack.render(),
                    title=(
                        f"Context pack ({outcome.context_pack.used_tokens}/"
                        f"{outcome.context_pack.max_tokens} tokens)"
                    ),
                )
            )


if __name__ == "__main__":
    app()
