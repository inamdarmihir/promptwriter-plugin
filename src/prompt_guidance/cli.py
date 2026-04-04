"""
CLI — pgt
─────────
Commands:
  pgt ingest <file>           Ingest a file into Qdrant
  pgt enhance <prompt>        Rewrite a prompt (core command)
  pgt search <query>          Raw similarity search (debug)
  pgt status                  Health check & ingested-source list
  pgt interactive             REPL mode

Global options:
  --provider / -p    LLM provider  (ollama | openai | anthropic | mistral)
  --model    / -m    Override model name
  --collection       Override Qdrant collection name
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prompt_guidance.config import settings

app = typer.Typer(
    name="pgt",
    help="[bold cyan]Prompt Guidance Tool[/bold cyan] — rewrite prompts with local/cloud LLMs + Qdrant RAG",
    rich_markup_mode="rich",
    add_completion=False,
)
console = Console()


# ─────────────────────────────────────────────────────────────────
# Display helpers
# ─────────────────────────────────────────────────────────────────

def _show_drift_report(report) -> None:  # type: ignore[annotation-unchecked]
    """Render a DriftReport as Rich panels."""
    from rich.markup import escape

    severity_colours = {
        "none": "green", "low": "green", "medium": "yellow",
        "high": "red", "critical": "bold red",
    }
    colour = severity_colours.get(report.severity, "white")

    console.print()
    console.print(Rule(f"[{colour}]DRIFT DETECTION REPORT — {report.severity.upper()}[/{colour}]", style=colour))
    console.print(Panel(escape(report.summary), title="[cyan]Summary[/cyan]", border_style="cyan"))

    if report.regressions:
        tbl = Table(title="Regressed Queries (shift beyond threshold)", border_style="red")
        tbl.add_column("Query", style="white")
        tbl.add_column("Sim Old", justify="right", style="dim")
        tbl.add_column("Sim New", justify="right", style="dim")
        tbl.add_column("Shift", justify="right", style="red")
        tbl.add_column("Risk", style="red")
        for r in report.regressions:
            tbl.add_row(
                r.query[:60],
                f"{r.sim_old:.3f}",
                f"{r.sim_new:.3f}",
                f"{r.shift:+.3f}",
                r.risk_level.upper(),
            )
        console.print(tbl)

    if report.improvements:
        tbl = Table(title="Improved Queries (shift beyond threshold)", border_style="green")
        tbl.add_column("Query", style="white")
        tbl.add_column("Sim Old", justify="right", style="dim")
        tbl.add_column("Sim New", justify="right", style="dim")
        tbl.add_column("Shift", justify="right", style="green")
        for r in report.improvements:
            tbl.add_row(r.query[:60], f"{r.sim_old:.3f}", f"{r.sim_new:.3f}", f"{r.shift:+.3f}")
        console.print(tbl)

    if report.questionnaire_impact:
        lines = []
        for category, queries in report.questionnaire_impact.items():
            lines.append(f"  [yellow]{category}[/yellow]: {len(queries)} quer{'y' if len(queries) == 1 else 'ies'}")
            for q in queries[:3]:
                lines.append(f"    • {escape(q[:70])}")
        console.print(
            Panel(
                "\n".join(lines),
                title="[yellow]Questionnaire Impact (affected categories)[/yellow]",
                border_style="yellow",
            )
        )

    console.print(
        f"[dim]Queries tested:[/dim] {report.total_queries}  "
        f"[dim]Threshold:[/dim] {report.threshold}  "
        f"[dim]Regression rate:[/dim] {report.regression_rate:.0%}  "
        f"[dim]Improvement rate:[/dim] {report.improvement_rate:.0%}"
    )


def _show_result(result, show_sources: bool = False) -> None:  # type: ignore[annotation-unchecked]
    console.print()
    console.print(Rule("[bold cyan]PROMPT GUIDANCE RESULT[/bold cyan]", style="cyan"))

    if result.analysis:
        console.print(Panel(result.analysis, title="[yellow]Analysis[/yellow]", border_style="yellow"))

    if result.issues:
        txt = "\n".join(f"  [red]•[/red] {i}" for i in result.issues)
        console.print(Panel(txt, title="[red]Issues Found[/red]", border_style="red"))

    original_panel = Panel(
        Text(result.original_prompt, style="dim"),
        title="[dim]Original[/dim]",
        border_style="dim",
        expand=True,
    )
    enhanced_panel = Panel(
        Text(result.enhanced_prompt, style="bold green"),
        title="[bold green]Enhanced Prompt[/bold green]",
        border_style="green",
        expand=True,
    )
    console.print(Columns([original_panel, enhanced_panel], equal=True))

    if result.changes:
        txt = "\n".join(f"  [cyan]→[/cyan] {c}" for c in result.changes)
        console.print(Panel(txt, title="[cyan]What Changed[/cyan]", border_style="cyan"))

    if result.techniques:
        console.print(
            Panel(
                f"[magenta]{result.techniques}[/magenta]",
                title="[magenta]Techniques Applied[/magenta]",
                border_style="magenta",
            )
        )

    if result.confidence:
        console.print(
            Panel(result.confidence, title="[blue]Confidence[/blue]", border_style="blue")
        )

    if result.alternatives and result.alternatives.strip().lower() not in ("none.", "none"):
        console.print(
            Panel(result.alternatives, title="[dim]Alternative Versions[/dim]", border_style="dim")
        )

    if result.drift_report is not None:
        _show_drift_report(result.drift_report)

    if show_sources and result.sources:
        tbl = Table(title="Knowledge Base Sources", border_style="blue")
        tbl.add_column("#", style="dim", width=3)
        tbl.add_column("Source", style="blue")
        tbl.add_column("Score", style="green", justify="right")
        tbl.add_column("Excerpt", style="dim")
        for i, chunk in enumerate(result.sources, 1):
            score = f"{chunk.score:.3f}" if chunk.score is not None else "—"
            basename = Path(chunk.source).name
            snippet = chunk.content[:70].replace("\n", " ")
            if len(chunk.content) > 70:
                snippet += "…"
            tbl.add_row(str(i), basename, score, snippet)
        console.print(tbl)


def _show_ingest_summary(result: dict) -> None:
    tbl = Table(title="[bold green]Ingestion Complete[/bold green]", border_style="green")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", style="green")
    tbl.add_row("File", result.get("file", ""))
    tbl.add_row("Documents loaded", str(result.get("documents", 0)))
    tbl.add_row("Chunks created", str(result.get("chunks", 0)))
    tbl.add_row("Vectors stored", str(result.get("stored", 0)))
    console.print()
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────
# ingest
# ─────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    file_path: Path = typer.Argument(..., help="File to ingest (.xlsx .csv .pdf .txt .md .json .docx)"),
    embed_provider: Optional[str] = typer.Option(None, "--embed-provider", "-e", help="ollama | openai | sentence-transformers"),
    embed_model: Optional[str] = typer.Option(None, "--embed-model", help="Override embedding model"),
    chunk_size: Optional[int] = typer.Option(None, "--chunk-size", "-c"),
    chunk_overlap: Optional[int] = typer.Option(None, "--overlap", "-o"),
    collection: Optional[str] = typer.Option(None, "--collection"),
) -> None:
    """Ingest a file into the Qdrant knowledge base."""
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.pipeline import IngestPipeline
    from prompt_guidance.vectorstore import QdrantStore

    emb = get_embeddings(provider=embed_provider, model=embed_model)
    store = QdrantStore(collection=collection, dimensions=emb.dimensions)

    try:
        store.initialize()
    except Exception as exc:
        console.print(f"[red]Qdrant error:[/red] {exc}")
        console.print("[yellow]Hint:[/yellow] Is Qdrant running?  →  docker compose up -d")
        raise typer.Exit(1)

    pipeline = IngestPipeline(embeddings=emb, store=store, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    try:
        summary = pipeline.run(file_path)
        _show_ingest_summary(summary)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────
# enhance  (the core command)
# ─────────────────────────────────────────────────────────────────

@app.command()
def enhance(
    prompt: Optional[str] = typer.Argument(None, help="Prompt to enhance (or pipe via stdin)"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="ollama | openai | anthropic | mistral"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model"),
    embed_provider: Optional[str] = typer.Option(None, "--embed-provider", "-e"),
    context: Optional[str] = typer.Option(None, "--context", "-c", help="Extra context about the task"),
    framework: Optional[str] = typer.Option(None, "--framework", "-f", help="Target AI system (e.g. 'Claude', 'GPT-4', 'LangChain')"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    show_sources: bool = typer.Option(False, "--sources", "-s", help="Show retrieved knowledge excerpts"),
    collection: Optional[str] = typer.Option(None, "--collection"),
    queries_file: Optional[Path] = typer.Option(
        None, "--queries-file", "-q",
        help="Path to a plain-text file of historical user queries (one per line) "
             "used for drift detection when enhancing a system prompt.",
    ),
    drift_threshold: float = typer.Option(
        0.15, "--drift-threshold",
        help="Min cosine-similarity shift to flag a query as drifted (default 0.15).",
    ),
) -> None:
    """[bold]Rewrite a prompt[/bold] using RAG-powered guidance."""
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.llm import get_llm
    from prompt_guidance.feedback import FeedbackStore
    from prompt_guidance.pipeline import RAGPipeline
    from prompt_guidance.rewriter import PromptRewriter
    from prompt_guidance.vectorstore import QdrantStore

    # Accept prompt from stdin
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            prompt = typer.prompt("Enter your prompt")

    if not prompt or not prompt.strip():
        console.print("[red]Error:[/red] Prompt cannot be empty.")
        raise typer.Exit(1)

    console.print(f"\n[dim]Provider:[/dim] {provider or settings.llm_provider}  "
                  f"[dim]Model:[/dim] {model or '(default)'}  "
                  f"[dim]RAG top-k:[/dim] {top_k}")

    llm = get_llm(provider=provider, model=model)
    emb = get_embeddings(provider=embed_provider)
    store = QdrantStore(collection=collection, dimensions=emb.dimensions)
    rag = RAGPipeline(embeddings=emb, store=store)

    # Wire in RLHF feedback store so rewriter can retrieve preference examples
    fb_store = FeedbackStore(dimensions=emb.dimensions)
    try:
        fb_store.initialize()
    except Exception:
        fb_store = None  # gracefully degrade if Qdrant not up yet

    rewriter = PromptRewriter(llm=llm, rag=rag, feedback_store=fb_store)

    # Load historical queries for drift detection (if provided)
    historical_queries: Optional[list[str]] = None
    if queries_file is not None:
        try:
            lines = queries_file.read_text(encoding="utf-8").splitlines()
            historical_queries = [ln.strip() for ln in lines if ln.strip()]
            console.print(f"[dim]Loaded {len(historical_queries)} queries from {queries_file.name}[/dim]")
        except OSError as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not read queries file: {exc}")

    try:
        result = rewriter.rewrite(
            prompt,
            context=context,
            framework=framework,
            top_k=top_k,
            user_queries=historical_queries,
            drift_threshold=drift_threshold,
        )
        _show_result(result, show_sources=show_sources)

        # Show classification and session ID for feedback
        if result.classification:
            console.print(
                f"\n[dim]Classification:[/dim] {result.classification.summary()}"
            )
        console.print(
            f"[dim]Session ID:[/dim] [bold]{result.session_id}[/bold]  "
            f"[dim]— rate this rewrite with:[/dim]  "
            f"pgt feedback --rating <1-5> --original '...' --enhanced '...'"
        )
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[yellow]Hints:[/yellow]")
        console.print("  • Ollama running?   ollama serve")
        console.print("  • Qdrant running?   docker compose up -d")
        console.print("  • Knowledge loaded? pgt ingest <file>")
        raise typer.Exit(1)


# ─────────────────────────────────────────────────────────────────
# search  (debug / explore the knowledge base)
# ─────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: str = typer.Argument(..., help="Similarity search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    embed_provider: Optional[str] = typer.Option(None, "--embed-provider", "-e"),
    collection: Optional[str] = typer.Option(None, "--collection"),
) -> None:
    """Search the Qdrant knowledge base directly."""
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.pipeline import RAGPipeline
    from prompt_guidance.vectorstore import QdrantStore

    emb = get_embeddings(provider=embed_provider)
    store = QdrantStore(collection=collection, dimensions=emb.dimensions)
    rag = RAGPipeline(embeddings=emb, store=store)

    console.print(f"\n[dim]Searching:[/dim] {query}\n")
    chunks = rag.retrieve(query, top_k=top_k)

    if not chunks:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, chunk in enumerate(chunks, 1):
        score = f"{chunk.score:.3f}" if chunk.score is not None else "—"
        basename = Path(chunk.source).name
        console.print(
            Panel(
                chunk.content,
                title=f"[bold][{i}][/bold] {basename}  score=[green]{score}[/green]",
                border_style="blue",
            )
        )


# ─────────────────────────────────────────────────────────────────
# status
# ─────────────────────────────────────────────────────────────────

@app.command()
def status(
    collection: Optional[str] = typer.Option(None, "--collection"),
) -> None:
    """Show system health, available models, and ingested sources."""
    from prompt_guidance.vectorstore import QdrantStore

    console.print()
    console.print(Rule("[bold]System Status[/bold]"))

    # ── Ollama check ──
    ollama_ok = False
    ollama_models: list[str] = []
    if settings.llm_provider == "ollama" or settings.embed_provider == "ollama":
        try:
            import ollama
            client = ollama.Client(host=settings.ollama_host)
            ollama_models = [m.model for m in client.list().models]
            ollama_ok = True
        except Exception:
            pass

    # ── Qdrant check ──
    qdrant_ok = False
    total_chunks = 0
    sources: list[str] = []
    store = QdrantStore(collection=collection)
    try:
        store.initialize()
        total_chunks = store.count()
        sources = store.list_sources()
        qdrant_ok = True
    except Exception:
        pass

    # Health table
    health = Table(show_header=False, border_style="green", box=None)
    health.add_column("", style="bold")
    health.add_column("")
    health.add_row(
        "Ollama",
        "[green]✓ connected[/green]" if ollama_ok else "[dim]not checked[/dim]" if settings.llm_provider != "ollama" else "[red]✗ unreachable[/red]",
    )
    health.add_row(
        "Qdrant",
        "[green]✓ connected[/green]" if qdrant_ok else "[red]✗ unreachable[/red]",
    )
    health.add_row("Collection", collection or settings.qdrant_collection)
    health.add_row("Total chunks", str(total_chunks))
    health.add_row("LLM provider", settings.llm_provider)
    health.add_row("Embed provider", settings.embed_provider)
    console.print(health)

    if ollama_models:
        console.print(f"\n[bold]Ollama models:[/bold] {', '.join(ollama_models)}")

    if sources:
        tbl = Table(title="\nIngested Sources", border_style="blue")
        tbl.add_column("File", style="blue")
        tbl.add_column("Path", style="dim")
        for s in sources:
            tbl.add_row(Path(s).name, s)
        console.print(tbl)
    else:
        console.print("\n[yellow]No documents ingested yet.[/yellow]  Run: [bold]pgt ingest <file>[/bold]")


# ─────────────────────────────────────────────────────────────────
# interactive  (REPL)
# ─────────────────────────────────────────────────────────────────

@app.command()
def interactive(
    provider: Optional[str] = typer.Option(None, "--provider", "-p"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    collection: Optional[str] = typer.Option(None, "--collection"),
) -> None:
    """Start an interactive prompt-enhancement REPL."""
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.llm import get_llm
    from prompt_guidance.pipeline import RAGPipeline
    from prompt_guidance.rewriter import PromptRewriter
    from prompt_guidance.vectorstore import QdrantStore

    llm = get_llm(provider=provider, model=model)
    emb = get_embeddings()
    store = QdrantStore(collection=collection, dimensions=emb.dimensions)
    rag = RAGPipeline(embeddings=emb, store=store)
    rewriter = PromptRewriter(llm=llm, rag=rag)

    console.print(
        Panel(
            f"[bold cyan]Prompt Guidance — Interactive Mode[/bold cyan]\n"
            f"Provider: [green]{llm.provider_name}[/green]  Model: [green]{llm.model_name}[/green]\n"
            "Type [bold]exit[/bold] or press Ctrl+C to quit.",
            border_style="cyan",
        )
    )

    while True:
        try:
            raw = typer.prompt("\nYour prompt")
            if raw.strip().lower() in ("exit", "quit", "q", ":q"):
                break
            result = rewriter.rewrite(raw)
            _show_result(result)
        except (KeyboardInterrupt, EOFError):
            break

    console.print("\n[dim]Session ended.[/dim]")


# ─────────────────────────────────────────────────────────────────
# feedback  (RLHF: rate a past rewrite)
# ─────────────────────────────────────────────────────────────────

@app.command()
def feedback(
    original: str = typer.Option(..., "--original", "-o", prompt="Original prompt", help="The original prompt that was enhanced"),
    enhanced: str = typer.Option(..., "--enhanced", "-e", prompt="Enhanced prompt", help="The rewritten prompt to rate"),
    rating: int = typer.Option(..., "--rating", "-r", prompt="Rating (1-5)", min=1, max=5, help="1=bad → 5=excellent"),
    comment: str = typer.Option("", "--comment", "-c", help="Optional explanation of why this rewrite worked or didn't"),
    techniques: str = typer.Option("", "--techniques", "-t", help="Techniques that were applied"),
    embed_provider: Optional[str] = typer.Option(None, "--embed-provider"),
    collection: Optional[str] = typer.Option(None, "--collection"),
) -> None:
    """
    [bold]Rate a rewrite[/bold] to train the RLHF feedback loop.

    High-rated rewrites are stored and injected as few-shot examples
    when similar prompts are enhanced in the future.
    """
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.feedback import FeedbackRecord, FeedbackStore
    from prompt_guidance.vectorstore import QdrantStore

    emb = get_embeddings(provider=embed_provider)
    store = QdrantStore(collection=collection, dimensions=emb.dimensions)
    fb_store = FeedbackStore(dimensions=emb.dimensions)

    try:
        store.initialize()
        fb_store.initialize()
    except Exception as exc:
        console.print(f"[red]Qdrant error:[/red] {exc}")
        raise typer.Exit(1)

    record = FeedbackRecord(
        original_prompt=original,
        enhanced_prompt=enhanced,
        rating=rating,
        comment=comment,
        techniques_used=techniques,
    )
    embedding = emb.embed(original)
    fb_store.save(record, embedding)

    stars = "★" * rating + "☆" * (5 - rating)
    console.print(f"\n[green]✓[/green] Feedback saved!  {stars}  ({rating}/5)")
    if comment:
        console.print(f"[dim]Comment:[/dim] {comment}")
    console.print(
        "\n[dim]This will be used as a few-shot example for similar prompts in the future.[/dim]"
    )


@app.command("feedback-stats")
def feedback_stats(
    embed_provider: Optional[str] = typer.Option(None, "--embed-provider"),
) -> None:
    """Show RLHF feedback statistics."""
    from prompt_guidance.embeddings import get_embeddings
    from prompt_guidance.feedback import FeedbackStore

    emb = get_embeddings(provider=embed_provider)
    fb_store = FeedbackStore(dimensions=emb.dimensions)

    try:
        fb_store.initialize()
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1)

    stats = fb_store.get_stats()
    console.print()
    tbl = Table(title="RLHF Feedback Statistics", border_style="magenta")
    tbl.add_column("Metric", style="bold")
    tbl.add_column("Value", style="magenta")
    tbl.add_row("Total rated rewrites", str(stats["total"]))
    tbl.add_row("Average rating", str(stats["avg_rating"]) if stats["avg_rating"] else "—")
    for star in range(5, 0, -1):
        count = stats.get("distribution", {}).get(str(star), 0)
        bar = "█" * count
        tbl.add_row(f"{'★' * star}", f"{count:3d}  {bar}")
    console.print(tbl)


# ─────────────────────────────────────────────────────────────────
# detect-drift  (standalone drift analysis between two prompts)
# ─────────────────────────────────────────────────────────────────

@app.command("detect-drift")
def detect_drift(
    old_prompt: Optional[str] = typer.Option(
        None, "--old", "-o",
        help="Original system/context prompt (or pipe via stdin)",
    ),
    new_prompt: str = typer.Option(
        ..., "--new", "-n",
        help="Modified/enhanced prompt to compare against the original",
    ),
    queries_file: Optional[Path] = typer.Option(
        None, "--queries-file", "-q",
        help="Plain-text file of historical user queries (one per line). "
             "Uses built-in sample queries when omitted.",
    ),
    embed_provider: Optional[str] = typer.Option(
        None, "--embed-provider", "-e",
        help="Embedding provider: ollama | openai | sentence-transformers",
    ),
    threshold: float = typer.Option(
        0.15, "--threshold", "-t",
        help="Min cosine-similarity shift to flag a query (default 0.15).",
    ),
) -> None:
    """
    [bold]Detect semantic drift[/bold] between two versions of a system prompt.

    Compares how historical user queries align with the old vs new prompt
    and reports regressions, improvements, and questionnaire impact.

    Examples:

      pgt detect-drift --old "You are a general assistant." \\
                       --new "You are a coding assistant. Only answer technical questions."

      pgt detect-drift --old "..." --new "..." --queries-file my_queries.txt
    """
    from prompt_guidance.drift_detector import DriftDetector
    from prompt_guidance.embeddings import get_embeddings

    # Accept old_prompt from stdin when not passed via option
    if old_prompt is None:
        if not sys.stdin.isatty():
            old_prompt = sys.stdin.read().strip()
        else:
            old_prompt = typer.prompt("Old prompt")

    if not old_prompt or not old_prompt.strip():
        console.print("[red]Error:[/red] Old prompt cannot be empty.")
        raise typer.Exit(1)

    if not new_prompt.strip():
        console.print("[red]Error:[/red] New prompt cannot be empty.")
        raise typer.Exit(1)

    # Load user queries
    user_queries: Optional[list[str]] = None
    if queries_file is not None:
        try:
            lines = queries_file.read_text(encoding="utf-8").splitlines()
            user_queries = [ln.strip() for ln in lines if ln.strip()]
            console.print(f"[dim]Using {len(user_queries)} queries from {queries_file.name}[/dim]")
        except OSError as exc:
            console.print(f"[yellow]Warning:[/yellow] Could not read queries file: {exc}")
    else:
        from prompt_guidance.drift_detector import SAMPLE_QUERIES
        console.print(f"[dim]Using {len(SAMPLE_QUERIES)} built-in sample queries[/dim]")

    try:
        emb = get_embeddings(provider=embed_provider)
        detector = DriftDetector(emb)
        report = detector.analyze(
            old_prompt=old_prompt,
            new_prompt=new_prompt,
            user_queries=user_queries,
            threshold=threshold,
        )
        _show_drift_report(report)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print("[yellow]Hints:[/yellow]")
        console.print("  • Ollama running?   ollama serve")
        console.print("  • OpenAI key set?   export OPENAI_API_KEY=...")
        console.print("  • Use local models? export EMBED_PROVIDER=sentence-transformers")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
