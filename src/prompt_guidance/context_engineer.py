"""
Context Engineer
────────────────
Intelligently constructs the optimal context window for the rewriter LLM,
replacing naive "top-k RAG" with a multi-signal, budget-aware pipeline:

  1. Classify the prompt  — type, domain, weaknesses, complexity
  2. Multi-query retrieval — 3 targeted Qdrant searches from different angles
  3. RLHF injection       — retrieve similar high-rated past rewrites
  4. Budget allocation    — compose the final context within a token budget

Why this beats naive RAG:
  • Naive: embed prompt → retrieve top-5 → paste
  • Context Engineering: understand intent → retrieve from 3 angles →
    fuse with feedback signal → allocate budget by signal quality →
    produce a richer, more targeted context window

The output of build() is a structured string that is inserted into the
rewriter's USER_TEMPLATE as the {context} block.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from prompt_guidance.feedback import FeedbackRecord, FeedbackStore
from prompt_guidance.llm import BaseLLM
from prompt_guidance.pipeline import RAGPipeline
from prompt_guidance.vectorstore import Chunk


# ─────────────────────────────────────────────────────────────────
# Prompt classification
# ─────────────────────────────────────────────────────────────────

CLASSIFIER_SYSTEM = """\
You are a prompt analysis engine. Analyze the user's prompt and return ONLY a JSON object.
No preamble, no markdown fences, no explanation — raw JSON only.

Required JSON schema:
{
  "type": "<one of: code_generation | code_review | creative_writing | data_analysis | question_answering | summarization | planning | instruction_following | comparison | other>",
  "domain": "<one of: technical | creative | business | scientific | educational | general>",
  "complexity": "<one of: simple | medium | complex>",
  "target_model": "<detected AI system name if mentioned, else null>",
  "main_weaknesses": ["<array of: no_role | no_format | too_vague | no_context | no_constraints | no_examples | ambiguous_intent | overloaded | no_audience>"],
  "retrieval_queries": [
    "<query 1: specific technique search>",
    "<query 2: domain/type-specific best practice>",
    "<query 3: weakness-targeted fix>"
  ]
}
"""

CLASSIFIER_USER = "Analyze this prompt:\n\n{prompt}"


@dataclass
class PromptClassification:
    type: str = "other"
    domain: str = "general"
    complexity: str = "medium"
    target_model: Optional[str] = None
    main_weaknesses: list[str] = field(default_factory=list)
    retrieval_queries: list[str] = field(default_factory=list)
    # Populated post-rewrite by DriftDetector when a system prompt is modified.
    drift_risk: Optional[str] = None        # "none" | "low" | "medium" | "high" | "critical"
    drift_affected_queries: int = 0         # number of queries that shifted beyond threshold

    @classmethod
    def from_json(cls, raw: str) -> "PromptClassification":
        # Strip markdown fences if present
        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            return cls()
        return cls(
            type=data.get("type", "other"),
            domain=data.get("domain", "general"),
            complexity=data.get("complexity", "medium"),
            target_model=data.get("target_model"),
            main_weaknesses=data.get("main_weaknesses", []),
            retrieval_queries=data.get("retrieval_queries", []),
        )

    def summary(self) -> str:
        parts = [f"type={self.type}", f"domain={self.domain}", f"complexity={self.complexity}"]
        if self.target_model:
            parts.append(f"target={self.target_model}")
        if self.main_weaknesses:
            parts.append(f"issues=[{', '.join(self.main_weaknesses)}]")
        if self.drift_risk and self.drift_risk != "none":
            parts.append(
                f"drift_risk={self.drift_risk} ({self.drift_affected_queries} queries affected)"
            )
        return "  ".join(parts)


# ─────────────────────────────────────────────────────────────────
# Context budget
# ─────────────────────────────────────────────────────────────────

@dataclass
class ContextBudget:
    """
    Governs how many excerpts / characters go to each context slot.
    Keeps the context window focused and under control.
    """
    knowledge_chunks: int = 4     # RAG technique excerpts
    rlhf_examples: int = 2        # high-rated past rewrites
    max_chunk_chars: int = 400    # truncate individual chunks at this length

    @classmethod
    def for_complexity(cls, complexity: str) -> "ContextBudget":
        return {
            "simple":  cls(knowledge_chunks=3, rlhf_examples=1, max_chunk_chars=300),
            "medium":  cls(knowledge_chunks=4, rlhf_examples=2, max_chunk_chars=400),
            "complex": cls(knowledge_chunks=5, rlhf_examples=3, max_chunk_chars=500),
        }.get(complexity, cls())


# ─────────────────────────────────────────────────────────────────
# Context Engineer
# ─────────────────────────────────────────────────────────────────

class ContextEngineer:
    """
    Builds the optimal context for a single prompt enhancement request.

    Pipeline:
        classify → multi-query RAG → RLHF lookup → budget → assemble
    """

    def __init__(
        self,
        llm: BaseLLM,
        rag: RAGPipeline,
        feedback_store: Optional[FeedbackStore] = None,
    ):
        self.llm = llm
        self.rag = rag
        self.feedback_store = feedback_store

    # ── Step 1: Classify ─────────────────────────────────────────

    def classify(self, prompt: str) -> PromptClassification:
        """Use a cheap LLM call to understand prompt intent and weaknesses."""
        raw = self.llm.generate(
            system=CLASSIFIER_SYSTEM,
            user=CLASSIFIER_USER.format(prompt=prompt[:800]),
        )
        return PromptClassification.from_json(raw)

    # ── Step 2: Multi-query RAG ───────────────────────────────────

    def multi_retrieve(
        self,
        prompt: str,
        classification: PromptClassification,
        budget: ContextBudget,
    ) -> list[Chunk]:
        """
        Execute multiple targeted Qdrant searches and fuse the results.
        Deduplicates by content hash, keeps the highest-scoring copy.
        """
        # Build search queries from classification
        queries: list[str] = []

        # Query 1: literal prompt text
        queries.append(f"prompt engineering techniques: {prompt[:200]}")

        # Query 2: type + domain specific
        queries.append(f"best practices for {classification.type} prompts in {classification.domain}")

        # Query 3: weakness-targeted
        if classification.main_weaknesses:
            fix = " ".join(classification.main_weaknesses[:2])
            queries.append(f"how to fix {fix} in prompts")

        # Query 4: classification-generated queries (if present)
        queries.extend(classification.retrieval_queries[:2])

        # Execute all searches
        seen: dict[str, Chunk] = {}
        for q in queries:
            hits = self.rag.retrieve(q, top_k=budget.knowledge_chunks)
            for chunk in hits:
                key = chunk.content[:100]  # deduplicate by content prefix
                if key not in seen or (chunk.score or 0) > (seen[key].score or 0):
                    seen[key] = chunk

        # Sort by score, take top budget.knowledge_chunks
        ranked = sorted(seen.values(), key=lambda c: c.score or 0, reverse=True)
        return ranked[: budget.knowledge_chunks]

    # ── Step 3: RLHF lookup ───────────────────────────────────────

    def get_rlhf_examples(
        self,
        query_vector: list[float],
        budget: ContextBudget,
        min_rating: int = 4,
    ) -> list[FeedbackRecord]:
        """Retrieve high-rated past rewrites for similar prompts."""
        if self.feedback_store is None:
            return []
        try:
            return self.feedback_store.get_good_examples(
                query_vector, min_rating=min_rating, top_k=budget.rlhf_examples
            )
        except Exception:
            return []

    # ── Step 4: Assemble context ──────────────────────────────────

    def assemble(
        self,
        chunks: list[Chunk],
        rlhf_examples: list[FeedbackRecord],
        classification: PromptClassification,
        budget: ContextBudget,
    ) -> str:
        """Compose the final context string from all signals."""
        sections: list[str] = []

        # 4a. Classification summary
        sections.append(
            f"## PROMPT ANALYSIS\n"
            f"Type: {classification.type}  Domain: {classification.domain}  "
            f"Complexity: {classification.complexity}"
            + (f"  Target: {classification.target_model}" if classification.target_model else "")
            + (f"\nDetected weaknesses: {', '.join(classification.main_weaknesses)}"
               if classification.main_weaknesses else "")
        )

        # 4b. Knowledge base excerpts (truncated to budget)
        if chunks:
            parts = ["## RELEVANT PROMPT ENGINEERING KNOWLEDGE"]
            for i, c in enumerate(chunks, 1):
                score_str = f" [score: {c.score:.2f}]" if c.score is not None else ""
                content = c.content[:budget.max_chunk_chars]
                if len(c.content) > budget.max_chunk_chars:
                    content += "…"
                basename = c.source.split("\\")[-1].split("/")[-1]
                meta = ""
                if "sheet" in c.metadata:
                    meta = f" — {c.metadata['sheet']}"
                parts.append(f"[{i}]{score_str} {basename}{meta}\n{content}")
            sections.append("\n\n".join(parts))

        # 4c. RLHF few-shot examples
        if rlhf_examples and self.feedback_store is not None:
            rlhf_block = self.feedback_store.format_as_few_shot(rlhf_examples)
            if rlhf_block:
                sections.append(rlhf_block)

        # 4d. Drift risk warning — injected into context so the rewriter
        #     knows to be conservative when drift severity is high/critical.
        if (
            classification.drift_risk
            and classification.drift_risk not in ("none", None)
            and classification.drift_affected_queries > 0
        ):
            severity = classification.drift_risk.upper()
            n = classification.drift_affected_queries
            sections.append(
                f"## DRIFT RISK ALERT\n"
                f"Severity: {severity}\n"
                f"This prompt modification affects {n} historical user quer{'y' if n == 1 else 'ies'}.\n"
                f"When rewriting, preserve alignment for the affected query types "
                f"and avoid narrowing the prompt's scope."
            )

        return "\n\n" + "\n\n---\n\n".join(sections) + "\n"

    # ── Public: build() ───────────────────────────────────────────

    def build(
        self,
        prompt: str,
        top_k: Optional[int] = None,
    ) -> tuple[str, PromptClassification, list[Chunk], list[FeedbackRecord]]:
        """
        Full context engineering pipeline.

        Returns:
            context_str      — ready-to-inject context block
            classification   — prompt type analysis
            chunks           — retrieved knowledge chunks
            rlhf_examples    — injected preference examples
        """
        # Classify
        classification = self.classify(prompt)

        # Budget based on complexity
        budget = ContextBudget.for_complexity(classification.complexity)
        if top_k:
            budget.knowledge_chunks = top_k

        # Multi-query retrieval
        chunks = self.multi_retrieve(prompt, classification, budget)

        # RLHF lookup (embed the prompt to search feedback store)
        rlhf_examples: list[FeedbackRecord] = []
        if self.feedback_store is not None:
            query_vector = self.rag.embeddings.embed(prompt)
            rlhf_examples = self.get_rlhf_examples(query_vector, budget)

        # Assemble
        context_str = self.assemble(chunks, rlhf_examples, classification, budget)

        return context_str, classification, chunks, rlhf_examples
