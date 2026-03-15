"""
Prompt Rewriter  (with Context Engineering + RLHF)
──────────────────────────────────────────────────
Pipeline for a single rewrite request:

  1. Context Engineering
       • Classify prompt type, domain, weaknesses, complexity
       • Multi-query Qdrant retrieval (3+ targeted searches)
       • RLHF: inject high-rated past rewrites as few-shot examples
       • Budget-aware context assembly

  2. LLM Generation
       • Structured system prompt with explicit rewriting policy
       • Full context block from step 1
       • Strict output format (7 sections)

  3. Result Parsing
       • Extract enhanced prompt, analysis, changes, techniques,
         confidence, and alternative versions

Rewriting Policy (enforced in the system prompt):
  1. NEVER truncate or omit information from the original intent
  2. ALWAYS output a complete, ready-to-use prompt in a code block
  3. ALWAYS explain every change — no silent rewrites
  4. Prefer additive improvements — strengthen what exists
  5. Preserve the user's domain language and terminology
  6. Apply only techniques supported by the retrieved knowledge
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from prompt_guidance.context_engineer import ContextEngineer, PromptClassification
from prompt_guidance.feedback import FeedbackRecord, FeedbackStore
from prompt_guidance.llm import BaseLLM, get_llm
from prompt_guidance.pipeline import RAGPipeline
from prompt_guidance.vectorstore import Chunk


# ─────────────────────────────────────────────────────────────────
# System prompt  (policy lives here — do not weaken)
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert prompt engineer. Your sole task is to rewrite user prompts \
so they produce the best possible results from any AI system.

══════════════════════════════════════
REWRITING POLICY  (never violate this)
══════════════════════════════════════
1. Preserve intent    — never change what the user is asking for.
2. Complete output    — the Enhanced Prompt must be a full, self-contained prompt
                        ready to paste directly into any AI system.
3. Explain every change — every modification must appear in "What Changed".
4. Additive only      — strengthen existing phrasing; do not replace meaning.
5. Keep domain terms  — do not substitute the user's technical vocabulary.
6. Evidence-based     — only apply techniques that appear in the retrieved knowledge
                        or the RLHF examples below.

══════════════════════════════════════
TECHNIQUES AVAILABLE
══════════════════════════════════════
• Role / Persona assignment
• Clear objective statement
• Context / background injection
• Output format specification  (JSON, table, bullet list, code block, prose…)
• Step-by-step / chain-of-thought instruction
• Constraints and guardrails
• Few-shot examples (when appropriate)
• Specificity over vagueness
• Audience calibration
• Negative prompting (tell the model what NOT to do)

══════════════════════════════════════
REQUIRED RESPONSE FORMAT  (strict — 7 sections)
══════════════════════════════════════
### ANALYSIS
[2-4 sentences: task type, weaknesses detected, what will be improved]

### ISSUES
- [issue 1]
- [issue 2]

### ENHANCED PROMPT
```
[The complete rewritten prompt — ready to copy-paste, nothing else in this block]
```

### WHAT CHANGED
- [change 1 and why]
- [change 2 and why]

### TECHNIQUES APPLIED
[comma-separated list of techniques used]

### CONFIDENCE
[HIGH | MEDIUM | LOW — one word + one sentence justification]

### ALTERNATIVE VERSIONS
[1-2 variant prompts (different tone/length) each in their own ``` block, or "None."]
"""

USER_TEMPLATE = """\
{context}

---

## ORIGINAL PROMPT TO REWRITE
{prompt}{extras}

Apply the rewriting policy and respond in the exact 7-section format.\
"""


# ─────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────

@dataclass
class RewriteResult:
    original_prompt: str
    enhanced_prompt: str
    analysis: str
    issues: list[str] = field(default_factory=list)
    changes: list[str] = field(default_factory=list)
    techniques: str = ""
    confidence: str = ""
    alternatives: str = ""
    classification: Optional[PromptClassification] = None
    sources: list[Chunk] = field(default_factory=list)
    rlhf_examples: list[FeedbackRecord] = field(default_factory=list)
    raw_response: str = ""
    session_id: str = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])


# ─────────────────────────────────────────────────────────────────
# Rewriter
# ─────────────────────────────────────────────────────────────────

class PromptRewriter:
    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        rag: Optional[RAGPipeline] = None,
        feedback_store: Optional[FeedbackStore] = None,
    ):
        self.llm = llm or get_llm()
        self.rag = rag or RAGPipeline()
        self.feedback_store = feedback_store
        self._context_engineer = ContextEngineer(
            llm=self.llm,
            rag=self.rag,
            feedback_store=self.feedback_store,
        )

    def rewrite(
        self,
        prompt: str,
        context: Optional[str] = None,
        framework: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> RewriteResult:
        # ── 1. Context Engineering ────────────────────────────────
        context_str, classification, chunks, rlhf_examples = (
            self._context_engineer.build(prompt, top_k=top_k)
        )

        # ── 2. Build extras ───────────────────────────────────────
        extras_parts: list[str] = []
        if context:
            extras_parts.append(f"\n\n## ADDITIONAL CONTEXT\n{context}")
        if framework:
            extras_parts.append(f"\n\n## TARGET SYSTEM / FRAMEWORK\n{framework}")
        extras = "".join(extras_parts)

        user_msg = USER_TEMPLATE.format(
            context=context_str,
            prompt=prompt,
            extras=extras,
        )

        # ── 3. Generate ───────────────────────────────────────────
        raw = self.llm.generate(system=SYSTEM_PROMPT, user=user_msg)

        # ── 4. Parse ──────────────────────────────────────────────
        result = self._parse(prompt, raw, chunks, rlhf_examples, classification)
        return result

    # ── Feedback ─────────────────────────────────────────────────

    def save_feedback(
        self,
        result: RewriteResult,
        rating: int,
        comment: str = "",
    ) -> None:
        """
        Record a human rating for a rewrite. Persists to the RLHF store.
        Future rewrites for similar prompts will see this as a preference example.
        """
        if self.feedback_store is None:
            raise RuntimeError(
                "No FeedbackStore attached to this PromptRewriter. "
                "Pass feedback_store=FeedbackStore() at construction time."
            )
        embedding = self.rag.embeddings.embed(result.original_prompt)
        record = FeedbackRecord(
            original_prompt=result.original_prompt,
            enhanced_prompt=result.enhanced_prompt,
            rating=rating,
            comment=comment,
            techniques_used=result.techniques,
            provider=self.llm.provider_name,
            model=self.llm.model_name,
        )
        self.feedback_store.save(record, embedding)

    # ── Parser ────────────────────────────────────────────────────

    def _parse(
        self,
        original: str,
        raw: str,
        sources: list[Chunk],
        rlhf_examples: list[FeedbackRecord],
        classification: Optional[PromptClassification],
    ) -> RewriteResult:

        def section(start: str, end: Optional[str] = None) -> str:
            s = raw.find(start)
            if s == -1:
                return ""
            s += len(start)
            if end:
                e = raw.find(end, s)
                return raw[s:e].strip() if e != -1 else raw[s:].strip()
            return raw[s:].strip()

        def code_block_in(text: str) -> str:
            m = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
            return m.group(1).strip() if m else text.strip()

        def bullets(text: str) -> list[str]:
            return [
                line.lstrip("-•* ").strip()
                for line in text.splitlines()
                if line.strip().startswith(("-", "•", "*"))
            ]

        enhanced_raw = section("### ENHANCED PROMPT", "### WHAT CHANGED")
        enhanced = code_block_in(enhanced_raw) if enhanced_raw else ""

        return RewriteResult(
            original_prompt=original,
            enhanced_prompt=enhanced or raw,
            analysis=section("### ANALYSIS", "### ISSUES"),
            issues=bullets(section("### ISSUES", "### ENHANCED PROMPT")),
            changes=bullets(section("### WHAT CHANGED", "### TECHNIQUES APPLIED")),
            techniques=section("### TECHNIQUES APPLIED", "### CONFIDENCE"),
            confidence=section("### CONFIDENCE", "### ALTERNATIVE VERSIONS"),
            alternatives=section("### ALTERNATIVE VERSIONS"),
            classification=classification,
            sources=sources,
            rlhf_examples=rlhf_examples,
            raw_response=raw,
        )
