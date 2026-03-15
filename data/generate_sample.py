"""
Generate a sample Excel knowledge base of prompt engineering techniques.
Run once to seed the vector store:

    python data/generate_sample.py
    pgt ingest data/prompt_techniques.xlsx
"""
from pathlib import Path
import pandas as pd

OUTPUT = Path(__file__).parent / "prompt_techniques.xlsx"


# ─────────────────────────────────────────────────────────────────
# Sheet 1: Core Techniques
# ─────────────────────────────────────────────────────────────────

TECHNIQUES = [
    {
        "Technique": "Role / Persona Assignment",
        "Category": "Framing",
        "Description": (
            "Open the prompt by assigning the model a specific expert role. "
            "This primes the model to use domain vocabulary, adopt appropriate tone, "
            "and apply specialist reasoning patterns."
        ),
        "When to Use": "Any time you need domain expertise, authoritative tone, or consistent perspective.",
        "Bad Example": "Explain quantum computing.",
        "Good Example": (
            "You are a quantum computing researcher explaining to a software engineer "
            "with no physics background. Explain quantum computing focusing on the "
            "programming implications, not the physics. Use analogies to classical bits."
        ),
        "Impact": "High",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Output Format Specification",
        "Category": "Structure",
        "Description": (
            "Explicitly define the output format: JSON, markdown table, numbered list, "
            "prose paragraph, code block, etc. Specify length constraints if needed."
        ),
        "When to Use": "When you need structured, parseable, or consistently shaped output.",
        "Bad Example": "List the differences between REST and GraphQL.",
        "Good Example": (
            "Compare REST and GraphQL across these dimensions: "
            "Performance, Flexibility, Caching, Learning Curve, Best Use Case. "
            "Format your answer as a markdown table with these column headers: "
            "Dimension | REST | GraphQL | Verdict."
        ),
        "Impact": "High",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Chain of Thought (CoT)",
        "Category": "Reasoning",
        "Description": (
            "Instruct the model to reason step-by-step before giving the final answer. "
            "Dramatically improves accuracy on multi-step math, logic, and planning tasks."
        ),
        "When to Use": "Complex reasoning, multi-step problems, debugging, analysis tasks.",
        "Bad Example": "What is 17 × 24 + 13 × 8?",
        "Good Example": (
            "Solve the following arithmetic problem step by step, showing each "
            "intermediate calculation before giving the final answer: 17 × 24 + 13 × 8."
        ),
        "Impact": "High",
        "Works With": "GPT-4, Claude, Llama3, Mistral",
    },
    {
        "Technique": "Few-Shot Examples",
        "Category": "Learning",
        "Description": (
            "Provide 2-5 input/output examples before the real task. "
            "Teaches the model the exact pattern, tone, and format you expect "
            "without lengthy instructions."
        ),
        "When to Use": "Classification, extraction, formatting tasks, creative tasks with specific style.",
        "Bad Example": "Classify this review as positive or negative: 'The camera quality is mediocre.'",
        "Good Example": (
            "Classify each customer review as POSITIVE, NEGATIVE, or NEUTRAL.\n\n"
            "Review: 'Absolutely love this product!' → POSITIVE\n"
            "Review: 'Stopped working after two days.' → NEGATIVE\n"
            "Review: 'It arrived on time.' → NEUTRAL\n\n"
            "Review: 'The camera quality is mediocre.' → "
        ),
        "Impact": "High",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Context Injection",
        "Category": "Framing",
        "Description": (
            "Provide all relevant background information the model needs. "
            "Never assume the model knows your codebase, dataset schema, "
            "business rules, or conversation history."
        ),
        "When to Use": "Any task referencing private data, project-specific terms, or custom constraints.",
        "Bad Example": "Fix the bug in the authentication code.",
        "Good Example": (
            "Below is a Python Flask route that handles JWT authentication. "
            "Users report a 401 error when the token has exactly 0 seconds remaining. "
            "Here is the code:\n\n```python\n# [paste code here]\n```\n\n"
            "Identify the bug and provide the corrected code with an explanation."
        ),
        "Impact": "High",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Constraint Setting",
        "Category": "Control",
        "Description": (
            "Add explicit constraints: word/token limits, forbidden topics, "
            "required sections, tone restrictions, audience level, language, etc."
        ),
        "When to Use": "When default responses are too long, off-topic, or in the wrong register.",
        "Bad Example": "Summarize this article.",
        "Good Example": (
            "Summarize the following article for a non-technical executive audience. "
            "Maximum 3 bullet points. Each bullet ≤ 20 words. "
            "Focus only on business impact. Do not mention technical implementation details.\n\n"
            "[article here]"
        ),
        "Impact": "Medium",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Self-Consistency / Reflection",
        "Category": "Reasoning",
        "Description": (
            "Ask the model to review its own answer, check for errors, and revise. "
            "Or generate multiple answers and pick the most consistent one."
        ),
        "When to Use": "High-stakes outputs, fact-checking, code generation, legal/medical drafts.",
        "Bad Example": "Write a contract clause for IP ownership.",
        "Good Example": (
            "Draft a contract clause for IP ownership when a contractor creates software "
            "using the client's proprietary data. After drafting, review it for: "
            "(1) ambiguous terms, (2) missing edge cases, (3) enforceability issues. "
            "Then provide a revised final version."
        ),
        "Impact": "Medium",
        "Works With": "GPT-4, Claude, Gemini",
    },
    {
        "Technique": "Audience Calibration",
        "Category": "Tone",
        "Description": (
            "Specify exactly who will read the output: expertise level, age group, "
            "cultural context, prior knowledge. The model adapts vocabulary and depth accordingly."
        ),
        "When to Use": "Educational content, documentation, customer-facing copy.",
        "Bad Example": "Explain recursion.",
        "Good Example": (
            "Explain recursion to a 12-year-old who knows what a function is "
            "but has never heard of recursion. Use a real-world analogy (not Fibonacci). "
            "Keep it under 150 words."
        ),
        "Impact": "Medium",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Negative Prompting",
        "Category": "Control",
        "Description": (
            "Explicitly tell the model what NOT to do. "
            "Prevents common failure modes: hedging, unnecessary caveats, "
            "repetition, off-topic content."
        ),
        "When to Use": "When the model consistently adds unwanted content or misinterprets scope.",
        "Bad Example": "Write marketing copy for our new app.",
        "Good Example": (
            "Write three punchy marketing taglines for a productivity app aimed at remote workers. "
            "Do NOT: use clichés like 'game-changer' or 'revolutionary', "
            "mention competitors, add disclaimers, or exceed 10 words per tagline."
        ),
        "Impact": "Medium",
        "Works With": "All LLMs",
    },
    {
        "Technique": "Decomposition / Sub-task Chaining",
        "Category": "Planning",
        "Description": (
            "Break a complex task into explicit sub-tasks within the prompt. "
            "The model addresses each step in order, producing higher quality results "
            "than asking for everything at once."
        ),
        "When to Use": "Large, multi-part tasks: writing full reports, building systems, analysis pipelines.",
        "Bad Example": "Create a business plan for a SaaS startup.",
        "Good Example": (
            "Create a concise business plan for a B2B SaaS startup in HR analytics. "
            "Complete these sections in order:\n"
            "1. Problem Statement (2 sentences)\n"
            "2. Target Customer (job title, company size, pain point)\n"
            "3. Solution Overview (3 bullet points)\n"
            "4. Revenue Model (1 paragraph)\n"
            "5. Top 3 Risks and Mitigations (table)\n"
            "Use professional but accessible language."
        ),
        "Impact": "High",
        "Works With": "All LLMs",
    },
]

# ─────────────────────────────────────────────────────────────────
# Sheet 2: Anti-Patterns (common mistakes)
# ─────────────────────────────────────────────────────────────────

ANTI_PATTERNS = [
    {
        "Anti-Pattern": "Vague Task Definition",
        "Description": "Prompt says 'help me with X' without specifying what kind of help.",
        "Example": "Help me with my Python code.",
        "Fix": "State the goal, the error/problem, the code, and expected vs actual behaviour.",
        "Severity": "High",
    },
    {
        "Anti-Pattern": "Missing Output Format",
        "Description": "No format specified, so the model guesses — and often guesses wrong.",
        "Example": "Give me a list of ideas.",
        "Fix": "Specify: numbered list, bullet points, JSON array, table, prose paragraphs.",
        "Severity": "Medium",
    },
    {
        "Anti-Pattern": "Prompt Overloading",
        "Description": "Asking for too many unrelated things in one prompt.",
        "Example": "Summarize this, translate it, and also write a tweet and an email about it.",
        "Fix": "One primary task per prompt. Use sub-tasks only for related sequential steps.",
        "Severity": "High",
    },
    {
        "Anti-Pattern": "Ambiguous Pronouns",
        "Description": "Using 'it', 'this', 'that' without clear referents.",
        "Example": "Compare it with the previous one and update the summary.",
        "Fix": "Name everything explicitly. Repeat nouns rather than use pronouns across sentences.",
        "Severity": "Medium",
    },
    {
        "Anti-Pattern": "Instruction Without Context",
        "Description": "Giving instructions that rely on knowledge the model doesn't have.",
        "Example": "Refactor the UserService class to follow our new pattern.",
        "Fix": "Include the class code and describe the new pattern inline.",
        "Severity": "High",
    },
    {
        "Anti-Pattern": "Leading the Model to a Wrong Answer",
        "Description": "The prompt contains assumptions that bias the model toward a specific (possibly wrong) conclusion.",
        "Example": "Explain why Python is always better than JavaScript.",
        "Fix": "Ask for an objective comparison: 'Compare Python and JavaScript for [specific use case].'",
        "Severity": "Medium",
    },
    {
        "Anti-Pattern": "Ignoring Token Limits",
        "Description": "Prompt + expected output exceeds the model's context window.",
        "Example": "Summarize all 500 customer reviews below: [pastes 50k tokens].",
        "Fix": "Chunk large inputs. Process batches. Use map-reduce style multi-call strategies.",
        "Severity": "High",
    },
]

# ─────────────────────────────────────────────────────────────────
# Sheet 3: Framework-specific tips
# ─────────────────────────────────────────────────────────────────

FRAMEWORK_TIPS = [
    {
        "Framework": "OpenAI GPT-4 / GPT-4o",
        "Tip": "Use the system message for persona and constraints. Use user message for the actual task.",
        "Notes": "GPT-4o responds well to JSON mode (response_format={type: json_object}).",
    },
    {
        "Framework": "Anthropic Claude",
        "Tip": "Claude responds very well to XML tags for structure: <task>, <context>, <output_format>.",
        "Notes": "Claude 3.5+ follows multi-step instructions reliably. Use <thinking> for CoT.",
    },
    {
        "Framework": "Ollama / Local LLMs",
        "Tip": "Keep prompts concise — smaller local models get confused by very long system prompts.",
        "Notes": "Use Modelfile SYSTEM parameter to set persistent persona. Test with llama3.2 or mistral.",
    },
    {
        "Framework": "Mistral",
        "Tip": "Mistral is instruction-following focused. Clear imperative sentences work best.",
        "Notes": "Mistral-Large handles complex reasoning well. Mistral-Small is faster for simple tasks.",
    },
    {
        "Framework": "LangChain",
        "Tip": "Use ChatPromptTemplate with input_variables. Separate system / human / ai message templates.",
        "Notes": "Use PromptTemplate.partial() to pre-fill static values and keep prompts DRY.",
    },
    {
        "Framework": "LlamaIndex",
        "Tip": "Customize the default prompts in ServiceContext. Override text_qa_template for RAG tasks.",
        "Notes": "Structured output is best achieved via Pydantic output parsers + function calling.",
    },
    {
        "Framework": "RAG Systems (general)",
        "Tip": "Include a 'If the context does not contain the answer, say so' instruction to prevent hallucination.",
        "Notes": "Always inject context before the question, not after. Number chunks for traceability.",
    },
    {
        "Framework": "Code Generation",
        "Tip": "Specify: language, version, style guide, forbidden libraries, error handling style.",
        "Notes": "Ask for tests alongside the implementation. Request docstrings for complex functions.",
    },
]

# ─────────────────────────────────────────────────────────────────
# Sheet 4: Prompt templates
# ─────────────────────────────────────────────────────────────────

TEMPLATES = [
    {
        "Use Case": "Code Review",
        "Template": (
            "You are a senior {language} engineer performing a code review.\n"
            "Review the following code for: (1) bugs, (2) security issues, (3) performance, "
            "(4) readability.\n\nFor each issue found, output:\n"
            "- **Severity**: [Critical / High / Medium / Low]\n"
            "- **Location**: [line number or function name]\n"
            "- **Issue**: [description]\n"
            "- **Fix**: [corrected code snippet]\n\n"
            "Code:\n```{language}\n{code}\n```"
        ),
    },
    {
        "Use Case": "Document Summarization",
        "Template": (
            "Summarize the following document for a {audience}.\n"
            "Output format:\n"
            "## Key Points (3-5 bullets, ≤ 20 words each)\n"
            "## Decisions / Actions Required\n"
            "## Summary (≤ 100 words)\n\n"
            "Document:\n{document}"
        ),
    },
    {
        "Use Case": "Data Analysis",
        "Template": (
            "You are a data analyst. Analyze the following dataset summary and answer the question.\n\n"
            "Dataset: {dataset_description}\n"
            "Schema: {schema}\n"
            "Sample rows:\n{sample_rows}\n\n"
            "Question: {question}\n\n"
            "Provide: (1) your reasoning, (2) the SQL or Python code to get the answer, "
            "(3) the expected result, (4) any caveats."
        ),
    },
    {
        "Use Case": "RAG Answer Generation",
        "Template": (
            "Use only the provided context to answer the question. "
            "If the context does not contain enough information, say 'I don't have enough information to answer this.'\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer (cite the relevant excerpt in [brackets]):"
        ),
    },
    {
        "Use Case": "Creative Writing",
        "Template": (
            "You are a professional {genre} writer.\n"
            "Write a {length}-word {format} about {topic}.\n\n"
            "Tone: {tone}\n"
            "Audience: {audience}\n"
            "Must include: {must_include}\n"
            "Avoid: {avoid}\n\n"
            "Begin the {format} directly without any preamble."
        ),
    },
]


# ─────────────────────────────────────────────────────────────────
# Write to Excel
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    with pd.ExcelWriter(OUTPUT, engine="openpyxl") as writer:
        pd.DataFrame(TECHNIQUES).to_excel(writer, sheet_name="Techniques", index=False)
        pd.DataFrame(ANTI_PATTERNS).to_excel(writer, sheet_name="Anti-Patterns", index=False)
        pd.DataFrame(FRAMEWORK_TIPS).to_excel(writer, sheet_name="Framework Tips", index=False)
        pd.DataFrame(TEMPLATES).to_excel(writer, sheet_name="Prompt Templates", index=False)

    print(f"✓ Sample data written to {OUTPUT}")
    print(f"  Sheets: Techniques ({len(TECHNIQUES)}), Anti-Patterns ({len(ANTI_PATTERNS)}), "
          f"Framework Tips ({len(FRAMEWORK_TIPS)}), Prompt Templates ({len(TEMPLATES)})")
    print("\nNext step:")
    print("  pgt ingest data/prompt_techniques.xlsx")


if __name__ == "__main__":
    main()
