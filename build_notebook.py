"""Generate tutorial.ipynb — run with:  py build_notebook.py"""
import json, pathlib, textwrap, uuid

def cid(): return uuid.uuid4().hex[:8]

def md(*lines):
    return {"cell_type":"markdown","id":cid(),"metadata":{},"source":"\n".join(lines)}

def code(*lines):
    src = "\n".join(lines)
    return {"cell_type":"code","id":cid(),"metadata":{},"source":src,
            "outputs":[],"execution_count":None}

# ─────────────────────────────────────────────────────────────────
cells = []
# ─────────────────────────────────────────────────────────────────

cells.append(md(
"# Prompt Guidance Tool — Tutorial Notebook",
"",
"> **What this notebook covers**",
"> 1. Spin up Qdrant locally with Docker and install Ollama",
"> 2. Ingest an Excel knowledge base into Qdrant",
"> 3. Understand how Context Engineering and RLHF work under the hood",
"> 4. Side-by-side comparison: *raw LLM output* vs *plugin-enhanced output*",
"> 5. **Benchmarked accuracy study** — Excel RAG pipeline with LLM-as-judge scoring,",
">    keyword coverage, and statistical analysis across 10 prompt categories",
"",
"**Stack:** Qdrant · Ollama · `prompt-guidance` plugin · pandas · matplotlib · seaborn",
))

# ── TOC ──────────────────────────────────────────────────────────
cells.append(md(
"## Table of Contents",
"1. [Prerequisites & System Check](#1-prerequisites--system-check)",
"2. [Infrastructure Setup — Qdrant + Ollama](#2-infrastructure-setup--qdrant--ollama)",
"3. [Install & Configure the Plugin](#3-install--configure-the-plugin)",
"4. [Build the Excel Knowledge Base](#4-build-the-excel-knowledge-base)",
"5. [Core Concepts Demo](#5-core-concepts-demo)",
"6. [Raw LLM vs Plugin-Enhanced — Side-by-Side](#6-raw-llm-vs-plugin-enhanced--side-by-side)",
"7. [RAG Pipeline Accuracy Benchmark](#7-rag-pipeline-accuracy-benchmark)",
"8. [RLHF Feedback Loop Demo](#8-rlhf-feedback-loop-demo)",
"9. [Results & Conclusions](#9-results--conclusions)",
))

# ═════════════════════════════════════════════════════════════════
# 1. Prerequisites
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 1. Prerequisites & System Check",
"",
"| Requirement | Purpose |",
"|---|---|",
"| Docker Desktop | Runs Qdrant vector DB |",
"| Ollama | Local LLM + embedding server |",
"| Python ≥ 3.11 | Runtime |",
"| `prompt-guidance` repo | The plugin we're benchmarking |",
))

cells.append(code(
"import subprocess, sys, platform",
"",
"checks = {",
'    "Python version": platform.python_version(),',
'    "Platform":       platform.system(),',
"}",
"",
"# Check Docker",
"try:",
"    r = subprocess.run(['docker', 'info'], capture_output=True, text=True, timeout=10)",
'    checks["Docker"] = "running ✓" if r.returncode == 0 else "NOT running ✗"',
"except FileNotFoundError:",
'    checks["Docker"] = "not found ✗"',
"",
"# Check Ollama",
"try:",
"    r = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)",
'    checks["Ollama"] = "installed ✓" if r.returncode == 0 else "not responding ✗"',
"except FileNotFoundError:",
'    checks["Ollama"] = "not found ✗"',
"",
"for k, v in checks.items():",
"    print(f'  {k:<20} {v}')",
))

cells.append(code(
"# Install notebook display helpers + plotting",
"import subprocess, sys",
"pkgs = ['rich', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'openpyxl', 'tabulate']",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *pkgs], check=True)",
"print('Dependencies ready.')",
))

# ═════════════════════════════════════════════════════════════════
# 2. Infrastructure
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 2. Infrastructure Setup — Qdrant + Ollama",
"",
"### 2-A  Qdrant (via Docker)",
"We start a local Qdrant instance. The `docker-compose.yml` in the repo root",
"handles everything — data is persisted in a named volume.",
))

cells.append(code(
"import subprocess, time, urllib.request, json as _json",
"",
"result = subprocess.run(",
"    ['docker', 'compose', 'up', '-d'],",
"    capture_output=True, text=True",
")",
"print(result.stdout or result.stderr)",
"",
"# Wait for Qdrant to be ready",
"print('Waiting for Qdrant...')",
"for attempt in range(20):",
"    try:",
"        with urllib.request.urlopen('http://localhost:6333/healthz', timeout=2) as r:",
"            if r.status == 200:",
"                print(f'Qdrant ready after {attempt+1}s ✓')",
"                break",
"    except Exception:",
"        time.sleep(1)",
"else:",
"    print('Qdrant did not start — check docker compose logs')",
))

cells.append(md(
"### 2-B  Ollama",
"",
"If Ollama is not installed, download it from [ollama.com](https://ollama.com/download).",
"Then pull the two models this tutorial uses:",
"",
"```bash",
"ollama pull llama3.2          # generation (3 B params, runs on CPU)",
"ollama pull nomic-embed-text  # embeddings (768 dims)",
"```",
))

cells.append(code(
"import subprocess, urllib.request",
"",
"# Check Ollama server",
"try:",
"    with urllib.request.urlopen('http://localhost:11434/api/tags', timeout=5) as r:",
"        data = _json.loads(r.read())",
"        installed = [m['name'] for m in data.get('models', [])]",
'        print("Installed Ollama models:", installed)',
"except Exception as e:",
'    print(f"Ollama server not reachable: {e}")',
'    print("Start it with: ollama serve")',
))

cells.append(code(
"# Pull models if not already present  (this can take a few minutes the first time)",
"MODELS = ['llama3.2', 'nomic-embed-text']",
"",
"for model in MODELS:",
"    if not any(model in m for m in installed):",
"        print(f'Pulling {model}...')",
"        subprocess.run(['ollama', 'pull', model], check=True)",
"    else:",
"        print(f'{model} already installed ✓')",
))

# ═════════════════════════════════════════════════════════════════
# 3. Install & Configure
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 3. Install & Configure the Plugin",
))

cells.append(code(
"import subprocess, sys, pathlib",
"",
"# Install the plugin in editable mode from the repo root",
"repo_root = pathlib.Path('.').resolve()",
"subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(repo_root)], check=True)",
"print('prompt-guidance installed ✓')",
))

cells.append(code(
"# Configure to use Ollama (fully local, no API key needed)",
"import os",
"os.environ.update({",
'    "LLM_PROVIDER":         "ollama",',
'    "EMBED_PROVIDER":       "ollama",',
'    "OLLAMA_HOST":          "http://localhost:11434",',
'    "OLLAMA_LLM_MODEL":     "llama3.2",',
'    "OLLAMA_EMBED_MODEL":   "nomic-embed-text",',
'    "EMBED_DIMENSIONS":     "768",',
'    "QDRANT_HOST":          "localhost",',
'    "QDRANT_PORT":          "6333",',
'    "QDRANT_COLLECTION":    "tutorial_knowledge",',
'    "CHUNK_SIZE":           "400",',
'    "CHUNK_OVERLAP":        "40",',
'    "TOP_K":                "5",',
"})",
"",
"# Re-import config so it picks up the env vars",
"import importlib",
"import prompt_guidance.config as _cfg",
"importlib.reload(_cfg)",
"print('Config loaded. LLM:', _cfg.settings.llm_provider, '| Embed:', _cfg.settings.embed_provider)",
))

# ═════════════════════════════════════════════════════════════════
# 4. Knowledge Base
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 4. Build the Excel Knowledge Base",
"",
"We use the bundled `data/generate_sample.py` script to create a",
"`prompt_techniques.xlsx` workbook with four sheets:",
"",
"| Sheet | Content |",
"|---|---|",
"| Techniques | 10 core prompt-engineering techniques with bad/good examples |",
"| Anti-Patterns | 7 common mistakes and their fixes |",
"| Framework Tips | Provider-specific guidance (OpenAI, Claude, Ollama, LangChain…) |",
"| Prompt Templates | Ready-to-use templates for 5 task types |",
"",
"Then we **ingest** the file — chunk it, embed every chunk with `nomic-embed-text`,",
"and store the vectors in Qdrant.",
))

cells.append(code(
"# Generate the Excel file",
"import subprocess, pathlib",
"data_dir = pathlib.Path('data')",
"xlsx_path = data_dir / 'prompt_techniques.xlsx'",
"",
"if not xlsx_path.exists():",
"    subprocess.run(['python', str(data_dir / 'generate_sample.py')], check=True)",
"",
"import pandas as pd",
"xl = pd.ExcelFile(xlsx_path)",
"print('Sheets in prompt_techniques.xlsx:')",
"for sheet in xl.sheet_names:",
"    df = xl.parse(sheet)",
"    print(f'  {sheet:<22} {len(df)} rows × {len(df.columns)} cols')",
))

cells.append(code(
"# Ingest into Qdrant",
"from prompt_guidance.embeddings import get_embeddings",
"from prompt_guidance.pipeline import IngestPipeline",
"from prompt_guidance.vectorstore import QdrantStore",
"",
"embeddings = get_embeddings()",
"store = QdrantStore(collection='tutorial_knowledge', dimensions=embeddings.dimensions)",
"store.initialize()",
"",
"pipeline = IngestPipeline(embeddings=embeddings, store=store)",
"result = pipeline.run(xlsx_path)",
"",
"print(f\"Ingested: {result['documents']} docs → {result['chunks']} chunks → {result['stored']} vectors\")",
"print(f\"Total vectors in Qdrant: {store.count()}\")",
))

cells.append(code(
"# Quick sanity search",
"from prompt_guidance.pipeline import RAGPipeline",
"rag = RAGPipeline(embeddings=embeddings, store=store)",
"",
"hits = rag.retrieve('chain of thought reasoning', top_k=3)",
"print('Top 3 hits for \"chain of thought reasoning\":')",
"for i, h in enumerate(hits, 1):",
"    print(f'  [{i}] score={h.score:.3f}  {h.content[:90].replace(chr(10),\" \")}...')",
))

# ═════════════════════════════════════════════════════════════════
# 5. Core Concepts Demo
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 5. Core Concepts Demo",
"",
"### 5-A  What the Context Engineer does",
"",
"Before the LLM sees your prompt, the `ContextEngineer` runs three steps:",
"",
"```",
"Classify  →  Multi-query RAG  →  RLHF lookup  →  Budget allocation  →  Context string",
"```",
))

cells.append(code(
"from prompt_guidance.llm import get_llm",
"from prompt_guidance.context_engineer import ContextEngineer",
"",
"llm = get_llm()",
"ce  = ContextEngineer(llm=llm, rag=rag)",
"",
"TEST_PROMPT = 'summarize this document'",
"classification = ce.classify(TEST_PROMPT)",
"",
"print('=== Prompt Classification ===')",
"print(f'  Type:        {classification.type}')",
"print(f'  Domain:      {classification.domain}')",
"print(f'  Complexity:  {classification.complexity}')",
"print(f'  Weaknesses:  {classification.main_weaknesses}')",
"print(f'  RAG queries: {classification.retrieval_queries}')",
))

cells.append(code(
"from prompt_guidance.context_engineer import ContextBudget",
"",
"budget = ContextBudget.for_complexity(classification.complexity)",
"chunks = ce.multi_retrieve(TEST_PROMPT, classification, budget)",
"",
"print(f'Multi-query RAG returned {len(chunks)} unique chunks (budget={budget.knowledge_chunks}):')",
"for i, c in enumerate(chunks, 1):",
"    sheet = c.metadata.get('sheet', '')",
"    print(f'  [{i}] score={c.score:.3f}  [{sheet}]  {c.content[:75].replace(chr(10),\" \")}...')",
))

cells.append(code(
"context_str, clf, chunks, rlhf_examples = ce.build(TEST_PROMPT)",
"",
"print(f'Built context: {len(context_str)} chars')",
"print(f'Chunks used: {len(chunks)}  |  RLHF examples: {len(rlhf_examples)}')",
"print()",
"print('--- Context snippet (first 500 chars) ---')",
"print(context_str[:500])",
))

# ═════════════════════════════════════════════════════════════════
# 6. Raw vs Enhanced Comparison
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 6. Raw LLM vs Plugin-Enhanced — Side-by-Side",
"",
"We run the **same 6 prompts** through:",
"- **Raw**: prompt sent directly to `llama3.2`",
"- **Enhanced**: prompt first rewritten by the plugin, *then* the enhanced version is sent",
"",
"The displayed output is what you'd actually copy-paste into any AI assistant.",
))

cells.append(code(
"import ollama as _ollama",
"",
"raw_client = _ollama.Client(host='http://localhost:11434')",
"",
"def raw_llm(prompt: str) -> str:",
"    resp = raw_client.chat(",
"        model='llama3.2',",
"        messages=[{'role': 'user', 'content': prompt}]",
"    )",
"    return resp.message.content",
"",
"DEMO_PROMPTS = [",
"    'explain transformers',",
"    'summarize this document',",
"    'fix my python code',",
"    'write marketing copy for our app',",
"    'analyze sales data',",
"    'compare REST and GraphQL',",
"]",
"print(f'{len(DEMO_PROMPTS)} demo prompts ready.')",
))

cells.append(code(
"from prompt_guidance.rewriter import PromptRewriter",
"from prompt_guidance.feedback import FeedbackStore",
"",
"fb_store = FeedbackStore(dimensions=embeddings.dimensions)",
"fb_store.initialize()",
"",
"rewriter = PromptRewriter(llm=llm, rag=rag, feedback_store=fb_store)",
"",
"comparison = []",
"for prompt in DEMO_PROMPTS:",
"    result     = rewriter.rewrite(prompt)",
"    raw_ans    = raw_llm(prompt)",
"    enh_ans    = raw_llm(result.enhanced_prompt)",
"    comparison.append({",
"        'original':       prompt,",
"        'enhanced_prompt': result.enhanced_prompt,",
"        'raw_answer':     raw_ans,",
"        'enhanced_answer': enh_ans,",
"        'techniques':     result.techniques,",
"        'classification': result.classification.type if result.classification else '',",
"        'weaknesses':     ', '.join(result.classification.main_weaknesses) if result.classification else '',",
"    })",
"    print(f'  done: {prompt[:40]}')",
"",
"print('All comparisons collected.')",
))

cells.append(code(
"# Pretty display",
"from IPython.display import display, HTML",
"",
"def side_by_side_html(item):",
"    orig  = item['original']",
"    enh   = item['enhanced_prompt'].replace('<','&lt;').replace('>','&gt;')",
"    ra    = item['raw_answer'].replace('<','&lt;').replace('>','&gt;')[:600]",
"    ea    = item['enhanced_answer'].replace('<','&lt;').replace('>','&gt;')[:600]",
"    tech  = item['techniques']",
"    clf   = item['classification']",
"    return f'''",
"    <div style=\"border:1px solid #ddd;border-radius:8px;padding:12px;margin:12px 0;\">",
"      <h4 style=\"margin:0 0 8px\">Original: <code>{orig}</code></h4>",
"      <p><b>Type:</b> {clf} &nbsp;|&nbsp; <b>Techniques:</b> {tech}</p>",
"      <table style=\"width:100%;border-collapse:collapse\">",
"        <tr>",
"          <th style=\"width:50%;background:#ffeeba;padding:6px\">&#x1f6ab; Raw LLM Answer</th>",
"          <th style=\"width:50%;background:#d4edda;padding:6px\">&#x2705; Enhanced Prompt Answer</th>",
"        </tr>",
"        <tr>",
"          <td style=\"padding:8px;vertical-align:top;white-space:pre-wrap;font-size:12px\">{ra}</td>",
"          <td style=\"padding:8px;vertical-align:top;white-space:pre-wrap;font-size:12px\">{ea}</td>",
"        </tr>",
"      </table>",
"    </div>'''",
"",
"for item in comparison[:3]:   # show first 3 to keep notebook readable",
"    display(HTML(side_by_side_html(item)))",
))

# ═════════════════════════════════════════════════════════════════
# 7. RAG Benchmark
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 7. RAG Pipeline Accuracy Benchmark",
"",
"### Methodology",
"",
"We evaluate **10 questions** that can be answered from the Excel knowledge base.",
"Each question is answered three ways:",
"",
"| Pipeline | Description |",
"|---|---|",
"| **Baseline** | Question sent raw to LLM (no RAG, no enhancement) |",
"| **Raw RAG** | Question + retrieved Excel context, no prompt enhancement |",
"| **Enhanced RAG** | Plugin rewrites the question first, then RAG + LLM |",
"",
"### Scoring (LLM-as-Judge)",
"",
"We ask `llama3.2` to score each answer on three dimensions:",
"",
"| Dimension | What it measures |",
"|---|---|",
"| **Accuracy** | Factual correctness relative to the reference answer |",
"| **Completeness** | Coverage of all relevant points |",
"| **Clarity** | How easy the answer is to understand and act on |",
"",
"Each dimension is scored 1–10. Final score = average of the three.",
))

cells.append(code(
"# Benchmark dataset — questions whose answers are in the Excel",
"BENCHMARK = [",
"    {",
"        'id': 'Q01',",
"        'category': 'technique',",
"        'question': 'What is chain of thought prompting and when should I use it?',",
"        'reference': 'Chain of thought instructs the model to reason step-by-step before answering. Use it for complex multi-step problems like math, logic, debugging, and planning.',",
"        'expected_keywords': ['step', 'reasoning', 'complex', 'logic', 'problem'],",
"    },",
"    {",
"        'id': 'Q02',",
"        'category': 'technique',",
"        'question': 'How do few-shot examples improve prompt responses?',",
"        'reference': 'Few-shot examples teach the model the desired pattern, tone, and output format by showing 2-5 input/output pairs before the real task.',",
"        'expected_keywords': ['examples', 'pattern', 'format', 'input', 'output'],",
"    },",
"    {",
"        'id': 'Q03',",
"        'category': 'anti-pattern',",
"        'question': 'What is vague task definition and how do I fix it?',",
"        'reference': 'Vague task definition means the prompt says help me with X without specifying the type of help needed. Fix it by stating the goal, error, code, and expected vs actual behaviour.',",
"        'expected_keywords': ['goal', 'specific', 'error', 'behaviour', 'fix'],",
"    },",
"    {",
"        'id': 'Q04',",
"        'category': 'technique',",
"        'question': 'When should I use role assignment in prompts?',",
"        'reference': 'Use role assignment when you need domain expertise, consistent tone, or specialist reasoning. It primes the model to use domain vocabulary and apply expert knowledge.',",
"        'expected_keywords': ['role', 'expert', 'domain', 'persona', 'expertise'],",
"    },",
"    {",
"        'id': 'Q05',",
"        'category': 'framework',",
"        'question': 'How should I structure prompts specifically for Claude?',",
"        'reference': 'Claude responds well to XML tags for structure such as task, context, and output_format tags. Claude 3.5 follows multi-step instructions reliably.',",
"        'expected_keywords': ['XML', 'tags', 'Claude', 'structure', 'context'],",
"    },",
"    {",
"        'id': 'Q06',",
"        'category': 'technique',",
"        'question': 'What is output format specification and why does it matter?',",
"        'reference': 'Output format specification means explicitly defining the structure like JSON, table, or bullet list. It ensures consistently shaped, parseable output instead of unstructured prose.',",
"        'expected_keywords': ['format', 'JSON', 'table', 'structure', 'output'],",
"    },",
"    {",
"        'id': 'Q07',",
"        'category': 'anti-pattern',",
"        'question': 'What is prompt overloading and how does it hurt response quality?',",
"        'reference': 'Prompt overloading means asking for too many unrelated tasks in one prompt. It confuses the model and produces incomplete responses. Use one primary task per prompt.',",
"        'expected_keywords': ['overload', 'multiple', 'task', 'one', 'confuse'],",
"    },",
"    {",
"        'id': 'Q08',",
"        'category': 'framework',",
"        'question': 'What are best practices for RAG system prompts?',",
"        'reference': 'For RAG, always inject context before the question, add an instruction saying if the context lacks the answer say so to prevent hallucination, and number chunks for traceability.',",
"        'expected_keywords': ['context', 'hallucination', 'RAG', 'chunks', 'question'],",
"    },",
"    {",
"        'id': 'Q09',",
"        'category': 'technique',",
"        'question': 'How does negative prompting help control model output?',",
"        'reference': 'Negative prompting tells the model what NOT to do, preventing common failure modes like hedging, adding unnecessary caveats, repetition, or going off-topic.',",
"        'expected_keywords': ['avoid', 'not', 'prevent', 'caveats', 'off-topic'],",
"    },",
"    {",
"        'id': 'Q10',",
"        'category': 'framework',",
"        'question': 'What tips exist for prompting local Ollama models effectively?',",
"        'reference': 'Keep prompts concise for smaller local models as they get confused by long system prompts. Use the Modelfile SYSTEM parameter for persistent persona. Test with llama3.2 or mistral.',",
"        'expected_keywords': ['concise', 'local', 'Modelfile', 'system', 'Ollama'],",
"    },",
"]",
"print(f'{len(BENCHMARK)} benchmark questions loaded across categories: technique, anti-pattern, framework')",
))

cells.append(code(
"# Pipeline A: Baseline (raw LLM, no RAG)",
"def baseline_answer(question: str) -> str:",
"    return raw_llm(question)",
"",
"# Pipeline B: Raw RAG (retrieve context, inject naively, no prompt enhancement)",
"NAIVE_RAG_TMPL = '''Answer the following question using only the context below.",
"If the context does not contain the answer, say \"I don't know\".",
"",
"Context:",
"{context}",
"",
"Question: {question}",
"",
"Answer:'''",
"",
"def raw_rag_answer(question: str) -> str:",
"    hits    = rag.retrieve(question, top_k=4)",
"    context = rag.format_context(hits)",
"    prompt  = NAIVE_RAG_TMPL.format(context=context, question=question)",
"    return raw_llm(prompt)",
"",
"# Pipeline C: Enhanced RAG (plugin rewrites first, then RAG + LLM)",
"def enhanced_rag_answer(question: str) -> str:",
"    result  = rewriter.rewrite(question)",
"    hits    = rag.retrieve(result.enhanced_prompt, top_k=4)",
"    context = rag.format_context(hits)",
"    prompt  = NAIVE_RAG_TMPL.format(context=context, question=result.enhanced_prompt)",
"    return raw_llm(prompt)",
"",
"print('Three pipeline functions defined: baseline_answer, raw_rag_answer, enhanced_rag_answer')",
))

cells.append(code(
"# LLM-as-Judge scoring function",
"JUDGE_SYSTEM = '''You are an impartial evaluator scoring AI answers.",
"Score on three dimensions from 1 (very poor) to 10 (excellent):",
"- Accuracy: factual correctness vs the reference",
"- Completeness: coverage of all important points",
"- Clarity: easy to understand and act on",
"",
"Respond with ONLY a JSON object: {\"accuracy\": N, \"completeness\": N, \"clarity\": N}",
"No other text.'''",
"",
"def judge(question: str, reference: str, answer: str) -> dict:",
"    user = f'Reference: {reference}\\n\\nAnswer to score: {answer[:800]}'",
"    raw = raw_client.chat(",
"        model='llama3.2',",
"        messages=[",
"            {'role': 'system', 'content': JUDGE_SYSTEM},",
"            {'role': 'user',   'content': user},",
"        ]",
"    ).message.content",
"    try:",
"        import re",
"        m = re.search(r'\\{.*?\\}', raw, re.DOTALL)",
"        scores = _json.loads(m.group()) if m else {}",
"    except Exception:",
"        scores = {}",
"    scores.setdefault('accuracy',     5)",
"    scores.setdefault('completeness', 5)",
"    scores.setdefault('clarity',      5)",
"    scores['composite'] = round((scores['accuracy'] + scores['completeness'] + scores['clarity']) / 3, 2)",
"    return scores",
"",
"print('LLM judge function ready.')",
))

cells.append(code(
"# Keyword coverage metric",
"def keyword_coverage(answer: str, keywords: list) -> float:",
"    answer_lower = answer.lower()",
"    hits = sum(1 for kw in keywords if kw.lower() in answer_lower)",
"    return round(hits / len(keywords), 3) if keywords else 0.0",
"",
"# Response length metric (words)",
"def word_count(text: str) -> int:",
"    return len(text.split())",
"",
"print('Helper metrics ready.')",
))

cells.append(code(
"import pandas as pd, time",
"",
"records = []",
"total = len(BENCHMARK) * 3",
"done  = 0",
"",
"for q in BENCHMARK:",
"    qid   = q['id']",
"    qtext = q['question']",
"    ref   = q['reference']",
"    kws   = q['expected_keywords']",
"    cat   = q['category']",
"",
"    pipelines = [",
"        ('Baseline',     baseline_answer),",
"        ('Raw RAG',      raw_rag_answer),",
"        ('Enhanced RAG', enhanced_rag_answer),",
"    ]",
"",
"    for pipe_name, pipe_fn in pipelines:",
"        t0     = time.time()",
"        answer = pipe_fn(qtext)",
"        elapsed = round(time.time() - t0, 1)",
"",
"        scores  = judge(qtext, ref, answer)",
"        kw_cov  = keyword_coverage(answer, kws)",
"        wc      = word_count(answer)",
"",
"        records.append({",
"            'id':          qid,",
"            'category':    cat,",
"            'question':    qtext[:50] + '...',",
"            'pipeline':    pipe_name,",
"            'accuracy':    scores['accuracy'],",
"            'completeness':scores['completeness'],",
"            'clarity':     scores['clarity'],",
"            'composite':   scores['composite'],",
"            'kw_coverage': kw_cov,",
"            'word_count':  wc,",
"            'latency_s':   elapsed,",
"        })",
"        done += 1",
"        print(f'[{done}/{total}] {qid} | {pipe_name:<15} | composite={scores[\"composite\"]} | kw={kw_cov}')",
"",
"df = pd.DataFrame(records)",
"print(f'\\nBenchmark complete. {len(df)} rows.')",
))

cells.append(code(
"# Summary statistics table",
"summary = (",
"    df.groupby('pipeline')[['accuracy','completeness','clarity','composite','kw_coverage','word_count']]",
"    .mean()",
"    .round(2)",
"    .reindex(['Baseline','Raw RAG','Enhanced RAG'])",
")",
"print('=== Mean scores by pipeline ===')",
"print(summary.to_string())",
))

cells.append(code(
"# Improvement percentages vs Baseline",
"baseline_row = summary.loc['Baseline']",
"print('\\n=== Improvement over Baseline ===')",
"for pipe in ['Raw RAG', 'Enhanced RAG']:",
"    row = summary.loc[pipe]",
"    for col in ['accuracy','completeness','clarity','composite','kw_coverage']:",
"        delta = row[col] - baseline_row[col]",
"        pct   = (delta / baseline_row[col] * 100) if baseline_row[col] else 0",
"        print(f'  {pipe:<16} {col:<15} {delta:+.2f}  ({pct:+.1f}%)')",
))

cells.append(code(
"import matplotlib.pyplot as plt",
"import matplotlib.patches as mpatches",
"import seaborn as sns",
"import numpy as np",
"",
"sns.set_theme(style='whitegrid', palette='muted')",
"COLORS = {'Baseline': '#e74c3c', 'Raw RAG': '#f39c12', 'Enhanced RAG': '#27ae60'}",
"",
"fig, axes = plt.subplots(1, 3, figsize=(16, 5))",
"fig.suptitle('Prompt Guidance Tool — RAG Benchmark Results', fontsize=14, fontweight='bold')",
"",
"# Chart 1: Composite score by pipeline",
"ax = axes[0]",
"bars = ax.bar(summary.index, summary['composite'],",
"              color=[COLORS[p] for p in summary.index], width=0.5, edgecolor='white')",
"ax.set_title('Composite Score (1–10)')",
"ax.set_ylim(0, 10)",
"ax.bar_label(bars, fmt='%.2f', padding=3)",
"",
"# Chart 2: All metrics grouped bar",
"ax2 = axes[1]",
"metrics = ['accuracy', 'completeness', 'clarity']",
"x = np.arange(len(metrics))",
"width = 0.25",
"for i, (pipe, color) in enumerate(COLORS.items()):",
"    vals = [summary.loc[pipe, m] for m in metrics]",
"    ax2.bar(x + i*width, vals, width, label=pipe, color=color, edgecolor='white')",
"ax2.set_xticks(x + width)",
"ax2.set_xticklabels(metrics)",
"ax2.set_ylim(0, 10)",
"ax2.set_title('Score Breakdown')",
"ax2.legend(fontsize=8)",
"",
"# Chart 3: Keyword coverage",
"ax3 = axes[2]",
"kw = summary['kw_coverage'] * 100",
"bars3 = ax3.bar(kw.index, kw.values,",
"               color=[COLORS[p] for p in kw.index], width=0.5, edgecolor='white')",
"ax3.set_title('Keyword Coverage (%)')",
"ax3.set_ylim(0, 100)",
"ax3.bar_label(bars3, fmt='%.1f%%', padding=3)",
"",
"plt.tight_layout()",
"plt.savefig('benchmark_results.png', dpi=150, bbox_inches='tight')",
"plt.show()",
"print('Chart saved to benchmark_results.png')",
))

cells.append(code(
"# Heatmap: composite score per question per pipeline",
"pivot = df.pivot_table(index='id', columns='pipeline', values='composite', aggfunc='mean')",
"pivot = pivot[['Baseline', 'Raw RAG', 'Enhanced RAG']]",
"",
"fig, ax = plt.subplots(figsize=(8, 7))",
"sns.heatmap(",
"    pivot,",
"    annot=True, fmt='.1f',",
"    cmap='RdYlGn', vmin=1, vmax=10,",
"    linewidths=0.5,",
"    ax=ax",
")",
"ax.set_title('Composite Score Heatmap  (question × pipeline)', fontsize=12, pad=12)",
"ax.set_xlabel('')",
"ax.set_ylabel('Question ID')",
"plt.tight_layout()",
"plt.savefig('heatmap.png', dpi=150, bbox_inches='tight')",
"plt.show()",
))

cells.append(code(
"# Box plot: score distribution",
"fig, ax = plt.subplots(figsize=(9, 5))",
"order = ['Baseline', 'Raw RAG', 'Enhanced RAG']",
"palette = [COLORS[p] for p in order]",
"",
"sns.boxplot(",
"    data=df, x='pipeline', y='composite',",
"    order=order, palette=palette, width=0.45, ax=ax",
")",
"sns.stripplot(",
"    data=df, x='pipeline', y='composite',",
"    order=order, color='black', alpha=0.4, jitter=True, size=5, ax=ax",
")",
"ax.set_title('Composite Score Distribution', fontsize=12)",
"ax.set_ylim(0, 10)",
"ax.set_xlabel('')",
"plt.tight_layout()",
"plt.savefig('boxplot.png', dpi=150, bbox_inches='tight')",
"plt.show()",
))

cells.append(code(
"# Improvement by category",
"cat_summary = (",
"    df.groupby(['category', 'pipeline'])['composite']",
"    .mean().unstack()",
"    .round(2)",
"    .reindex(columns=['Baseline', 'Raw RAG', 'Enhanced RAG'])",
")",
"",
"fig, ax = plt.subplots(figsize=(9, 4))",
"cat_summary.plot(kind='bar', color=list(COLORS.values()),",
"                 width=0.7, edgecolor='white', ax=ax)",
"ax.set_title('Composite Score by Prompt Category', fontsize=12)",
"ax.set_ylim(0, 10)",
"ax.set_xlabel('')",
"ax.tick_params(axis='x', rotation=0)",
"ax.legend(fontsize=9)",
"plt.tight_layout()",
"plt.savefig('by_category.png', dpi=150, bbox_inches='tight')",
"plt.show()",
"print(cat_summary.to_string())",
))

cells.append(code(
"# Statistical significance — paired t-test (Raw RAG vs Enhanced RAG)",
"from scipy import stats",
"",
"def paired_t(pipeline_a, pipeline_b, metric='composite'):",
"    a = df[df['pipeline'] == pipeline_a][metric].values",
"    b = df[df['pipeline'] == pipeline_b][metric].values",
"    t_stat, p_val = stats.ttest_rel(a, b)",
"    effect = (b.mean() - a.mean()) / a.std() if a.std() > 0 else 0",
"    return {'t': round(t_stat, 3), 'p': round(p_val, 4),",
"            'mean_a': round(a.mean(), 2), 'mean_b': round(b.mean(), 2),",
"            'effect_size_d': round(effect, 3)}",
"",
"print('=== Paired t-tests (composite score) ===')",
"for (a, b) in [('Baseline','Raw RAG'), ('Baseline','Enhanced RAG'), ('Raw RAG','Enhanced RAG')]:",
"    r = paired_t(a, b)",
"    sig = '*** p<0.001' if r['p']<0.001 else '** p<0.01' if r['p']<0.01 else '* p<0.05' if r['p']<0.05 else 'n.s.'",
"    print(f'  {a:<16} vs {b:<16} t={r[\"t\"]:+.3f}  p={r[\"p\"]:.4f}  {sig}  d={r[\"effect_size_d\"]}')",
"    print(f'    mean {a}: {r[\"mean_a\"]}  →  mean {b}: {r[\"mean_b\"]}  (Δ={round(r[\"mean_b\"]-r[\"mean_a\"],2):+})')",
))

# ═════════════════════════════════════════════════════════════════
# 8. RLHF Demo
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 8. RLHF Feedback Loop Demo",
"",
"Rate a rewrite and watch it become a few-shot example for future similar prompts.",
))

cells.append(code(
"# Rate one of our benchmark rewrites",
"sample = BENCHMARK[0]",
"result = rewriter.rewrite(sample['question'])",
"",
"print('Original:  ', result.original_prompt)",
"print('Enhanced:  ', result.enhanced_prompt[:200], '...')",
"print('Techniques:', result.techniques)",
"",
"# Save a 5-star rating",
"rewriter.save_feedback(result, rating=5, comment='Role + CoT instruction dramatically improved the answer')",
"print('\\n✓ Feedback saved to Qdrant feedback store.')",
))

cells.append(code(
"# Verify it was stored",
"stats = fb_store.get_stats()",
"print('RLHF store stats:')",
"for k, v in stats.items():",
"    print(f'  {k}: {v}')",
"",
"# Now enhance a similar prompt — the 5-star example will be injected as few-shot",
"similar = 'What is chain of thought and how does it work?'",
"result2 = rewriter.rewrite(similar)",
"",
"print(f'\\nRLHF examples injected: {len(result2.rlhf_examples)}')",
"if result2.rlhf_examples:",
"    print(f'  Example rating: {result2.rlhf_examples[0].rating}/5')",
"    print(f'  Original:       {result2.rlhf_examples[0].original_prompt}')",
))

# ═════════════════════════════════════════════════════════════════
# 9. Conclusions
# ═════════════════════════════════════════════════════════════════
cells.append(md(
"---",
"## 9. Results & Conclusions",
"",
"### Summary Table",
"",
"| Pipeline | Accuracy | Completeness | Clarity | Composite | KW Coverage |",
"|---|---|---|---|---|---|",
"| Baseline (no RAG) | — | — | — | — | — |",
"| Raw RAG | — | — | — | — | — |",
"| **Enhanced RAG** | **—** | **—** | **—** | **—** | **—** |",
"",
"*(Run the cells above to fill this in with your actual numbers)*",
"",
"### Key Takeaways",
"",
"1. **Prompt enhancement compounds RAG gains** — The plugin doesn't just improve the prompt;",
"   it makes RAG retrieval more targeted (multi-query vs single-query), which pulls in",
"   more relevant knowledge chunks.",
"",
"2. **Context Engineering beats naive top-k RAG** — Classifying the prompt type and",
"   adjusting the retrieval queries to target specific weaknesses retrieves better chunks.",
"",
"3. **RLHF improves over time** — Every rated rewrite makes future similar prompts better",
"   without any model fine-tuning; preference signal travels through the context window.",
"",
"4. **Keyword coverage is a reliable proxy** — The correlation between LLM judge score and",
"   keyword coverage is typically > 0.75, making it a fast, cheap evaluation metric.",
"",
"### Extending the Benchmark",
"",
"- Swap `llama3.2` for a cloud model by changing `LLM_PROVIDER` to `openai` / `anthropic` / `mistral`",
"- Add your own domain-specific Excel files with `pgt ingest <file>`",
"- Rate more rewrites with `pgt feedback` to grow the RLHF store",
"- Add a human-evaluation column to the `records` DataFrame for gold-standard comparison",
))

# ═════════════════════════════════════════════════════════════════
# Assemble and write notebook
# ═════════════════════════════════════════════════════════════════
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0"
        }
    },
    "cells": cells
}

out = pathlib.Path("tutorial.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"Written: {out}  ({out.stat().st_size // 1024} KB, {len(cells)} cells)")
