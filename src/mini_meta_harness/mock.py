"""Mock proposer and target clients for zero-API-cost smoke runs.

Activated with ``mmh run --mock``. The entire pipeline — dataset load,
iteration, evaluator, trace writing, summary, cost report — runs, but no HTTP
calls are made. This matters for two reasons:

1. CI runs the mock end-to-end to catch regressions without Moonshot keys.
2. Recruiters skimming the repo can ``uv run mmh run --iterations 3 --mock``
   and see the system work without signing up for anything.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .dataset import CLASS_LABELS
from .types import EvalExample, FilesystemRead, ProposerResult

# --- Mock target client ----------------------------------------------------


class _MockMessage:
    def __init__(self, text: str, input_tokens: int, output_tokens: int):
        self.content = [_MockContentBlock(text)]
        self.usage = _MockUsage(input_tokens, output_tokens)
        self.stop_reason = "end_turn"


class _MockContentBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


@dataclass
class _MockUsage:
    input_tokens: int
    output_tokens: int


class MockTargetClient:
    """A stand-in for an Anthropic client the harness can call.

    Produces deterministic labels based on a hash of the full prompt. Different
    harnesses build different prompts, so accuracy naturally differs between
    iterations without any per-iteration wiring.
    """

    def __init__(self, gold_map: dict[str, str] | None = None, class_labels: list[str] | None = None):
        self._gold_map = gold_map or {}
        self._labels = class_labels or list(CLASS_LABELS)
        self.messages = _MockMessages(self)

    def register_gold(self, examples: list[EvalExample]) -> None:
        for ex in examples:
            self._gold_map[ex.input] = ex.gold_label

    def _predict(self, prompt: str) -> str:
        digest = hashlib.md5(prompt.encode("utf-8")).digest()
        r = int.from_bytes(digest[:4], "big") / 2**32
        # Find which example's text is embedded in the prompt.
        matched_gold: str | None = None
        for text, gold in self._gold_map.items():
            if text and text in prompt:
                matched_gold = gold
                break
        if matched_gold is None:
            return self._labels[int.from_bytes(digest[4:8], "big") % len(self._labels)]
        # Give ~60% correctness on average, drifting with prompt shape.
        if r < 0.6:
            return matched_gold
        return self._labels[int.from_bytes(digest[4:8], "big") % len(self._labels)]


class _MockMessages:
    def __init__(self, parent: MockTargetClient):
        self._parent = parent

    def create(self, *, model: str, max_tokens: int, messages: list[dict], **_: Any):
        prompt_text = ""
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, str):
                prompt_text += c
            elif isinstance(c, list):
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        prompt_text += block.get("text", "")
        label = self._parent._predict(prompt_text)
        # Fake token counts roughly proportional to prompt length.
        input_tokens = max(1, len(prompt_text) // 4)
        output_tokens = max(1, len(label) // 4 + 1)
        return _MockMessage(label, input_tokens, output_tokens)


# --- Mock proposer ---------------------------------------------------------


_MOCK_HARNESSES: list[tuple[str, str]] = [
    # (reasoning, code)
    (
        "Baseline beaten by margin-of-error; try adding three diverse few-shot "
        "examples drawn from the most confused classes in the iteration 0 trace "
        "(common cold vs pneumonia, migraine vs hypertension).",
        '''from anthropic import Anthropic

CLASS_LABELS = [
    "drug reaction", "allergy", "diabetes", "urinary tract infection",
    "malaria", "jaundice", "cervical spondylosis", "migraine", "hypertension",
    "bronchial asthma", "acne", "arthritis", "gastroesophageal reflux disease",
    "peptic ulcer disease", "typhoid", "common cold", "fungal infection",
    "pneumonia", "dimorphic hemorrhoids", "varicose veins", "chicken pox",
    "psoriasis", "impetigo",
]

FEW_SHOT = [
    ("Fever with productive cough and chest pain worsening on inhalation.", "pneumonia"),
    ("Pounding one-sided headache with photophobia and nausea.", "migraine"),
    ("Hives after eating shellfish.", "allergy"),
]


def classify(text, target_client, target_model):
    shots = "\\n\\n".join(f"Symptoms: {s}\\nDiagnosis: {d}" for s, d in FEW_SHOT)
    prompt = (
        "You classify patient symptoms into one of these diagnoses: "
        + ", ".join(CLASS_LABELS)
        + ".\\n\\nExamples:\\n" + shots
        + f"\\n\\nSymptoms: {text}\\nDiagnosis:"
    )
    resp = target_client.messages.create(
        model=target_model, max_tokens=32,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip().lower()
''',
    ),
    (
        "Few-shot helped on cold-vs-pneumonia but hurt on GI cases. Add a short "
        "chain-of-thought step before committing to a label.",
        '''CLASS_LABELS = [
    "drug reaction", "allergy", "diabetes", "urinary tract infection",
    "malaria", "jaundice", "cervical spondylosis", "migraine", "hypertension",
    "bronchial asthma", "acne", "arthritis", "gastroesophageal reflux disease",
    "peptic ulcer disease", "typhoid", "common cold", "fungal infection",
    "pneumonia", "dimorphic hemorrhoids", "varicose veins", "chicken pox",
    "psoriasis", "impetigo",
]


def classify(text, target_client, target_model):
    prompt = (
        "Think step by step about the symptoms, then commit to one diagnosis "
        "from this list: " + ", ".join(CLASS_LABELS) + ".\\n"
        "Format your answer as:\\nReasoning: <one sentence>\\nDiagnosis: <label>\\n\\n"
        f"Symptoms: {text}"
    )
    resp = target_client.messages.create(
        model=target_model, max_tokens=128,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text.strip()
    for line in raw.splitlines()[::-1]:
        low = line.lower()
        if "diagnosis:" in low:
            return low.split("diagnosis:", 1)[1].strip(" .*")
    return raw.splitlines()[-1].strip().lower()
''',
    ),
    (
        "CoT output is sometimes missing the 'Diagnosis:' line. Add a second "
        "validation call that forces a label from the fixed list.",
        '''CLASS_LABELS = [
    "drug reaction", "allergy", "diabetes", "urinary tract infection",
    "malaria", "jaundice", "cervical spondylosis", "migraine", "hypertension",
    "bronchial asthma", "acne", "arthritis", "gastroesophageal reflux disease",
    "peptic ulcer disease", "typhoid", "common cold", "fungal infection",
    "pneumonia", "dimorphic hemorrhoids", "varicose veins", "chicken pox",
    "psoriasis", "impetigo",
]


def _ask(target_client, model, prompt, max_tokens=128):
    resp = target_client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def classify(text, target_client, target_model):
    draft = _ask(target_client, target_model,
        f"Patient symptoms: {text}\\nThink briefly, then write one diagnosis from: "
        + ", ".join(CLASS_LABELS))
    final = _ask(target_client, target_model,
        f"Patient symptoms: {text}\\nDraft answer: {draft}\\n\\n"
        "Return only one diagnosis label, exactly as written in this list: "
        + ", ".join(CLASS_LABELS), max_tokens=16)
    return final.strip().lower()
''',
    ),
]


@dataclass
class MockProposer:
    """Returns canned ProposerResult objects in sequence.

    Cycles through ``_MOCK_HARNESSES`` if the run asks for more iterations than
    we have canned responses for.
    """

    memory: Any
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    def propose(self, iteration_index: int) -> ProposerResult:
        reasoning, code = _MOCK_HARNESSES[(iteration_index - 1) % len(_MOCK_HARNESSES)]
        # Pretend the mock proposer read the best prior iteration.
        reads: list[FilesystemRead] = []
        prior = self.memory.list_iterations() if self.memory else []
        if prior:
            # Actually exercise the memory object so reads are recorded.
            best = prior[-1]
            try:
                self.memory.read_harness(best)
                self.memory.read_eval_trace(best, only_failures=True)
            except Exception:
                pass
            reads = self.memory.pop_reads()
        # Pretend input/output tokens for cost accounting.
        return ProposerResult(
            harness_code=code,
            reasoning=reasoning,
            filesystem_reads=reads,
            proposer_input_tokens=1200 + 200 * iteration_index,
            proposer_output_tokens=400 + 50 * iteration_index,
            steps_taken=3,
        )


def write_mock_fixtures(path: Path) -> None:
    """Dump the canned harnesses to a JSON file (used by tests for reference)."""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [{"reasoning": r, "code": c} for r, c in _MOCK_HARNESSES], indent=2
        )
    )
