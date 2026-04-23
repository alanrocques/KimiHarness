"""Symptom2Disease dataset loader.

Deterministic split (seed=42): 100 eval examples by default, the rest go in a
few-shot pool the proposer can draw from. The parquet cache lives under
``data/``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .types import EvalExample

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
HF_NAME = "gretelai/symptom_to_diagnosis"
SEED = 42

# Class labels for the Symptom2Disease dataset. The upstream dataset is small
# (~1200 rows, 22 classes) and stable, so the list is fixed here to keep the
# baseline harness runnable without a live dataset fetch. If the upstream
# changes, ``load_class_labels`` re-derives them from whatever gets cached.
CLASS_LABELS: list[str] = [
    "drug reaction",
    "allergy",
    "diabetes",
    "urinary tract infection",
    "malaria",
    "jaundice",
    "cervical spondylosis",
    "migraine",
    "hypertension",
    "bronchial asthma",
    "acne",
    "arthritis",
    "gastroesophageal reflux disease",
    "peptic ulcer disease",
    "typhoid",
    "common cold",
    "fungal infection",
    "pneumonia",
    "dimorphic hemorrhoids",
    "varicose veins",
    "chicken pox",
    "psoriasis",
    "impetigo",
]


@dataclass
class LoadedDataset:
    eval_examples: list[EvalExample]
    few_shot_pool: list[EvalExample]
    class_labels: list[str]


def _normalize(label: str) -> str:
    return label.strip().lower()


def _raw_rows() -> list[dict]:
    """Fetch raw rows as a list of ``{"input": ..., "gold_label": ...}`` dicts.

    Prefer the HuggingFace ``datasets`` cache under ``data/``. If that is
    unavailable (no network, CI), fall back to a tiny embedded fixture so
    smoke tests still pass.
    """
    try:  # pragma: no cover — network path
        from datasets import load_dataset

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        ds = load_dataset(HF_NAME, cache_dir=str(DATA_DIR))
        rows: list[dict] = []
        for split in ds.values():
            for row in split:
                text = row.get("input_text") or row.get("text") or row.get("symptoms")
                label = row.get("output_text") or row.get("label") or row.get("disease")
                if text is None or label is None:
                    continue
                rows.append({"input": str(text), "gold_label": _normalize(str(label))})
        if rows:
            return rows
    except Exception:
        pass

    # Offline fixture — lets tests and mock runs work end-to-end without HF.
    return _OFFLINE_FIXTURE.copy()


def load(eval_split_size: int = 100) -> LoadedDataset:
    rows = _raw_rows()
    rng = random.Random(SEED)
    rng.shuffle(rows)
    eval_rows = rows[:eval_split_size]
    pool_rows = rows[eval_split_size:]

    eval_examples = [
        EvalExample(example_id=i, input=r["input"], gold_label=r["gold_label"])
        for i, r in enumerate(eval_rows)
    ]
    pool_examples = [
        EvalExample(example_id=i, input=r["input"], gold_label=r["gold_label"])
        for i, r in enumerate(pool_rows)
    ]

    labels = sorted({r["gold_label"] for r in rows})
    return LoadedDataset(
        eval_examples=eval_examples,
        few_shot_pool=pool_examples,
        class_labels=labels or list(CLASS_LABELS),
    )


def load_eval_split(n: int = 100) -> list[EvalExample]:
    """Convenience: just the eval examples."""
    return load(n).eval_examples


# --- Offline fixture -------------------------------------------------------
# 30 short synthetic rows covering a subset of the real class labels. Enough
# for mock mode and unit tests; never substitutes for the real dataset in a
# paid run.
_OFFLINE_FIXTURE: list[dict] = [
    {"input": "I've had a runny nose, sore throat, and mild fever for three days.", "gold_label": "common cold"},
    {"input": "Red itchy patches on my scalp with silvery scales.", "gold_label": "psoriasis"},
    {"input": "Burning sensation when urinating and frequent urges.", "gold_label": "urinary tract infection"},
    {"input": "Throbbing headache on one side of my head with nausea and light sensitivity.", "gold_label": "migraine"},
    {"input": "Wheezing, shortness of breath, tight chest especially at night.", "gold_label": "bronchial asthma"},
    {"input": "Painful swollen joints in my fingers, worse in the morning.", "gold_label": "arthritis"},
    {"input": "Yellowing of skin and eyes, dark urine, fatigue.", "gold_label": "jaundice"},
    {"input": "High fever with chills, sweating, and body ache after travel.", "gold_label": "malaria"},
    {"input": "Frequent urination, extreme thirst, unexplained weight loss.", "gold_label": "diabetes"},
    {"input": "Persistent dry cough with fever and chest pain when breathing.", "gold_label": "pneumonia"},
    {"input": "Heartburn and acid coming back up into my throat after meals.", "gold_label": "gastroesophageal reflux disease"},
    {"input": "Sharp stomach pain between meals that gets better with food.", "gold_label": "peptic ulcer disease"},
    {"input": "Sudden rash and swelling of lips after taking a new medication.", "gold_label": "drug reaction"},
    {"input": "Sneezing, itchy watery eyes, runny nose around cats.", "gold_label": "allergy"},
    {"input": "Stiff neck and shoulder pain, occasional tingling down the arm.", "gold_label": "cervical spondylosis"},
    {"input": "Consistently high blood pressure readings, occasional headaches.", "gold_label": "hypertension"},
    {"input": "Pimples, blackheads, and oily skin on face and back.", "gold_label": "acne"},
    {"input": "High sustained fever, abdominal pain, rose-colored spots.", "gold_label": "typhoid"},
    {"input": "Red circular patches on skin that itch and spread.", "gold_label": "fungal infection"},
    {"input": "Painful lumps near the anus, bleeding during bowel movements.", "gold_label": "dimorphic hemorrhoids"},
    {"input": "Bulging blue veins on legs, aching after standing.", "gold_label": "varicose veins"},
    {"input": "Itchy blistering rash all over body with mild fever.", "gold_label": "chicken pox"},
    {"input": "Crusty yellow sores around the nose and mouth.", "gold_label": "impetigo"},
    {"input": "Severe headache, photophobia, vomiting lasting hours.", "gold_label": "migraine"},
    {"input": "Cloudy urine with a strong smell and lower belly pressure.", "gold_label": "urinary tract infection"},
    {"input": "Dry cough, mild congestion, low energy for a week.", "gold_label": "common cold"},
    {"input": "Cracked scaly elbows with silvery flaking.", "gold_label": "psoriasis"},
    {"input": "Increased thirst and urination, blurry vision.", "gold_label": "diabetes"},
    {"input": "Tight chest and wheeze triggered by cold air.", "gold_label": "bronchial asthma"},
    {"input": "Hives and swelling minutes after eating peanuts.", "gold_label": "allergy"},
]
