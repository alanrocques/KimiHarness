"""Baseline harness: zero-shot single LLM call.

Scored as iteration 0 of every run. The proposer sees this file (along with
its trace and score) starting at iteration 1.
"""

from __future__ import annotations

CLASS_LABELS = [
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


def classify(text: str, target_client, target_model: str) -> str:
    prompt = (
        "Classify the following patient symptom description into exactly one "
        f"of these diagnoses: {', '.join(CLASS_LABELS)}.\n\n"
        f"Symptoms: {text}\n\n"
        "Respond with just the diagnosis name, nothing else."
    )
    response = target_client.messages.create(
        model=target_model,
        max_tokens=64,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()
