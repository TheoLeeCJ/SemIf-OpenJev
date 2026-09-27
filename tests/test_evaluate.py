import importlib.util
import math
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "openjev_benchmark_evaluate", Path(__file__).parents[1] / "benchmarks" / "evaluate.py"
)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evaluator)


OPTIONS = [
    {"id": "refund", "description": "Issue a refund."},
    {"id": "replace", "description": "Send a replacement."},
    {"id": "explain", "description": "Explain the policy."},
]


def answer_set_row(**changes):
    row = {
        "id": "answer-set-row",
        "group_id": "faq-source",
        "family": "faq",
        "options": OPTIONS,
        "answer_set": ["refund"],
        "target_distribution": {"refund": 1.0, "replace": 0.0, "explain": 0.0},
    }
    row.update(changes)
    return row


def distribution(row_id, probabilities):
    return {"id": row_id, "probabilities": probabilities}


def test_legacy_hard_label_alignment_and_report_keep_the_legacy_shape():
    gold = [{
        "id": "legacy-hard",
        "group_id": "source",
        "family": "faq",
        "options": OPTIONS[:2],
        "label": 0,
        "target_distribution": None,
    }]
    predictions = [distribution("legacy-hard", {"refund": 0.75, "replace": 0.25})]

    assert evaluator.align(gold, predictions) == [{
        "id": "legacy-hard",
        "group_id": "source",
        "family": "faq",
        "gold_id": "refund",
        "predicted_id": "refund",
        "correct": True,
        "confidence": 0.75,
        "nll": -math.log(0.75),
        "brier": 0.125,
        "status": "distribution",
        "probabilities": [0.75, 0.25],
    }]

    report = evaluator.evaluate(gold, predictions)
    assert set(report) == {
        "available_gold", "scored", "evaluated", "coverage", "missing", "invalid",
        "family_results", "mean_family_balanced_accuracy", "mean_family_macro_f1",
        "analytic_uncertainty", "errors", "limitations",
    }
    assert report["family_results"]["faq"]["n"] == 1
    assert report["family_results"]["faq"]["nll"] == -math.log(0.75)
    assert "answer_set_cohort" not in report


def test_answer_set_cohort_reports_singleton_and_multi_answer_metrics():
    singleton = answer_set_row(id="singleton", group_id="single-source")
    multi = answer_set_row(
        id="multi",
        group_id="multi-source",
        answer_set=["refund", "replace"],
        target_distribution={"refund": 0.6, "replace": 0.4, "explain": 0.0},
    )
    predictions = [
        distribution("singleton", {"refund": 0.7, "replace": 0.2, "explain": 0.1}),
        distribution("multi", {"refund": 0.2, "replace": 0.5, "explain": 0.3}),
    ]

    rows = evaluator.align([singleton, multi], predictions)
    assert [row["top_1_in_set"] for row in rows] == [True, True]
    assert rows[0]["acceptable_set_probability_mass"] == pytest.approx(0.7)
    assert rows[1]["acceptable_set_probability_mass"] == pytest.approx(0.7)
    assert rows[0]["expected_nll"] == pytest.approx(-math.log(0.7))
    assert rows[1]["expected_nll"] == pytest.approx(-0.6 * math.log(0.2) - 0.4 * math.log(0.5))
    assert rows[0]["expected_brier"] == pytest.approx(0.14)
    assert rows[1]["expected_brier"] == pytest.approx(0.74)

    cohort = evaluator.evaluate([singleton, multi], predictions)["answer_set_cohort"]
    assert cohort["n"] == 2
    assert cohort["top_1_in_set"] == 1.0
    assert cohort["probability_coverage"] == 1.0
    assert cohort["expected_nll"] == pytest.approx((rows[0]["expected_nll"] + rows[1]["expected_nll"]) / 2)
    assert cohort["brier"] == pytest.approx((0.14 + 0.74) / 2)
    assert cohort["acceptable_set_probability_mass"] == pytest.approx(0.7)


@pytest.mark.parametrize("target_distribution", [
    {"refund": 0.8, "replace": 0.0, "explain": 0.0},
    {"refund": 0.5, "replace": 0.5, "explain": 0.0},
    {"refund": float("nan"), "replace": 0.0, "explain": 0.0},
    {"refund": 1.1, "replace": -0.1, "explain": 0.0},
])
def test_answer_set_contract_rejects_malformed_target_distributions(target_distribution):
    row = answer_set_row(id="bad-distribution", target_distribution=target_distribution)

    with pytest.raises(ValueError, match="bad-distribution"):
        evaluator.align([row], [])


def test_answer_set_contract_uses_the_declared_normalization_tolerance():
    row = answer_set_row(
        id="normalization-tolerance",
        target_distribution={"refund": 0.99995, "replace": 0.0, "explain": 0.0},
    )

    aligned = evaluator.align(
        [row],
        [distribution("normalization-tolerance", {"refund": 0.8, "replace": 0.1, "explain": 0.1})],
    )[0]
    assert aligned["expected_nll"] == pytest.approx(-math.log(0.8))
    assert aligned["expected_brier"] == pytest.approx(0.06)


def test_answer_set_top_one_ties_are_order_invariant_and_conservative():
    original = answer_set_row(
        id="tie-original",
        answer_set=["refund"],
        target_distribution={"refund": 1.0, "replace": 0.0, "explain": 0.0},
    )
    reordered = answer_set_row(
        id="tie-reordered",
        options=[OPTIONS[2], OPTIONS[0], OPTIONS[1]],
        answer_set=["refund"],
        target_distribution={"explain": 0.0, "refund": 1.0, "replace": 0.0},
    )
    predictions = [
        distribution("tie-original", {"refund": 0.5, "replace": 0.0, "explain": 0.5}),
        distribution("tie-reordered", {"explain": 0.5, "refund": 0.5, "replace": 0.0}),
    ]

    rows = evaluator.align([original, reordered], predictions)
    assert [row["predicted_id"] for row in rows] == ["refund", "explain"]
    assert [row["top_1_in_set"] for row in rows] == [False, False]


def test_answer_set_contract_requires_a_unique_set_and_distribution():
    duplicate = answer_set_row(id="duplicate-answer-set", answer_set=["refund", "refund"])
    missing_target = answer_set_row(id="missing-target")
    missing_target.pop("target_distribution")

    for invalid in (duplicate, missing_target):
        with pytest.raises(ValueError, match=invalid["id"]):
            evaluator.align([invalid], [])


def test_singleton_hard_projection_must_be_complete_and_consistent():
    projected = answer_set_row(
        id="projected",
        label=0,
        target_option_id="refund",
        target_letter="A",
    )
    row = evaluator.align(
        [projected],
        [distribution("projected", {"refund": 0.8, "replace": 0.1, "explain": 0.1})],
    )[0]
    assert row["gold_id"] == "refund"
    assert row["top_1_in_set"] is True
    assert row["expected_nll"] == pytest.approx(-math.log(0.8))
    evaluator.require_hard_label_gold([projected], "Screening gate")
    assert evaluator.paired_comparison([row], [row], samples=40)["difference"] == 0.0

    partial = answer_set_row(id="partial-projection", label=0)
    mismatched_id = answer_set_row(
        id="mismatched-id",
        label=1,
        target_option_id="refund",
        target_letter="B",
    )
    mismatched_letter = answer_set_row(
        id="mismatched-letter",
        label=0,
        target_option_id="refund",
        target_letter="B",
    )
    multi_with_projection = answer_set_row(
        id="multi-projection",
        answer_set=["refund", "replace"],
        target_distribution={"refund": 0.5, "replace": 0.5, "explain": 0.0},
        label=0,
        target_option_id="refund",
        target_letter="A",
    )
    for invalid in (partial, mismatched_id, mismatched_letter, multi_with_projection):
        with pytest.raises(ValueError, match=invalid["id"]):
            evaluator.align([invalid], [])


def test_native_answer_set_decisions_only_contribute_top_one_in_set():
    gold = answer_set_row(
        id="native",
        answer_set=["refund", "replace"],
        target_distribution={"refund": 0.5, "replace": 0.5, "explain": 0.0},
    )
    row = evaluator.align([gold], [{"id": "native", "prediction_id": "replace"}])[0]
    assert row["status"] == "native_decision"
    assert row["probabilities"] is None
    assert row["top_1_in_set"] is True
    assert row["expected_nll"] is None
    assert row["expected_brier"] is None
    assert row["acceptable_set_probability_mass"] is None

    cohort = evaluator.evaluate([gold], [{"id": "native", "prediction_id": "replace"}])["answer_set_cohort"]
    assert cohort["top_1_in_set"] == 1.0
    assert cohort["probability_rows"] == 0
    assert cohort["expected_nll"] is None
    assert cohort["brier"] is None
    assert cohort["acceptable_set_probability_mass"] is None


def test_native_singleton_projection_keeps_hard_classification_without_probability_metrics():
    gold = answer_set_row(
        id="native-projected",
        label=0,
        target_option_id="refund",
        target_letter="A",
    )
    prediction = {"id": "native-projected", "prediction_id": "refund"}

    row = evaluator.align([gold], [prediction])[0]
    assert row["gold_id"] == "refund"
    assert row["top_1_in_set"] is True
    assert row["probabilities"] is None
    assert row["nll"] is None
    assert row["expected_nll"] is None
    report = evaluator.evaluate([gold], [prediction])
    assert report["family_results"]["faq"]["accuracy"] == 1.0
    assert report["answer_set_cohort"]["expected_nll"] is None


def test_hard_only_paths_reject_multi_answer_rows_but_accept_projected_singletons():
    multi = answer_set_row(
        id="multi-hard-path",
        answer_set=["refund", "replace"],
        target_distribution={"refund": 0.5, "replace": 0.5, "explain": 0.0},
    )
    with pytest.raises(ValueError, match="multi-hard-path"):
        evaluator.screening_gate([multi], [distribution("multi-hard-path", {"refund": 0.5, "replace": 0.5, "explain": 0.0})])

    with pytest.raises(ValueError, match="unprojected-singleton"):
        evaluator.require_hard_label_gold([answer_set_row(id="unprojected-singleton")], "Screening gate")

    projected = answer_set_row(
        id="singleton-hard-path",
        label=0,
        target_option_id="refund",
        target_letter="A",
    )
    predictions = [distribution("singleton-hard-path", {"refund": 0.9, "replace": 0.05, "explain": 0.05})]
    report = evaluator.evaluate([projected], predictions, comparison=predictions)
    assert report["paired_comparison"]["difference"] == 0.0
