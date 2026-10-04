from lineage_robustness.dwbench import DWGraph, compare_lineage_gold, generate_lineage_questions


def graph():
    return DWGraph(
        dataset="toy",
        names=("src", "mid", "fact", "child"),
        fk_edges=frozenset({(3, 2)}),
        lineage_edges=frozenset({(0, 1), (1, 2)}),
    )


def test_lineage_semantics_include_transitive_and_fk_children():
    rows = generate_lineage_questions(graph())
    trans = [r for r in rows if r["subtype"] == "transitive" and "src" in r["question"]]
    assert len(trans) == 1
    assert trans[0]["answer"] == ["fact", "mid"]

    combined = [r for r in rows if r["subtype"] == "combined_impact" and r["question"].startswith("If src ")]
    assert len(combined) == 1
    assert combined[0]["answer"] == ["child", "fact", "mid"]


def test_pristine_gold_exact_match():
    gold = generate_lineage_questions(graph())
    result = compare_lineage_gold(graph(), gold)
    assert result["exact"] is True
    assert result["missing"] == 0
    assert result["extra"] == 0


def test_parity_detects_answer_drift():
    gold = generate_lineage_questions(graph())
    gold[0] = {**gold[0], "answer": ["wrong"]}
    result = compare_lineage_gold(graph(), gold)
    assert result["exact"] is False
    assert result["missing"] == 1
    assert result["extra"] == 1

