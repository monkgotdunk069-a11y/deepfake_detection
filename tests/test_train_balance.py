from src.train import compute_class_balance


def test_compute_class_balance_detects_imbalance():
    items = [
        ("real_1.mp4", 0), ("real_2.mp4", 0), ("real_3.mp4", 0),
        ("real_4.mp4", 0), ("real_5.mp4", 0),
        ("fake_1.mp4", 1),
    ]

    result = compute_class_balance(items)

    assert result["counts"] == {0: 5, 1: 1}
    assert result["is_imbalanced"] is True
    assert result["imbalance_ratio"] == 5.0
    assert result["pos_weight"] == 5.0
    assert result["neg_weight"] == 0.2
