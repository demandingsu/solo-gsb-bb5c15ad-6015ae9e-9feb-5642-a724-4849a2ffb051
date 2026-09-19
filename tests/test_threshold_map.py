"""阈值扫描网格接口端到端测试与二维前缀和聚合单测。"""

from __future__ import annotations

import os
import random

import pytest

from app.threshold_map import threshold_map_cells


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("threshold_map_db")
    os.environ["PRIMER_DB_PATH"] = str(tmp_dir / "test.db")

    from app import db, main

    db.init_db()
    from fastapi.testclient import TestClient

    with TestClient(main.app) as c:
        yield c


PAYLOAD = {
    "primers": [
        {"name": "F", "sequence": "GGGGGAAAAA"},
        {"name": "R", "sequence": "GGGGGTTTTT"},
        {"name": "degenerate", "sequence": "ACGTNRYSWKMBDHV"},
    ],
    "continuous_threshold": 4,
    "three_prime_threshold": 3,
}

BODY = {
    "continuous_min": 2,
    "continuous_max": 6,
    "three_prime_min": 1,
    "three_prime_max": 5,
}


def _brute_force(metrics, continuous, three_prime):
    """逐配对朴素判定，作为前缀和聚合的对照。"""
    high = medium = low = 0
    for c, t in metrics:
        if t >= three_prime:
            high += 1
        elif c >= continuous:
            medium += 1
        else:
            low += 1
    return high, medium, low


# ---------- 纯算法单测：二维前缀和聚合 vs 逐配对朴素判定 ----------


def test_cells_match_brute_force_random_metrics():
    rng = random.Random(20260919)
    metrics = [(rng.randrange(0, 30), rng.randrange(0, 30)) for _ in range(300)]
    cells = threshold_map_cells(
        metrics,
        continuous_min=1,
        continuous_max=25,
        three_prime_min=1,
        three_prime_max=25,
    )
    assert len(cells) == 25 * 25
    for cell in cells:
        expected = _brute_force(
            metrics, cell.continuous_threshold, cell.three_prime_threshold
        )
        got = (cell.high_count, cell.medium_count, cell.low_count)
        assert got == expected
        assert sum(got) == len(metrics)


def test_cells_empty_metrics_and_metric_above_threshold_domain():
    # 无配对时全部单元为 0
    cells = threshold_map_cells(
        [], continuous_min=1, continuous_max=3, three_prime_min=1, three_prime_max=3
    )
    assert len(cells) == 9
    assert all(
        (c.high_count, c.medium_count, c.low_count) == (0, 0, 0) for c in cells
    )

    # 指标可超出阈值域（前缀和表按实际指标上界扩容），仍须归类正确
    metrics = [(200, 0), (0, 200), (1, 1)]
    cells = threshold_map_cells(
        metrics, continuous_min=1, continuous_max=2, three_prime_min=1, three_prime_max=2
    )
    by_key = {(c.three_prime_threshold, c.continuous_threshold): c for c in cells}
    cell = by_key[(1, 1)]
    assert (cell.high_count, cell.medium_count, cell.low_count) == (2, 1, 0)
    cell = by_key[(2, 2)]
    assert (cell.high_count, cell.medium_count, cell.low_count) == (1, 1, 1)


def test_cells_ordering():
    cells = threshold_map_cells(
        [(5, 5)], continuous_min=3, continuous_max=4, three_prime_min=1, three_prime_max=2
    )
    keys = [(c.three_prime_threshold, c.continuous_threshold) for c in cells]
    assert keys == [(1, 3), (1, 4), (2, 3), (2, 4)]


# ---------- API 端到端 ----------


@pytest.fixture(scope="module")
def batch_id(client):
    resp = client.post("/api/batches", json=PAYLOAD)
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def test_threshold_map_counts_match_pairs(client, batch_id):
    resp = client.post(f"/api/batches/{batch_id}/threshold-map", json=BODY)
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["batch_id"] == batch_id
    assert data["pair_count"] == 6
    assert data["cell_count"] == 5 * 5 == len(data["cells"])
    assert data["original_continuous_threshold"] == 4
    assert data["original_three_prime_threshold"] == 3

    # 与批次已保存配对的落库指标逐一对照（测试侧朴素逐对判定）
    pairs = client.get(f"/api/batches/{batch_id}").json()["pairs"]
    metrics = [
        (p["continuous_complementary_length"], p["three_prime_run_length"])
        for p in pairs
    ]
    for cell in data["cells"]:
        expected = _brute_force(
            metrics, cell["continuous_threshold"], cell["three_prime_threshold"]
        )
        got = (cell["high_count"], cell["medium_count"], cell["low_count"])
        assert got == expected
        assert sum(got) == data["pair_count"]


def test_threshold_map_ordering_and_original_marker(client, batch_id):
    data = client.post(f"/api/batches/{batch_id}/threshold-map", json=BODY).json()

    keys = [
        (c["three_prime_threshold"], c["continuous_threshold"]) for c in data["cells"]
    ]
    assert keys == sorted(keys)

    originals = [c for c in data["cells"] if c["is_original"]]
    assert len(originals) == 1
    assert originals[0]["continuous_threshold"] == 4
    assert originals[0]["three_prime_threshold"] == 3

    # 原阈值单元的三级计数恒等于批次落库的汇总计数
    batch = client.get(f"/api/batches/{batch_id}").json()
    assert originals[0]["high_count"] == batch["high_risk_count"]
    assert originals[0]["medium_count"] == batch["medium_risk_count"]
    assert originals[0]["low_count"] == batch["low_risk_count"]


def test_threshold_map_original_outside_range_not_marked(client, batch_id):
    body = {
        "continuous_min": 10,
        "continuous_max": 12,
        "three_prime_min": 10,
        "three_prime_max": 12,
    }
    data = client.post(f"/api/batches/{batch_id}/threshold-map", json=body).json()
    assert data["cell_count"] == 9
    assert all(not c["is_original"] for c in data["cells"])


def test_threshold_map_full_domain(client, batch_id):
    # 200×200 = 40000 个单元，恰为组合数上限
    body = {
        "continuous_min": 1,
        "continuous_max": 200,
        "three_prime_min": 1,
        "three_prime_max": 200,
    }
    resp = client.post(f"/api/batches/{batch_id}/threshold-map", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["cell_count"] == 40000 == len(data["cells"])
    assert sum(1 for c in data["cells"] if c["is_original"]) == 1


def test_threshold_map_does_not_modify_batch(client, batch_id):
    before = client.get(f"/api/batches/{batch_id}").json()
    client.post(f"/api/batches/{batch_id}/threshold-map", json=BODY)
    after = client.get(f"/api/batches/{batch_id}").json()
    assert before == after


def test_threshold_map_batch_not_found(client):
    resp = client.post("/api/batches/does-not-exist/threshold-map", json=BODY)
    assert resp.status_code == 404


def test_threshold_map_validation_errors(client, batch_id):
    url = f"/api/batches/{batch_id}/threshold-map"

    # 下限大于上限
    resp = client.post(url, json={**BODY, "continuous_min": 7})
    assert resp.status_code == 422
    resp = client.post(url, json={**BODY, "three_prime_min": 6})
    assert resp.status_code == 422

    # 越出 1～200
    for field, value in [
        ("continuous_min", 0),
        ("continuous_max", 201),
        ("three_prime_min", 0),
        ("three_prime_max", 201),
    ]:
        resp = client.post(url, json={**BODY, field: value})
        assert resp.status_code == 422, field

    # 非整数
    resp = client.post(url, json={**BODY, "continuous_min": 1.5})
    assert resp.status_code == 422

    # 缺字段
    resp = client.post(url, json={"continuous_min": 1})
    assert resp.status_code == 422
