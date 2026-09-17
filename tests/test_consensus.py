"""多读数共识入口的领域与端到端测试。

测试侧以**独立暴力参考实现**（逐位置建候选域后笛卡尔积枚举，用独立
结构判定与校验位复算筛选，逐读数求汉明距离之和）核对服务端动态规划
给出的最小代价、精确解数与字典序前 100 解；并额外钉死需求点名的
场景：局部多数违反校验位时必须全局最优、超过 100 个同分最优时截断
且计数精确、两类无解边界、唯一共识。
"""

from __future__ import annotations

import itertools
import json
import random
import string
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.checksum import (
    CONSENSUS_MAX_READINGS,
    CONSENSUS_SOLUTION_LIMIT,
    ConsensusResult,
    consensus_container_numbers,
)
from app.main import app

client = TestClient(app)

CONSENSUS_ENDPOINT = "/api/v1/container-numbers/consensus"


# ----------------------------------------------------------- 独立参考实现


def _reference_letter_values() -> dict[str, int]:
    table: dict[str, int] = {}
    nxt = 10
    for letter in string.ascii_uppercase:
        table[letter] = nxt
        nxt += 1
        if nxt % 11 == 0:
            nxt += 1
    return table


LETTERS = _reference_letter_values()


def _reference_check_digit(first_ten: str) -> int:
    total = sum(
        (int(ch) if ch.isdigit() else LETTERS[ch]) * 2**i
        for i, ch in enumerate(first_ten)
    )
    remainder = total % 11
    return 0 if remainder == 10 else remainder


def _reference_domain(position: int) -> str:
    if position < 3:
        return string.ascii_uppercase
    if position == 3:
        return "UJZ"
    return string.digits


def _reference_valid(number: str) -> bool:
    if len(number) != 11:
        return False
    if not all(c in string.ascii_uppercase for c in number[:3]):
        return False
    if number[3] not in "UJZ":
        return False
    if not all(c in string.digits for c in number[4:]):
        return False
    return _reference_check_digit(number[:10]) == int(number[10])


def _reference_consensus(
    readings: list[str],
) -> tuple[int, list[str]] | None:
    """暴力参考：建候选域 -> 笛卡尔积 -> 结构与校验位筛选 -> 最小代价。

    返回 (最小代价, 全部最优解按完整箱号字典序)；任一位置无合法观测
    或候选域内无满足校验位的组合时返回 None。与服务端 DP 实现刻意
    不同（不做余数状态递推、不做剪枝）。
    """
    domains: list[list[str]] = []
    for position in range(11):
        allowed = set(_reference_domain(position))
        observed = {reading[position] for reading in readings}
        domain = sorted(observed & allowed)
        if not domain:
            return None
        domains.append(domain)

    minimum_cost: int | None = None
    optimal: list[str] = []
    for combo in itertools.product(*domains):
        number = "".join(combo)
        if not _reference_valid(number):
            continue
        cost = sum(
            char != reading[position]
            for reading in readings
            for position, char in enumerate(number)
        )
        if minimum_cost is None or cost < minimum_cost:
            minimum_cost = cost
            optimal = [number]
        elif cost == minimum_cost:
            optimal.append(number)
    if minimum_cost is None:
        return None
    return minimum_cost, sorted(optimal)


def _make_valid(prefix: str) -> str:
    return f"{prefix}{_reference_check_digit(prefix)}"


def _assert_matches_reference(
    body: dict[str, Any], readings: list[str]
) -> tuple[int, list[str]]:
    """断言响应与独立暴力参考逐字段一致，返回参考结论供场景继续断言。"""
    reference = _reference_consensus(readings)
    if reference is None:
        assert body["status"] == "not_found"
        assert body["minimum_cost"] is None
        assert body["solution_count"] == 0
        assert body["truncated"] is False
        assert body["solutions"] == []
        return -1, []

    minimum_cost, optimal = reference
    assert body["status"] == ("unique" if len(optimal) == 1 else "multiple")
    assert body["minimum_cost"] == minimum_cost
    assert body["solution_count"] == len(optimal)
    expected_list = optimal[:CONSENSUS_SOLUTION_LIMIT]
    assert body["truncated"] is (len(optimal) > CONSENSUS_SOLUTION_LIMIT)
    assert body["solutions"] == [
        {"container_number": number, "cost": minimum_cost}
        for number in expected_list
    ]
    return minimum_cost, optimal


def _noisy_reading(base: str, rng: random.Random) -> str:
    """在合法箱号上随机替换若干位；注入字符含合法与非法两类。"""
    chars = list(base)
    for position in rng.sample(range(11), rng.randrange(0, 4)):
        if rng.random() < 0.35:
            # 非法字符：小写字母 / 符号 / 类别码越域字符，永不进候选域。
            chars[position] = rng.choice("abx!?#z")
        else:
            chars[position] = rng.choice(_reference_domain(position))
    return "".join(chars)


# --------------------------------------------------------------- 唯一共识


def test_consensus_identical_readings_cost_zero_unique() -> None:
    readings = ["CSQU3054383", "CSQU3054383"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "status": "unique",
        "reading_count": 2,
        "minimum_cost": 0,
        "solution_count": 1,
        "truncated": False,
        "solutions": [
            {"container_number": "CSQU3054383", "cost": 0}
        ],
    }
    _assert_matches_reference(body, readings)


def test_consensus_single_noisy_reading_still_unique() -> None:
    readings = ["CSQU3054383"] * 3 + ["DSQU3054383"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    # 三条读数一致，另一条仅首字母偏差：真实号代价 1，任何改动只会
    # 同时得罪三条一致读数，故唯一最优即真实号。
    assert body["status"] == "unique"
    assert body["minimum_cost"] == 1
    assert body["solution_count"] == 1
    assert body["solutions"] == [
        {"container_number": "CSQU3054383", "cost": 1}
    ]
    assert body["reading_count"] == 4


def test_consensus_local_majority_violates_check_digit_global_optimum_wins() -> None:
    # 反“先逐位多数再修补末位”的钉死场景：
    # 逐位多数（含末位）合成 CSQU3054384 —— 结构合法但校验位不符
    # （CSQU305438 期望末位 3）。若先取多数再“修补”末位会得到
    # CSQU3054383；但 5 条 DSQU3054384 读数在首位是稳定多数，全局
    # 最优应保留首位 D、让 3+4 条 CSQ* 读数各承担首位分歧。
    readings = (
        ["CSQU3054383"] * 3
        + ["DSQU3054384"] * 5
        + ["CSQU3054384"] * 4
    )
    # 逐位多数合成号必须确实违反校验位，场景前提钉死。
    greedy_majority = "CSQU3054384"
    assert greedy_majority[10] == "4"
    assert _reference_check_digit(greedy_majority[:10]) == 3

    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "unique"
    assert body["minimum_cost"] == 10
    assert body["solution_count"] == 1
    assert body["solutions"] == [
        {"container_number": "DSQU3054384", "cost": 10}
    ]
    # 贪心“逐位多数 + 修补末位”得到 CSQU3054383：对 5 条 D 系读数各
    # 有 2 位分歧（首位与末位）、对 4 条 CSQU3054384 各有末位 1 位
    # 分歧，共 14，严格劣于全局最优 10。
    greedy_repaired_cost = sum(
        char != reading[position]
        for reading in readings
        for position, char in enumerate("CSQU3054383")
    )
    assert greedy_repaired_cost == 14
    assert body["minimum_cost"] < greedy_repaired_cost


# --------------------------------------------------------------- 歧义多解


def test_consensus_multiple_tied_optima_exact_list_in_lexicographic_order() -> None:
    # 两个合法读数在 5 个位置不同：差异位无论取哪一侧，代价恒为 5，
    # 恰好 4 个合法组合并列最优（原读数各一 + 两个合法杂交号）。
    readings = ["BXWZ6351759", "LWDZ6395759"]
    assert _reference_valid(readings[0]) and _reference_valid(readings[1])

    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "multiple"
    assert body["minimum_cost"] == 5
    assert body["solution_count"] == 4
    assert body["truncated"] is False
    assert [s["container_number"] for s in body["solutions"]] == [
        "BWDZ6395759",
        "BXWZ6351759",
        "LWDZ6395759",
        "LXWZ6351759",
    ]
    assert {s["cost"] for s in body["solutions"]} == {5}
    _assert_matches_reference(body, readings)


def test_consensus_over_one_hundred_optima_count_exact_list_truncated() -> None:
    # 两个合法读数 11 位全异：候选域共 2**11=2048 个组合，其中 180 个
    # 结构合法且校验位通过；所有差异位都计一次不一致，故合法组合代价
    # 恒为 11 —— 180 个同分最优，列表只给字典序前 100 且截断。
    readings = ["RNBZ1390991", "MBHU8246287"]
    assert _reference_valid(readings[0]) and _reference_valid(readings[1])

    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "multiple"
    assert body["minimum_cost"] == 11
    assert body["solution_count"] == 180  # 精确计数，不是列表长度
    assert body["truncated"] is True
    assert len(body["solutions"]) == CONSENSUS_SOLUTION_LIMIT == 100
    numbers = [s["container_number"] for s in body["solutions"]]
    assert numbers == sorted(numbers)  # 稳定字典序
    assert numbers[0] == "MBBU1290291"
    assert all(s["cost"] == 11 for s in body["solutions"])
    # 截断点恰好是全部最优解排序后的第 100/101 个边界。
    _, optimal = _assert_matches_reference(body, readings)
    assert numbers == optimal[:100]
    assert optimal[99] != optimal[100]


def test_consensus_repeated_readings_participate_in_tally() -> None:
    # 同一读数重复出现必须按重复次数计票：单条 A 对四条 B 时，B 系
    # 合法号以 1 条分歧胜出；去掉重复则是另一番结论。
    base = _make_valid("MSCU635589")  # MSCU6355895
    other = _make_valid("MSCU635580")  # 顺序号末位不同 -> 校验位不同
    readings = [base] * 4 + [other]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    body = response.json()
    assert body["status"] == "unique"
    assert body["solutions"][0]["container_number"] == base
    assert body["minimum_cost"] == 2  # 两条箱号在顺序号末位与校验位两处不同

    swapped = [other] * 4 + [base]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": swapped}
    )
    body = response.json()
    assert body["solutions"][0]["container_number"] == other
    assert body["minimum_cost"] == 2


# --------------------------------------------------------------- 两类无解


def test_consensus_no_legal_observation_at_position_is_not_found() -> None:
    # 边界一：首位只见小写字母（不合法且不进候选域）-> 候选域为空。
    readings = ["csqu3054383", "bsqu3054383"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["minimum_cost"] is None
    assert body["solution_count"] == 0
    assert body["truncated"] is False
    assert body["solutions"] == []
    assert body["reading_count"] == 2

    # 中间位置无合法观测：第 4 位两位读数都是小写 x。
    mid = ["CSQx3054383", "CSQx3054383"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": mid}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"
    assert response.json()["minimum_cost"] is None

    # 末位（校验位）只见非法字符：无合法数字可放末位。
    tail = ["CSQU305438X", "CSQU305438Y"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": tail}
    )
    assert response.status_code == 200
    assert response.json()["status"] == "not_found"


def test_consensus_no_check_digit_combination_is_not_found() -> None:
    # 边界二：各位置都有合法候选，但候选域内拼不出满足校验位的组合。
    # 前缀域唯一为 CSQU305438（期望末位 3），末位只见数字 1。
    readings = ["CSQU3054381", "CSQU3054381"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["minimum_cost"] is None
    assert body["solution_count"] == 0
    assert body["solutions"] == []
    _assert_matches_reference(body, readings)


def test_consensus_illegal_characters_count_as_mismatch_but_never_candidates() -> None:
    # 首位观测 C（合法）与 c（非法）各两次：候选域首位只有 C；两条
    # 小写读数在首位恒计不一致（其余 10 位与 C 系读数完全相同）。
    # 非法字符不得“凭票”进入候选。
    readings = ["CSQU3054383", "CSQU3054383", "cSQU3054383", "cSQU3054383"]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": readings}
    )
    body = response.json()
    assert body["status"] == "unique"
    assert body["solutions"][0]["container_number"] == "CSQU3054383"
    # 两条非法读数仅首位分歧 -> 总代价 2。
    assert body["minimum_cost"] == 2
    _assert_matches_reference(body, readings)

    # 全小写读数有 4 个非法字母位：每个位置的非法观测都计不一致，
    # 候选域仍只能取合法字符，代价为 4*2=8。
    all_lower = [
        "CSQU3054383",
        "CSQU3054383",
        "csqu3054383",
        "csqu3054383",
    ]
    response = client.post(
        CONSENSUS_ENDPOINT, json={"container_numbers": all_lower}
    )
    body = response.json()
    assert body["status"] == "unique"
    assert body["minimum_cost"] == 8
    assert body["solutions"][0]["container_number"] == "CSQU3054383"


# ----------------------------------------------------- 随机性质：对比暴力


@pytest.mark.parametrize("seed", range(60))
def test_consensus_matches_brute_force_on_random_readings(seed: int) -> None:
    rng = random.Random(seed)
    base_numbers = [
        _make_valid(
            "".join(rng.choice(string.ascii_uppercase) for _ in range(3))
            + rng.choice("UJZ")
            + "".join(rng.choice(string.digits) for _ in range(6))
        )
        for _ in range(3)
    ]
    reading_count = rng.randrange(2, 6)
    readings = [
        _noisy_reading(rng.choice(base_numbers), rng)
        for _ in range(reading_count)
    ]
    reference = _reference_consensus(readings)
    outcome = consensus_container_numbers(readings)
    if reference is None:
        assert outcome.status == "not_found"
        assert outcome.minimum_cost is None
        assert outcome.solution_count == 0
        assert outcome.solutions == ()
    else:
        minimum_cost, optimal = reference
        assert outcome.minimum_cost == minimum_cost
        assert outcome.solution_count == len(optimal)
        assert tuple(s.container_number for s in outcome.solutions) == tuple(
            optimal[:100]
        )
        assert all(s.cost == minimum_cost for s in outcome.solutions)


# --------------------------------------------------------------- 请求形状


def test_consensus_boundary_counts_two_and_one_hundred() -> None:
    two = client.post(
        CONSENSUS_ENDPOINT,
        json={"container_numbers": ["CSQU3054383"] * 2},
    )
    assert two.status_code == 200
    assert two.json()["reading_count"] == 2

    hundred = client.post(
        CONSENSUS_ENDPOINT,
        json={"container_numbers": ["CSQU3054383"] * CONSENSUS_MAX_READINGS},
    )
    assert hundred.status_code == 200
    assert hundred.json()["reading_count"] == 100
    assert hundred.json()["minimum_cost"] == 0


@pytest.mark.parametrize(
    "payload",
    [
        {"container_numbers": ["CSQU3054383"]},  # 仅 1 条，低于下限
        {"container_numbers": ["CSQU3054383"] * 101},  # 超过 100
        {"container_numbers": []},  # 空批
        {"container_numbers": ["CSQU3054383", "CSQU305438"]},  # 元素 10 位
        {"container_numbers": ["CSQU3054383", "CSQU30543834"]},  # 元素 12 位
        {"container_numbers": ["CSQU3054383", 12345678901]},  # 元素非字符串
        {"container_numbers": "CSQU3054383"},  # 顶层不是数组
        {},  # 缺字段
        {"container_numbers": ["CSQU3054383"] * 2, "normalize": True},  # 多余字段
    ],
)
def test_consensus_request_shape_errors_return_standard_detail(
    payload: dict[str, Any],
) -> None:
    response = client.post(CONSENSUS_ENDPOINT, json=payload)
    assert response.status_code == 422
    body = response.json()
    assert "detail" in body
    assert isinstance(body["detail"], list)
    assert body.get("status") not in ("unique", "multiple", "not_found")


def test_consensus_does_not_normalize_case_or_whitespace() -> None:
    # 小写不归一化：按非法观测处理（此处与另一读数组合仍可有解或无解，
    # 关键是绝不当作同一字符）；含空白的 12 位读数直接请求形状 422。
    spaced = client.post(
        CONSENSUS_ENDPOINT,
        json={"container_numbers": [" CSQU3054383", "CSQU3054383"]},
    )
    assert spaced.status_code == 422
    assert "detail" in spaced.json()


# ----------------------------------------------------- 未配对代理字符


SURROGATE_READING = "\ud800SQU3054383"


def _post_raw_json(url: str, payload: Any):
    return client.post(
        url,
        content=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def test_consensus_unpaired_surrogate_is_illegal_observation_not_crash() -> None:
    # 两条读数首位都是未配对代理字符：首位无合法观测 -> 无解，而非 500。
    response = _post_raw_json(
        CONSENSUS_ENDPOINT,
        {"container_numbers": [SURROGATE_READING, SURROGATE_READING]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_found"
    assert body["minimum_cost"] is None

    # 与一条正常读数混合时，代理字符只计不一致，候选域首位仅有 C。
    mixed = _post_raw_json(
        CONSENSUS_ENDPOINT,
        {"container_numbers": [SURROGATE_READING, "CSQU3054383"]},
    )
    assert mixed.status_code == 200
    mixed_body = mixed.json()
    assert mixed_body["status"] == "unique"
    assert mixed_body["solutions"][0]["container_number"] == "CSQU3054383"
    assert mixed_body["minimum_cost"] == 1

    # 12 位代理字符读数属于请求形状错误，detail 原样回显也不能 500。
    bad_shape = _post_raw_json(
        CONSENSUS_ENDPOINT,
        {"container_numbers": [SURROGATE_READING + "X"] * 2},
    )
    assert bad_shape.status_code == 422
    assert "detail" in bad_shape.json()


# ----------------------------------------------------- 领域层直接断言


def test_consensus_domain_rejects_non_eleven_character_readings() -> None:
    # 请求模型之外直接调用领域层：长度不是 11 属于编程错误，快速失败。
    with pytest.raises(ValueError):
        consensus_container_numbers(["CSQU3054383", "CSQU305438"])
    assert isinstance(
        consensus_container_numbers(["CSQU3054383"] * 2), ConsensusResult
    )


def test_consensus_does_not_disturb_existing_endpoints() -> None:
    # 引入共识入口后，原批量校验、纠错、明细、清单核对契约不变。
    verify = client.post(
        "/api/v1/container-numbers/verify",
        json={"container_numbers": ["CSQU3054383", "CSQU3054384"]},
    )
    assert verify.status_code == 200
    assert [r["passed"] for r in verify.json()["results"]] == [True, False]

    correct = client.post(
        "/api/v1/container-numbers/correct",
        json={"container_number": "CSQX3054383"},
    )
    assert correct.status_code == 200
    assert correct.json()["status"] == "unique"

    explain = client.post(
        "/api/v1/container-numbers/explain",
        json={"container_number": "CSQU3054383"},
    )
    assert explain.status_code == 200
    assert explain.json()["passed"] is True

    reconcile = client.post(
        "/api/v1/container-numbers/reconcile",
        json={
            "expected_container_numbers": ["CSQU3054383"],
            "onsite_container_numbers": ["CSQU3054383"],
        },
    )
    assert reconcile.status_code == 200
    assert reconcile.json()["matched_count"] == 1
