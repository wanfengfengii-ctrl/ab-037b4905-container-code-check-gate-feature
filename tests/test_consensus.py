"""多读数共识求解（领域层）的独立测试。

重点覆盖需求点名的场景：
* 局部多数违反校验位时，动态规划求全局最优而非"逐位取多数再修补末位"；
* 超过一百个同分最优解时的精确计数、字典序前一百个解与截断标记；
* 两类无解边界（任一位置无合法观测、候选域中不存在满足校验位的组合）；
* 重复读数逐次计票、非法字符只计不一致且不进入候选域。

测试侧以暴力枚举候选域全部组合的独立参考实现复核全局最优、精确计数
与稳定顺序，避免用同一套实现自我作证。
"""

from __future__ import annotations

import dataclasses
import itertools
import random
import string

import pytest

from app.checksum import (
    CONSENSUS_MAX_SOLUTIONS,
    ConsensusResult,
    expected_check_digit,
    solve_consensus,
)


def _reference_consensus(readings: list[str]) -> tuple[int | None, int, list[str]]:
    """测试侧独立参考实现：暴力枚举候选域全部组合。

    返回 (最小代价, 最优解总数, 字典序全部最优解)；无解时代价为 None。
    与服务端"按（位置, 余数）的动态规划 + 剪枝枚举"的实现刻意不同。
    """
    domains: list[list[str]] = []
    for position in range(11):
        if position < 3:
            allowed = set(string.ascii_uppercase)
        elif position == 3:
            allowed = set("UJZ")
        else:
            allowed = set(string.digits)
        observed = {r[position] for r in readings if r[position] in allowed}
        domains.append(sorted(observed))
    if any(not domain for domain in domains):
        return None, 0, []

    best_cost: int | None = None
    solutions: list[str] = []
    for combo in itertools.product(*domains):
        number = "".join(combo)
        remainder = 0
        for index, char in enumerate(number[:10]):
            value = int(char) if char.isdigit() else _REFERENCE_LETTERS[char]
            remainder += value * 2**index
        remainder %= 11
        expected = 0 if remainder == 10 else remainder
        if expected != int(number[10]):
            continue
        cost = sum(
            char != reading[position]
            for reading in readings
            for position, char in enumerate(combo)
        )
        if best_cost is None or cost < best_cost:
            best_cost = cost
            solutions = [number]
        elif cost == best_cost:
            solutions.append(number)
    if best_cost is None:
        return None, 0, []
    solutions.sort()
    return best_cost, len(solutions), solutions


def _reference_letter_values() -> dict[str, int]:
    table: dict[str, int] = {}
    nxt = 10
    for letter in string.ascii_uppercase:
        table[letter] = nxt
        nxt += 1
        if nxt % 11 == 0:
            nxt += 1
    return table


_REFERENCE_LETTERS = _reference_letter_values()


def _hamming_total(number: str, readings: list[str]) -> int:
    """候选号对全部读数的逐位不一致总数。"""
    return sum(
        char != reading[position]
        for reading in readings
        for position, char in enumerate(number)
    )


# 多数派样例：三条 CSQU3054384（前 10 位期望校验位为 3，末位 4 不符）与
# 两条 CSQU3054784（合法箱号）。逐位多数派得到 CSQU3054384——非法；若再
# "修补末位"会得到 CSQU3054383，但 3 从未在末位观测中出现，不在候选域内。
MAJORITY_VIOLATING_READINGS = ["CSQU3054384"] * 3 + ["CSQU3054784"] * 2

# 超过一百个同分最优的样例：两条读数 11 位全部不同，每个位置两个候选各
# 一票，任何组合的代价恒为 11，全部满足校验位的组合同分最优（独立参考
# 确认共 294 个）。
TIED_READINGS = ["TEUZ0258588", "YJIU3376050"]


# ---------------------------------------------------------------- 唯一共识


def test_unique_consensus_is_determined() -> None:
    result = solve_consensus(["CSQU3054383", "CSQU3054383", "CSQU3054384"])
    assert result.status == "determined"
    assert result.minimum_cost == 1  # 仅第三条读数末位不一致
    assert result.solution_count == 1
    assert result.solutions == ("CSQU3054383",)
    assert result.truncated is False


def test_solution_is_global_optimum_not_majority_then_patch() -> None:
    # 局部多数违反校验位：逐位多数派 CSQU3054384 非法，"修补末位"得到的
    # CSQU3054383 不在候选域中（末位只观测到 4）。全局最优必须偏离多数派
    # 的第 9 位（3 -> 7），得到合法箱号 CSQU3054784，代价 3。
    majority = "CSQU3054384"
    assert expected_check_digit(majority[:10]) != int(majority[10])
    assert "3" not in {r[10] for r in MAJORITY_VIOLATING_READINGS}

    result = solve_consensus(MAJORITY_VIOLATING_READINGS)
    assert result.status == "determined"
    assert result.solutions == ("CSQU3054784",)
    assert result.minimum_cost == 3
    assert result.minimum_cost == _hamming_total(
        "CSQU3054784", MAJORITY_VIOLATING_READINGS
    )
    # 与独立暴力枚举参考一致。
    assert (result.minimum_cost, result.solution_count, list(result.solutions)) == (
        _reference_consensus(MAJORITY_VIOLATING_READINGS)
    )


# ---------------------------------------------------------------- 歧义多解


def test_ambiguous_optima_in_lexicographic_order() -> None:
    # 两个合法箱号各观测一次：两个同分最优解，按完整箱号字典序排列。
    result = solve_consensus(["CSQU3054784", "CSQU3054383"])  # 输入乱序
    assert result.status == "ambiguous"
    assert result.minimum_cost == 2
    assert result.solution_count == 2
    assert result.solutions == ("CSQU3054383", "CSQU3054784")
    assert list(result.solutions) == sorted(result.solutions)
    assert result.truncated is False


def test_duplicate_readings_vote_and_decide_winner() -> None:
    # 重复读数逐次计票：同一对候选，票数多者胜出。
    readings = ["CSQU3054383", "CSQU3054784", "CSQU3054784"]
    result = solve_consensus(readings)
    assert result.status == "determined"
    assert result.solutions == ("CSQU3054784",)
    assert result.minimum_cost == 2

    swapped = solve_consensus(["CSQU3054383", "CSQU3054383", "CSQU3054784"])
    assert swapped.solutions == ("CSQU3054383",)
    assert swapped.minimum_cost == 2


# ----------------------------------------------------- 超过一百个同分最优


def test_more_than_one_hundred_tied_optima_exact_count_and_truncation() -> None:
    result = solve_consensus(TIED_READINGS)
    reference_cost, reference_count, reference_solutions = _reference_consensus(
        TIED_READINGS
    )
    assert reference_count == 294  # 独立参考确认的精确最优解总数
    assert result.status == "ambiguous"
    assert result.minimum_cost == reference_cost == 11
    # 精确计数不受返回上限影响。
    assert result.solution_count == 294
    # 只返回字典序前一百个解，并置截断标记。
    assert len(result.solutions) == CONSENSUS_MAX_SOLUTIONS == 100
    assert result.truncated is True
    assert list(result.solutions) == reference_solutions[:100]
    assert list(result.solutions) == sorted(result.solutions)


def test_max_solutions_parameter_limits_returned_list_only() -> None:
    result = solve_consensus(TIED_READINGS, max_solutions=3)
    assert result.solution_count == 294  # 总数仍精确
    assert len(result.solutions) == 3
    assert result.truncated is True
    assert list(result.solutions) == _reference_consensus(TIED_READINGS)[2][:3]


def test_exactly_at_limit_is_not_truncated() -> None:
    # 最优解总数不超过返回上限时全部返回，截断标记为 False。
    result = solve_consensus(["CSQU3054383", "CSQU3054784"])
    assert result.solution_count == 2
    assert len(result.solutions) == 2
    assert result.truncated is False


# ---------------------------------------------------------------- 两类无解


def test_no_legal_observation_at_any_position_is_no_solution() -> None:
    # 位置 1 全部是小写（非法字符）：该位置无合法观测，候选域为空。
    result = solve_consensus(["csqu3054383", "csqu3054384"])
    assert result.status == "no_solution"
    assert result.minimum_cost is None
    assert result.solution_count == 0
    assert result.solutions == ()
    assert result.truncated is False


def test_no_combination_satisfies_check_digit_is_no_solution() -> None:
    # 各位置候选域均非空，但末位候选 {4, 5} 都不等于期望校验位 3。
    result = solve_consensus(["CSQU3054384", "CSQU3054385"])
    assert result.status == "no_solution"
    assert result.minimum_cost is None
    assert result.solution_count == 0
    assert result.solutions == ()
    assert result.truncated is False
    assert _reference_consensus(["CSQU3054384", "CSQU3054385"]) == (None, 0, [])


# -------------------------------------------------- 非法字符与重复读数口径


def test_illegal_characters_count_as_mismatches_but_never_candidates() -> None:
    # 第二条读数首位 '1' 非法：只在代价中计为不一致，不进入位置 1 候选域。
    readings = ["CSQU3054383", "1SQU3054383", "CSQU3054383"]
    result = solve_consensus(readings)
    assert result.status == "determined"
    assert result.solutions == ("CSQU3054383",)
    assert result.minimum_cost == 1  # 恰为非法字符造成的一次不一致


def test_unpaired_surrogate_reading_counts_as_mismatch_only() -> None:
    # 未配对代理字符不属于任何字符域：恰 11 位的读数合法参与计票，
    # 代理字符位置只计不一致、不产生候选。
    readings = ["\ud800SQU3054383", "CSQU3054383", "CSQU3054383"]
    result = solve_consensus(readings)
    assert result.status == "determined"
    assert result.solutions == ("CSQU3054383",)
    assert result.minimum_cost == 1


# ---------------------------------------------------------------- 输入守卫


def test_empty_readings_rejected() -> None:
    with pytest.raises(ValueError):
        solve_consensus([])


def test_wrong_length_reading_rejected() -> None:
    for bad in ("CSQU305438", "CSQU30543834", "", " CSQU3054383"):
        with pytest.raises(ValueError):
            solve_consensus(["CSQU3054383", bad])


def test_result_is_immutable() -> None:
    result = solve_consensus(["CSQU3054383", "CSQU3054383"])
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.minimum_cost = 0  # type: ignore[misc]


# ------------------------------------------------- 随机对照独立暴力枚举


def test_matches_independent_brute_force_on_random_readings() -> None:
    # 小字母表（含非法字符：小写 x、符号 !、空格）使候选域组合可控，
    # 暴力枚举参考可逐一复核全局最优、精确计数、字典序与截断标记。
    rng = random.Random(20260917)
    alphabet = "ABCUJZ01358x! "
    for _ in range(300):
        readings = [
            "".join(rng.choice(alphabet) for _ in range(11))
            for _ in range(rng.randint(2, 5))
        ]
        reference_cost, reference_count, reference_solutions = (
            _reference_consensus(readings)
        )
        result = solve_consensus(readings)
        assert result.minimum_cost == reference_cost, readings
        assert result.solution_count == reference_count, readings
        assert list(result.solutions) == reference_solutions[:100], readings
        assert result.truncated == (reference_count > CONSENSUS_MAX_SOLUTIONS)
        expected_status = (
            "no_solution"
            if reference_count == 0
            else ("determined" if reference_count == 1 else "ambiguous")
        )
        assert result.status == expected_status, readings
        if result.minimum_cost is not None:
            for solution in result.solutions:
                assert _hamming_total(solution, readings) == result.minimum_cost
