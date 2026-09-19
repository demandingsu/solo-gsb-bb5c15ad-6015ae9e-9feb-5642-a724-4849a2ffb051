"""阈值扫描网格的三级风险计数（二维前缀和区间累加）。

对已保存批次的全部配对，仅取两个已落库指标：

* ``c`` = 最长连续互补长度（``max_complementary_run``）
* ``t`` = 3′端连续互补长度（``terminal_run``，即任一 3′端连续互补的最大值）

沿用既有判定规则，任意阈值组合 ``(C, T)`` 下每个配对的三级归类为：

* ``t >= T``          -> high（3′端连续互补达阈值）
* ``t < T 且 c >= C`` -> medium（仅最长连续互补达阈值）
* 其余                -> low

朴素做法对网格每个单元逐一扫描全部配对，复杂度 O(配对数 × 单元数)。
本模块先把配对撒到 ``(t, c)`` 二维直方图，再求二维前缀和
``ps[t][c] = 满足 t' <= t 且 c' <= c 的配对数``（与二维差分等价的
区间累加），此后每个单元的三级计数都是 O(1) 查表：

* ``low    = ps[T-1][C-1]``
* ``medium = ps[T-1][LIM] - ps[T-1][C-1]``
* ``high   = total - ps[T-1][LIM]``

整体复杂度 O(LIM² + 配对数 + 单元数)，其中 LIM 为指标上界
（不超过引物长度上限），不随网格单元数放大。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdCell:
    """单个阈值组合 ``(continuous, three_prime)`` 下的三级风险配对计数。"""

    continuous_threshold: int       # 连续互补阈值 C
    three_prime_threshold: int      # 3′端连续互补阈值 T
    high_count: int                 # t >= T 的配对数
    medium_count: int               # t < T 且 c >= C 的配对数
    low_count: int                  # t < T 且 c < C 的配对数


def threshold_map_cells(
    metrics: Sequence[tuple[int, int]],
    *,
    continuous_min: int,
    continuous_max: int,
    three_prime_min: int,
    three_prime_max: int,
) -> list[ThresholdCell]:
    """对阈值网格逐单元返回三级风险计数。

    ``metrics`` 为每个已保存配对的 ``(max_complementary_run, terminal_run)``；
    四个边界均为闭区间端点。返回单元按 3′端阈值升序、再按连续互补阈值升序。
    """
    # 指标上界：覆盖阈值域与全部实际指标，保证前缀和表可查任意单元
    lim = max(continuous_max, three_prime_max)
    for c, t in metrics:
        if c > lim:
            lim = c
        if t > lim:
            lim = t

    # (t, c) 二维直方图：hist[t][c] = 指标恰为 (c, t) 的配对数
    width = lim + 1
    hist = [[0] * width for _ in range(width)]
    for c, t in metrics:
        hist[t][c] += 1

    # 二维前缀和：ps[t][c] = 满足 t' <= t 且 c' <= c 的配对数
    ps = [[0] * width for _ in range(width)]
    for t in range(width):
        row_sum = 0
        hist_row = hist[t]
        ps_row = ps[t]
        ps_prev = ps[t - 1] if t > 0 else None
        for c in range(width):
            row_sum += hist_row[c]
            ps_row[c] = row_sum + (ps_prev[c] if ps_prev is not None else 0)

    total = len(metrics)
    full = width - 1  # 覆盖全部指标的下标

    def le(t: int, c: int) -> int:
        # 满足 t' <= t 且 c' <= c 的配对数；负下标按空集处理
        if t < 0 or c < 0:
            return 0
        return ps[min(t, full)][min(c, full)]

    cells: list[ThresholdCell] = []
    for three_prime in range(three_prime_min, three_prime_max + 1):
        below = three_prime - 1  # t < T 即 t <= T-1
        not_high = le(below, full)
        high = total - not_high
        for continuous in range(continuous_min, continuous_max + 1):
            low = le(below, continuous - 1)
            cells.append(
                ThresholdCell(
                    continuous_threshold=continuous,
                    three_prime_threshold=three_prime,
                    high_count=high,
                    medium_count=not_high - low,
                    low_count=low,
                )
            )
    return cells
