"""
GSR Snapping Loss — 보조 Regularizer (선택적 사용)
===================================================
주의: GSR의 핵심 메커니즘은 이 파일이 아니라
      snap_gt_transform.py (데이터 파이프라인 GT 교체)이다.

이 파일의 역할
--------------
추론 시 또는 추가 실험에서 예측 좌표를 separator 쪽으로
"밀어주는" 보조 regularization 항으로 사용 가능하다.
메인 loss (L1/SmoothL1)에 더하는 optional 항이며,
단독으로 GSR 효과를 대체하지 않는다.

GSR 학습 흐름 (Option B)
-------------------------
  1. SnapGTBoxesToSeparators (transform) → GT snap
  2. 표준 SmoothL1(pred, snapped_gt)    → 메인 loss
  3. [선택] HardSnappingLoss / SoftSnappingLoss → 보조 regularizer

HardSnappingLoss : pred를 현재 위치에서 nearest sep으로 push
SoftSnappingLoss : temperature τ로 soft-push (τ 크면 분산, 작으면 sharp)

입력 좌표 규약
--------------
pred_boxes      : Tensor[N, 4]  (x1, y1, x2, y2)
row_separators  : Tensor[R]     y좌표, 오름차순 정렬, stop-gradient
col_separators  : Tensor[C]     x좌표, 오름차순 정렬, stop-gradient
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ──────────────────────────────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────────────────────────────

def _split_coords(pred_boxes: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """[N,4] → x1, y1, x2, y2 각각 [N]"""
    return pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]


# ──────────────────────────────────────────────────────────────────────
# HardSnappingLoss
# ──────────────────────────────────────────────────────────────────────

class HardSnappingLoss(nn.Module):
    """argmin snap: 각 좌표를 가장 가까운 separator에 1:1 대응시킨다.

    gradient는 pred_boxes에만 흐른다 (separator는 stop-gradient).

    Args:
        loss_weight (float): 최종 loss 배율.
    """

    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @staticmethod
    def _snap(coords: Tensor, seps: Tensor) -> Tensor:
        """coords [N] → seps 중 가장 가까운 값 [N] (no gradient on seps).

        Args:
            coords : [N]  예측 좌표
            seps   : [S]  separator (stop-gradient 전제)
        Returns:
            snapped : [N]  nearest separator 값
        """
        # [N, S]
        dist = (coords.unsqueeze(-1) - seps.unsqueeze(0)).abs()
        idx = dist.argmin(dim=-1)          # [N]
        return seps[idx]                   # [N]

    def forward(
        self,
        pred_boxes: Tensor,
        row_seps: Tensor,
        col_seps: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_boxes : [N, 4]  (x1, y1, x2, y2)
            row_seps   : [R]     y 좌표
            col_seps   : [C]     x 좌표
        Returns:
            scalar loss
        """
        x1, y1, x2, y2 = _split_coords(pred_boxes)

        snapped = torch.stack([
            self._snap(x1, col_seps),
            self._snap(y1, row_seps),
            self._snap(x2, col_seps),
            self._snap(y2, row_seps),
        ], dim=-1)                         # [N, 4]

        loss = F.l1_loss(pred_boxes, snapped.detach())
        return loss * self.loss_weight


# ──────────────────────────────────────────────────────────────────────
# SoftSnappingLoss
# ──────────────────────────────────────────────────────────────────────

class SoftSnappingLoss(nn.Module):
    """Softmax-weighted expected snap.

    weight_i = softmax(-|coord - sep_i| / τ)
    expected_target = Σ sep_i * weight_i

    τ → 0  : HardSnappingLoss와 동일 (one-hot 수렴)
    τ → ∞  : 전체 separator에 대한 단순 평균

    Args:
        tau         (float): temperature. 기본 1.0.
        loss_weight (float): 최종 loss 배율.
    """

    def __init__(self, tau: float = 1.0, loss_weight: float = 1.0):
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau
        self.loss_weight = loss_weight

    def _expected_snap(self, coords: Tensor, seps: Tensor) -> Tensor:
        """coords [N] → expected separator 값 [N].

        Args:
            coords : [N]
            seps   : [S]  stop-gradient 전제
        Returns:
            expected : [N]
        """
        # [N, S]
        dist = (coords.unsqueeze(-1) - seps.unsqueeze(0)).abs()
        weights = F.softmax(-dist / self.tau, dim=-1)      # [N, S]
        return (weights * seps.unsqueeze(0)).sum(dim=-1)   # [N]

    def forward(
        self,
        pred_boxes: Tensor,
        row_seps: Tensor,
        col_seps: Tensor,
    ) -> Tensor:
        """
        Args:
            pred_boxes : [N, 4]  (x1, y1, x2, y2)
            row_seps   : [R]     y 좌표
            col_seps   : [C]     x 좌표
        Returns:
            scalar loss
        """
        x1, y1, x2, y2 = _split_coords(pred_boxes)

        expected = torch.stack([
            self._expected_snap(x1, col_seps),
            self._expected_snap(y1, row_seps),
            self._expected_snap(x2, col_seps),
            self._expected_snap(y2, row_seps),
        ], dim=-1)                          # [N, 4]

        loss = F.l1_loss(pred_boxes, expected.detach())
        return loss * self.loss_weight


# ──────────────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────────────

def _run_tests():
    import math

    print("=== HardSnappingLoss ===")

    # ── 테스트 1: pred가 separator와 정확히 일치 → loss = 0 ────────────
    row_seps = torch.tensor([10.0, 20.0, 30.0])
    col_seps = torch.tensor([5.0, 15.0, 25.0])
    pred = torch.tensor([[5.0, 10.0, 25.0, 30.0]])   # 정확히 separator 위
    loss = HardSnappingLoss()(pred, row_seps, col_seps)
    assert loss.item() < 1e-6, f"Test 1 FAIL: {loss.item()}"
    print(f"  [PASS] 정확히 일치 → loss={loss.item():.6f}")

    # ── 테스트 2: 두 separator 사이에 있을 때 가까운 쪽으로 snap ─────
    # x1=7 → col_sep 5 vs 15, 가까운 건 5, diff=2
    # y1=13 → row_sep 10 vs 20, 가까운 건 10, diff=3
    # x2=22 → col_sep 15 vs 25, 가까운 건 25, diff=3  (22-25=3, 22-15=7)
    # y2=27 → row_sep 20 vs 30, 가까운 건 30, diff=3  (27-30=3, 27-20=7)
    pred2 = torch.tensor([[7.0, 13.0, 22.0, 27.0]])
    expected_loss = (2.0 + 3.0 + 3.0 + 3.0) / 4.0
    loss2 = HardSnappingLoss()(pred2, row_seps, col_seps)
    assert abs(loss2.item() - expected_loss) < 1e-5, \
        f"Test 2 FAIL: got {loss2.item()}, expected {expected_loss}"
    print(f"  [PASS] 가장 가까운 sep으로 snap → loss={loss2.item():.4f} (expected {expected_loss:.4f})")

    # ── 테스트 3: loss_weight 적용 ──────────────────────────────────────
    loss3 = HardSnappingLoss(loss_weight=2.0)(pred2, row_seps, col_seps)
    assert abs(loss3.item() - loss2.item() * 2.0) < 1e-5, "Test 3 FAIL"
    print(f"  [PASS] loss_weight=2.0 → loss={loss3.item():.4f}")

    # ── 테스트 4: stop-gradient (row_seps에 grad 없음) ──────────────────
    pred4 = pred2.clone().requires_grad_(True)
    row_seps4 = row_seps.clone().requires_grad_(True)
    col_seps4 = col_seps.clone().requires_grad_(True)
    loss4 = HardSnappingLoss()(pred4, row_seps4, col_seps4)
    loss4.backward()
    assert pred4.grad is not None, "Test 4 FAIL: pred_boxes grad is None"
    # seps는 detach() 적용했으므로 grad가 흐르지 않아야 함
    assert row_seps4.grad is None, "Test 4 FAIL: row_seps should have no grad"
    assert col_seps4.grad is None, "Test 4 FAIL: col_seps should have no grad"
    print(f"  [PASS] pred_boxes에만 gradient 흐름 확인")

    print()
    print("=== SoftSnappingLoss ===")

    # ── 테스트 5: pred가 separator와 정확히 일치 → loss ≈ 0 (잔류 미세) ─
    # τ=1.0, sep 간격 10px → 인접 sep 기여 weight ≈ exp(-10) ≈ 4.5e-5
    # 이론적 잔류 오차 < 0.01
    loss5 = SoftSnappingLoss(tau=1.0)(pred, row_seps, col_seps)
    assert loss5.item() < 0.01, f"Test 5 FAIL: {loss5.item()}"
    print(f"  [PASS] sep 정확히 일치 → loss≈0 ({loss5.item():.6f}, τ=1.0 잔류 허용)")

    # ── 테스트 6: τ → 0이면 HardSnapping에 수렴 ────────────────────────
    # τ=0.001로 충분히 sharp하면 hard와 거의 동일
    hard_loss = HardSnappingLoss()(pred2, row_seps, col_seps).item()
    soft_sharp = SoftSnappingLoss(tau=0.001)(pred2, row_seps, col_seps).item()
    assert abs(soft_sharp - hard_loss) < 0.01, \
        f"Test 6 FAIL: soft_sharp={soft_sharp:.4f}, hard={hard_loss:.4f}"
    print(f"  [PASS] τ=0.001 → soft≈hard ({soft_sharp:.4f} ≈ {hard_loss:.4f})")

    # ── 테스트 7: τ 클수록 loss가 부드럽게 (단조 감소하지 않으나, hard보다 달라야 함) ─
    soft_large = SoftSnappingLoss(tau=100.0)(pred2, row_seps, col_seps).item()
    assert soft_large != hard_loss, "Test 7 FAIL: large tau should differ from hard"
    print(f"  [PASS] τ=100 → soft={soft_large:.4f} (hard={hard_loss:.4f}, 다름 확인)")

    # ── 테스트 8: tau <= 0 → ValueError ─────────────────────────────────
    try:
        SoftSnappingLoss(tau=0.0)
        assert False, "Test 8 FAIL: should raise ValueError"
    except ValueError:
        pass
    print(f"  [PASS] tau=0 → ValueError 발생")

    # ── 테스트 9: 배치 차원 확인 (N>1) ─────────────────────────────────
    pred_batch = torch.tensor([
        [5.0, 10.0, 25.0, 30.0],
        [7.0, 13.0, 22.0, 27.0],
        [6.0, 11.0, 24.0, 29.0],
    ])
    loss_hard = HardSnappingLoss()(pred_batch, row_seps, col_seps)
    loss_soft = SoftSnappingLoss()(pred_batch, row_seps, col_seps)
    assert loss_hard.shape == torch.Size([]), "Test 9a FAIL: not scalar"
    assert loss_soft.shape == torch.Size([]), "Test 9b FAIL: not scalar"
    print(f"  [PASS] 배치(N=3) → scalar loss 확인 (hard={loss_hard.item():.4f}, soft={loss_soft.item():.4f})")

    print()
    print("모든 테스트 통과.")


if __name__ == '__main__':
    _run_tests()
