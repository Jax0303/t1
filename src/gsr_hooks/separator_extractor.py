"""
SeparatorExtractor
==================
Row/col detector의 bbox 출력을 받아 GSR loss에 필요한
separator 좌표 Tensor를 생성한다.

설계 원칙
---------
- 모든 반환 Tensor는 stop-gradient (torch.no_grad + detach).
  separator는 학습 중 고정된 외부 신호로 취급.
- score 필터링 → 중심좌표 추출 → 정렬 순서로 처리.
- row detector  : bbox의 y 중심좌표 → row_separators
- col detector  : bbox의 x 중심좌표 → col_separators

입력 형식
---------
detections : List[List[float]]
    각 원소 = [x1, y1, x2, y2, score]
    detector의 raw output을 그대로 넘긴다.

반환
----
row_separators : Tensor[R]   y좌표, 오름차순 (stop-gradient)
col_separators : Tensor[C]   x좌표, 오름차순 (stop-gradient)
"""

from __future__ import annotations

from typing import List

import torch
from torch import Tensor


Detection = List[float]   # [x1, y1, x2, y2, score]


class SeparatorExtractor:
    """Row/col detector 출력 → separator Tensor 변환기.

    Args:
        score_threshold (float): 이 값 초과인 bbox만 separator로 사용.
            기본 0.5.
        device (str | torch.device): 반환 Tensor가 올라갈 디바이스.
    """

    def __init__(
        self,
        score_threshold: float = 0.5,
        device: str | torch.device = 'cpu',
    ):
        self.score_threshold = score_threshold
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    def extract_row_separators(self, detections: List[Detection]) -> Tensor:
        """row detector 출력 → y 중심좌표 Tensor.

        Args:
            detections: List of [x1, y1, x2, y2, score]
        Returns:
            Tensor[R] y좌표, 오름차순, stop-gradient.
            탐지 결과가 없으면 빈 Tensor 반환.
        """
        return self._extract(detections, axis='y')

    def extract_col_separators(self, detections: List[Detection]) -> Tensor:
        """col detector 출력 → x 중심좌표 Tensor.

        Args:
            detections: List of [x1, y1, x2, y2, score]
        Returns:
            Tensor[C] x좌표, 오름차순, stop-gradient.
        """
        return self._extract(detections, axis='x')

    def extract(
        self,
        row_detections: List[Detection],
        col_detections: List[Detection],
    ) -> tuple[Tensor, Tensor]:
        """row/col 동시 추출 convenience 메서드.

        Returns:
            (row_separators [R], col_separators [C])
        """
        return (
            self.extract_row_separators(row_detections),
            self.extract_col_separators(col_detections),
        )

    # ------------------------------------------------------------------
    def _extract(self, detections: List[Detection], axis: str) -> Tensor:
        """score 필터 → 중심좌표 추출 → 정렬 → stop-gradient Tensor."""
        filtered = [d for d in detections if d[4] > self.score_threshold]

        if not filtered:
            with torch.no_grad():
                return torch.tensor([], dtype=torch.float32,
                                    device=self.device)

        if axis == 'y':
            centers = [(d[1] + d[3]) / 2.0 for d in filtered]
        else:  # 'x'
            centers = [(d[0] + d[2]) / 2.0 for d in filtered]

        centers.sort()

        with torch.no_grad():
            return torch.tensor(
                centers, dtype=torch.float32, device=self.device
            ).detach()


# ──────────────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────────────

def _run_tests():
    extractor = SeparatorExtractor(score_threshold=0.5)

    # ── 테스트 1: 기본 row separator 추출 ─────────────────────────────
    row_dets = [
        [0.0, 8.0, 100.0, 12.0, 0.9],   # y_center = 10
        [0.0, 18.0, 100.0, 22.0, 0.8],  # y_center = 20
        [0.0, 28.0, 100.0, 32.0, 0.7],  # y_center = 30
    ]
    row_seps = extractor.extract_row_separators(row_dets)
    expected = torch.tensor([10.0, 20.0, 30.0])
    assert torch.allclose(row_seps, expected), f"Test 1 FAIL: {row_seps}"
    print(f"[PASS] row separator 추출: {row_seps.tolist()}")

    # ── 테스트 2: score threshold 필터링 ──────────────────────────────
    row_dets_mixed = [
        [0.0, 8.0, 100.0, 12.0, 0.9],   # y=10, 통과
        [0.0, 18.0, 100.0, 22.0, 0.3],  # y=20, 필터 (score ≤ 0.5)
        [0.0, 28.0, 100.0, 32.0, 0.7],  # y=30, 통과
    ]
    row_seps2 = extractor.extract_row_separators(row_dets_mixed)
    expected2 = torch.tensor([10.0, 30.0])
    assert torch.allclose(row_seps2, expected2), f"Test 2 FAIL: {row_seps2}"
    print(f"[PASS] score 필터 (threshold=0.5): {row_seps2.tolist()}")

    # ── 테스트 3: col separator 추출 ──────────────────────────────────
    col_dets = [
        [4.0, 0.0, 6.0, 100.0, 0.95],   # x_center = 5
        [14.0, 0.0, 16.0, 100.0, 0.85], # x_center = 15
        [24.0, 0.0, 26.0, 100.0, 0.75], # x_center = 25
    ]
    col_seps = extractor.extract_col_separators(col_dets)
    expected3 = torch.tensor([5.0, 15.0, 25.0])
    assert torch.allclose(col_seps, expected3), f"Test 3 FAIL: {col_seps}"
    print(f"[PASS] col separator 추출: {col_seps.tolist()}")

    # ── 테스트 4: 정렬 순서 보장 (입력이 뒤섞인 경우) ─────────────────
    row_dets_unsorted = [
        [0.0, 28.0, 100.0, 32.0, 0.9],  # y=30
        [0.0, 8.0, 100.0, 12.0, 0.8],   # y=10
        [0.0, 18.0, 100.0, 22.0, 0.7],  # y=20
    ]
    row_seps4 = extractor.extract_row_separators(row_dets_unsorted)
    assert (row_seps4 == row_seps4.sort().values).all(), "Test 4 FAIL: 정렬 안됨"
    assert torch.allclose(row_seps4, torch.tensor([10.0, 20.0, 30.0])), \
        f"Test 4 FAIL: {row_seps4}"
    print(f"[PASS] 입력 순서 무관 정렬: {row_seps4.tolist()}")

    # ── 테스트 5: stop-gradient 확인 ──────────────────────────────────
    assert not row_seps.requires_grad, "Test 5 FAIL: requires_grad=True"
    assert not col_seps.requires_grad, "Test 5 FAIL: requires_grad=True"
    print(f"[PASS] stop-gradient (requires_grad=False)")

    # ── 테스트 6: 탐지 결과 없음 → 빈 Tensor ─────────────────────────
    empty = extractor.extract_row_separators([])
    assert empty.shape == torch.Size([0]), f"Test 6 FAIL: {empty.shape}"
    print(f"[PASS] 빈 입력 → shape={empty.shape}")

    # ── 테스트 7: threshold를 통과하는 것이 없을 때 ──────────────────
    all_low = [[0.0, 8.0, 100.0, 12.0, 0.1]]
    empty2 = extractor.extract_row_separators(all_low)
    assert empty2.shape == torch.Size([0]), f"Test 7 FAIL: {empty2.shape}"
    print(f"[PASS] 전체 score 미달 → 빈 Tensor")

    # ── 테스트 8: extract() 동시 반환 ────────────────────────────────
    r, c = extractor.extract(row_dets, col_dets)
    assert torch.allclose(r, torch.tensor([10.0, 20.0, 30.0])), "Test 8 FAIL"
    assert torch.allclose(c, torch.tensor([5.0, 15.0, 25.0])), "Test 8 FAIL"
    print(f"[PASS] extract() 동시 반환: row={r.tolist()}, col={c.tolist()}")

    # ── 테스트 9: gsr_loss와 연동 확인 ───────────────────────────────
    from gsr_loss import HardSnappingLoss, SoftSnappingLoss

    pred_boxes = torch.tensor([
        [5.0, 10.0, 25.0, 30.0],
        [7.0, 13.0, 22.0, 27.0],
    ])
    hard_loss = HardSnappingLoss()(pred_boxes, r, c)
    soft_loss = SoftSnappingLoss(tau=1.0)(pred_boxes, r, c)
    assert hard_loss.shape == torch.Size([]), "Test 9 FAIL"
    assert soft_loss.shape == torch.Size([]), "Test 9 FAIL"
    print(f"[PASS] gsr_loss 연동: hard={hard_loss.item():.4f}, soft={soft_loss.item():.4f}")

    print()
    print("모든 테스트 통과.")


if __name__ == '__main__':
    _run_tests()
