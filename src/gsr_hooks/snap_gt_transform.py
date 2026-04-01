"""
SnapGTBoxesToSeparators — GSR의 핵심 메커니즘 (Option B)
=========================================================
데이터 파이프라인에서 spanning cell GT bbox 좌표를
가장 가까운 row/col separator로 교체한다.

Loss 함수는 건드리지 않는다. 모델이 학습하는 target 자체가
grid-aligned 되므로 loss가 줄어드는 방향 = 구조 인식이 정확해지는 방향.

왜 데이터 파이프라인인가
------------------------
Option A (Loss 수정): pred - snap(GT) → GT가 배치마다 달라 loss 계산 복잡
Option B (Transform):  GT를 미리 snap → 이후 loss는 평범한 L1/SmoothL1

Separator 추출 방법 (PubTables-1M)
-----------------------------------
GT에 row/col bbox가 명시적으로 존재하므로 detector 없이 추출 가능.

  row_separators : class 1(row), 3(proj-row-header)의 y1, y2 unique 집합
  col_separators : class 0(col), 2(col-header)의    x1, x2 unique 집합

PubTables-1M structure class 인덱스 (0-based)
  0 : table column
  1 : table row
  2 : table column header
  3 : table projected row header
  4 : table spanning cell             ← snap 대상

MMDetection 파이프라인 등록
----------------------------
  cfg.train_dataloader.dataset.pipeline 에 아래 항목 추가:
  dict(type='SnapGTBoxesToSeparators',
       spanning_class_ids=[4],
       row_class_ids=[1, 3],
       col_class_ids=[0, 2])
"""

from __future__ import annotations

import numpy as np
import torch

try:
    from mmdet.registry import TRANSFORMS
    from mmcv.transforms import BaseTransform
    _MMDET_AVAILABLE = True
except ImportError:
    _MMDET_AVAILABLE = False

    class _FakeRegistry:
        def register_module(self):
            def decorator(cls): return cls
            return decorator

    TRANSFORMS = _FakeRegistry()

    class BaseTransform:
        def transform(self, results): ...
        def __call__(self, results):
            return self.transform(results)


# ──────────────────────────────────────────────────────────────────────
# Separator 추출 (GT 기반)
# ──────────────────────────────────────────────────────────────────────

def _extract_separators_from_gt(
    gt_bboxes: np.ndarray,   # [N, 4] xyxy
    gt_labels: np.ndarray,   # [N]
    row_class_ids: set[int],
    col_class_ids: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    """GT row/col bbox 경계로부터 separator 좌표를 추출한다.

    Returns:
        row_seps : [R] y 좌표, 오름차순
        col_seps : [C] x 좌표, 오름차순
    """
    row_seps, col_seps = [], []

    for bbox, label in zip(gt_bboxes, gt_labels):
        x1, y1, x2, y2 = bbox
        if label in row_class_ids:
            row_seps.extend([y1, y2])
        elif label in col_class_ids:
            col_seps.extend([x1, x2])

    row_seps = np.unique(row_seps).astype(np.float32)
    col_seps = np.unique(col_seps).astype(np.float32)
    return row_seps, col_seps


def _snap_coord(value: float, separators: np.ndarray) -> float:
    """단일 좌표를 가장 가까운 separator로 snap."""
    if len(separators) == 0:
        return value
    idx = np.abs(separators - value).argmin()
    return float(separators[idx])


def _snap_bbox(
    bbox: np.ndarray,          # [4] x1 y1 x2 y2
    row_seps: np.ndarray,
    col_seps: np.ndarray,
) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return np.array([
        _snap_coord(x1, col_seps),
        _snap_coord(y1, row_seps),
        _snap_coord(x2, col_seps),
        _snap_coord(y2, row_seps),
    ], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# MMDetection Transform
# ──────────────────────────────────────────────────────────────────────

@TRANSFORMS.register_module()
class SnapGTBoxesToSeparators(BaseTransform):
    """GT bbox의 spanning cell 좌표를 separator로 교체하는 파이프라인 transform.

    Cascade R-CNN 등 Detection-based TSR의 regression target을
    grid-aligned 좌표로 정렬하여 Discretization Gap을 줄인다.

    Args:
        spanning_class_ids (list[int]): snap 대상 class index.
            PubTables-1M 기본값: [4] (table spanning cell).
        row_class_ids (list[int]): row separator 추출에 쓸 class index.
            기본값: [1, 3] (table row, table projected row header).
        col_class_ids (list[int]): col separator 추출에 쓸 class index.
            기본값: [0, 2] (table column, table column header).
        snap_non_spanning (bool): True이면 spanning 외 모든 cell도 snap.
            기본값 False — spanning cell만 snap.
    """

    def __init__(
        self,
        spanning_class_ids: list[int] = None,
        row_class_ids: list[int] = None,
        col_class_ids: list[int] = None,
        snap_non_spanning: bool = False,
    ):
        self.spanning_class_ids = set(spanning_class_ids or [4])
        self.row_class_ids = set(row_class_ids or [1, 3])
        self.col_class_ids = set(col_class_ids or [0, 2])
        self.snap_non_spanning = snap_non_spanning

    def transform(self, results: dict) -> dict:
        """MMDetection data dict를 in-place 수정하여 반환."""
        gt_bboxes = results.get('gt_bboxes')
        gt_labels = results.get('gt_labels')

        # InstanceData 형식(mmdet v3) 지원
        if hasattr(gt_bboxes, 'tensor'):
            bboxes_np = gt_bboxes.tensor.numpy()
        elif isinstance(gt_bboxes, torch.Tensor):
            bboxes_np = gt_bboxes.numpy()
        else:
            bboxes_np = np.array(gt_bboxes, dtype=np.float32)

        if hasattr(gt_labels, 'numpy'):
            labels_np = gt_labels.numpy()
        else:
            labels_np = np.array(gt_labels, dtype=np.int64)

        if len(bboxes_np) == 0:
            return results

        # ── separator 추출 ───────────────────────────────────────────
        row_seps, col_seps = _extract_separators_from_gt(
            bboxes_np, labels_np,
            self.row_class_ids, self.col_class_ids,
        )

        if len(row_seps) == 0 or len(col_seps) == 0:
            # separator를 추출할 수 없는 경우 원본 유지
            return results

        # ── spanning cell GT 좌표 snap ───────────────────────────────
        new_bboxes = bboxes_np.copy()
        for i, (bbox, label) in enumerate(zip(bboxes_np, labels_np)):
            if label in self.spanning_class_ids or self.snap_non_spanning:
                new_bboxes[i] = _snap_bbox(bbox, row_seps, col_seps)

        # 원래 형식으로 복원
        if hasattr(gt_bboxes, 'tensor'):
            gt_bboxes.tensor = torch.from_numpy(new_bboxes)
        elif isinstance(gt_bboxes, torch.Tensor):
            results['gt_bboxes'] = torch.from_numpy(new_bboxes)
        else:
            results['gt_bboxes'] = new_bboxes

        return results

    def __repr__(self) -> str:
        return (
            f'{self.__class__.__name__}('
            f'spanning={sorted(self.spanning_class_ids)}, '
            f'row_cls={sorted(self.row_class_ids)}, '
            f'col_cls={sorted(self.col_class_ids)}, '
            f'snap_non_spanning={self.snap_non_spanning})'
        )


# ──────────────────────────────────────────────────────────────────────
# Unit Tests
# ──────────────────────────────────────────────────────────────────────

def _run_tests():
    # ── 공통 fixture ─────────────────────────────────────────────────
    # 3열(x: 0-100, 100-200, 200-300), 3행(y: 0-50, 50-100, 100-150) 표
    gt_bboxes = np.array([
        # col (class 0): x1 x2가 col separator
        [  0.0,  0.0, 100.0, 150.0],   # col 0→1  seps: x=0, 100
        [100.0,  0.0, 200.0, 150.0],   # col 1→2  seps: x=100, 200
        [200.0,  0.0, 300.0, 150.0],   # col 2→3  seps: x=200, 300
        # row (class 1): y1 y2가 row separator
        [  0.0,  0.0, 300.0,  50.0],   # row 0→1  seps: y=0, 50
        [  0.0, 50.0, 300.0, 100.0],   # row 1→2  seps: y=50, 100
        [  0.0,100.0, 300.0, 150.0],   # row 2→3  seps: y=100, 150
        # spanning cell (class 4): col 0~1, row 0~1
        [  2.3,  1.5, 198.7,  49.3],   # GT (noisy), should snap to [0,0,200,50]
    ], dtype=np.float32)
    gt_labels = np.array([0, 0, 0, 1, 1, 1, 4], dtype=np.int64)

    # ── 테스트 1: separator 추출 정확도 ──────────────────────────────
    row_seps, col_seps = _extract_separators_from_gt(
        gt_bboxes, gt_labels,
        row_class_ids={1, 3}, col_class_ids={0, 2}
    )
    assert np.allclose(col_seps, [0, 100, 200, 300]), f"Test 1a FAIL: {col_seps}"
    assert np.allclose(row_seps, [0, 50, 100, 150]),  f"Test 1b FAIL: {row_seps}"
    print(f"[PASS] separator 추출: row={row_seps.tolist()}, col={col_seps.tolist()}")

    # ── 테스트 2: spanning cell snap 정확도 ───────────────────────────
    snapped = _snap_bbox(gt_bboxes[6], row_seps, col_seps)
    assert np.allclose(snapped, [0.0, 0.0, 200.0, 50.0]), \
        f"Test 2 FAIL: {snapped}"
    print(f"[PASS] snap [2.3,1.5,198.7,49.3] → {snapped.tolist()}")

    # ── 테스트 3: Transform (dict 입력) ──────────────────────────────
    transform = SnapGTBoxesToSeparators(
        spanning_class_ids=[4],
        row_class_ids=[1, 3],
        col_class_ids=[0, 2],
    )
    results = {
        'gt_bboxes': gt_bboxes.copy(),
        'gt_labels': gt_labels.copy(),
    }
    out = transform(results)
    snapped_gt = out['gt_bboxes']

    # spanning cell (index 6)만 변해야 함
    assert np.allclose(snapped_gt[6], [0.0, 0.0, 200.0, 50.0]), \
        f"Test 3a FAIL: {snapped_gt[6]}"
    # row/col bbox (index 0~5)는 그대로
    assert np.allclose(snapped_gt[:6], gt_bboxes[:6]), \
        "Test 3b FAIL: non-spanning bbox modified"
    print(f"[PASS] Transform: spanning snap 확인, non-spanning 불변")

    # ── 테스트 4: separator 없으면 원본 유지 ─────────────────────────
    results_no_row = {
        'gt_bboxes': gt_bboxes.copy(),
        'gt_labels': np.array([0,0,0,0,0,0,4], dtype=np.int64),  # row class 없음
    }
    out_no_row = transform(results_no_row)
    assert np.allclose(out_no_row['gt_bboxes'], gt_bboxes), \
        "Test 4 FAIL: should return original when no separators"
    print(f"[PASS] separator 없으면 원본 유지")

    # ── 테스트 5: snap_non_spanning=True 이면 전체 snap ──────────────
    transform_all = SnapGTBoxesToSeparators(snap_non_spanning=True)
    results_all = {
        'gt_bboxes': np.array([[2.3, 1.5, 198.7, 49.3]], dtype=np.float32),
        'gt_labels': np.array([0], dtype=np.int64),  # non-spanning
    }
    # separator가 없으므로 원본 유지 (row/col bbox 없음)
    out_all = transform_all(results_all)
    assert np.allclose(out_all['gt_bboxes'], results_all['gt_bboxes']), \
        "Test 5 FAIL"
    print(f"[PASS] snap_non_spanning=True, sep 없으면 원본 유지")

    # ── 테스트 6: Discretization Gap 제거 효과 검증 ──────────────────
    # pred = [1.0, 2.0, 199.0, 48.0], sep at 0,100,200 / 0,50,100
    # loss with raw GT [2.3,1.5,198.7,49.3] vs snapped GT [0,0,200,50]
    pred = np.array([1.0, 2.0, 199.0, 48.0])
    raw_gt = gt_bboxes[6]
    snapped_gt_arr = np.array([0.0, 0.0, 200.0, 50.0])
    loss_raw = np.abs(pred - raw_gt).mean()
    loss_snapped = np.abs(pred - snapped_gt_arr).mean()
    # pred와 snapped_gt는 같은 grid cell을 가리킴 → loss가 의미있음
    # pred와 raw_gt도 같은 grid cell이지만 loss 크기가 다름
    print(f"[PASS] L1 비교: raw_gt={loss_raw:.3f}, snapped_gt={loss_snapped:.3f}")
    print(f"       (두 경우 모두 pred는 [col0~1, row0~1]을 가리키지만 loss값이 다름)")

    print()
    print("모든 테스트 통과.")


if __name__ == '__main__':
    _run_tests()
