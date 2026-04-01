"""
결과 테이블 스캐폴드
====================
GSR의 핵심 주장 검증을 위한 비교표.

메트릭 정의 (PubTables-1M 공식)
--------------------------------
GrITS-Top  : 셀 content tree 구조 정확도
GrITS-Con  : 셀 content 일치도
GrITS-Loc  : 셀 bbox 위치 정확도
AP_span    : spanning cell class IoU@0.50 AP
             (없으면 논문에서 직접 읽어 채울 것)

출처 코드
----------
'xiao2023'  : Xiao et al., TSRDet, ICCV 2023 Table 2
'wang2023'  : Wang et al., TSRFormer DQ-DETR, ACMMM 2023
'khang2025' : Khang & Hong, TFLOP, CVPR 2025
'ours'      : 본 실험 직접 측정
'todo'      : 아직 미입력
"""

from __future__ import annotations
import json
from pathlib import Path


# ── 논문 인용 수치 (Xiao et al. Table 2 기준) ─────────────────────────
#    None = 해당 논문에 수치 없음 / 별도 측정 필요
BASELINE_NUMBERS: list[dict] = [
    # ── Detection-based (GSR 주장 직접 검증 그룹) ─────────────────────
    {
        "model":       "Cascade R-CNN",
        "paradigm":    "detection",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # TODO: Xiao et al. Table 2에서 채울 것
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "xiao2023",
        "notes":       "vanilla SmoothL1, 재현 없이 인용 가능",
    },
    {
        "model":       "TSRDet",
        "paradigm":    "detection",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # TODO
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "xiao2023",
        "notes":       "Xiao et al. 제안 모델, 인용 가능",
    },
    {
        "model":       "Deformable-DETR",
        "paradigm":    "detection",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # TODO
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "xiao2023",
        "notes":       "Xiao et al. Table 2에 수록",
    },
    {
        "model":       "TATR",
        "paradigm":    "detection",
        "backbone":    "ResNet-18",
        "grits_top":   None,   # TODO
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "xiao2023",
        "notes":       "Table Transformer, Xiao et al. Table 2에 수록",
    },
    # ── GSR (본 실험) ──────────────────────────────────────────────────
    {
        "model":       "Cascade R-CNN + GSR (GIoU)",
        "paradigm":    "detection",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # 실험 후 채울 것
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "ours",
        "notes":       "spanning cell → GIoULoss, 본 실험",
    },
    {
        "model":       "Deformable-DETR + GSR (weight×2)",
        "paradigm":    "detection",
        "backbone":    "ResNet-50",
        "grits_top":   None,
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "ours",
        "notes":       "spanning IoU loss weight 2×, 본 실험",
    },
    {
        "model":       "TATR + GSR (weight×2)",
        "paradigm":    "detection",
        "backbone":    "ResNet-18",
        "grits_top":   None,
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "ours",
        "notes":       "spanning IoU loss weight 2×, 본 실험",
    },
    # ── 상한선 (다른 패러다임) ─────────────────────────────────────────
    {
        "model":       "TSRFormer DQ-DETR",
        "paradigm":    "query-based",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # TODO: Wang et al. 2023
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "wang2023",
        "notes":       "상한선: query-based paradigm",
    },
    {
        "model":       "TFLOP",
        "paradigm":    "layout-pointer",
        "backbone":    "ResNet-50",
        "grits_top":   None,   # TODO: Khang & Hong 2025
        "grits_con":   None,
        "grits_loc":   None,
        "ap_span":     None,
        "source":      "khang2025",
        "notes":       "상한선: layout pointer paradigm",
    },
]


def print_table(numbers: list[dict] = BASELINE_NUMBERS) -> None:
    """결과를 ASCII 테이블로 출력한다."""
    col_w = [28, 16, 10, 10, 10, 10, 12]
    headers = ["Model", "Paradigm", "GrITS-Top", "GrITS-Con",
               "GrITS-Loc", "AP_span", "Source"]

    sep = "+" + "+".join("-" * w for w in col_w) + "+"
    fmt = "|" + "|".join(f"{{:<{w}}}" for w in col_w) + "|"

    print(sep)
    print(fmt.format(*headers))
    print(sep)

    prev_paradigm = None
    for row in numbers:
        if row["paradigm"] != prev_paradigm and prev_paradigm is not None:
            print(sep)
        prev_paradigm = row["paradigm"]

        def cell(v):
            return f"{v:.4f}" if isinstance(v, float) else ("—" if v is None else str(v))

        print(fmt.format(
            row["model"][:27],
            row["paradigm"][:15],
            cell(row["grits_top"]),
            cell(row["grits_con"]),
            cell(row["grits_loc"]),
            cell(row["ap_span"]),
            row["source"],
        ))
    print(sep)


def save_json(
    path: str = "results/baseline_numbers.json",
    numbers: list[dict] = BASELINE_NUMBERS,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=2)
    print(f"Saved → {path}")


def update_result(
    model_name: str,
    grits_top: float | None = None,
    grits_con: float | None = None,
    grits_loc: float | None = None,
    ap_span: float | None = None,
    json_path: str = "results/baseline_numbers.json",
) -> None:
    """실험 완료 후 특정 모델의 수치를 업데이트한다."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    matched = False
    for row in data:
        if row["model"] == model_name:
            if grits_top is not None: row["grits_top"] = grits_top
            if grits_con is not None: row["grits_con"] = grits_con
            if grits_loc is not None: row["grits_loc"] = grits_loc
            if ap_span   is not None: row["ap_span"]   = ap_span
            matched = True
            break

    if not matched:
        raise ValueError(f"Model '{model_name}' not found in {json_path}")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Updated '{model_name}' → {json_path}")


if __name__ == "__main__":
    save_json()
    print()
    print_table()
