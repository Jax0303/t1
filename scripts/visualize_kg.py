#!/usr/bin/env python3
"""
추출된 표를 지식 그래프로 변환해 이미지로 시각화하는 스크립트

예시:
    python scripts/visualize_kg.py \
        --tables-json data/extracted_DMS_pdfplumber.json \
        --table-id "DMS_[기재정정]사업보고서 (2023.12)_20250321000022_page8_table0" \
        --output outputs/kg_page8_table0.png
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

# src 모듈을 로드하기 위해 프로젝트 루트를 경로에 추가
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 헤드리스 환경에서도 그릴 수 있도록 Agg 백엔드 사용
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from src.kg.table_to_kg import TableToKnowledgeGraph


def load_tables(json_path: Path) -> List[Dict]:
    """추출 JSON에서 표 리스트를 로드"""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "tables" in data:
        return data["tables"]
    if isinstance(data, list):
        return data
    raise ValueError(f"지원하지 않는 JSON 형식입니다: {json_path}")


def select_table(
    tables: List[Dict], table_id: Optional[str], index: int
) -> Dict:
    """table_id 또는 index로 표 선택"""
    if table_id:
        for table in tables:
            if table.get("table_id") == table_id:
                return table
        raise ValueError(f"table_id '{table_id}'를 찾을 수 없습니다.")

    if index < 0 or index >= len(tables):
        raise IndexError(
            f"index {index}가 범위를 벗어났습니다 (총 {len(tables)}개 표)."
        )
    return tables[index]


def filter_node_types(kg: nx.MultiDiGraph, allowed_types: Optional[List[str]]):
    """원하는 노드 타입만 남긴 서브그래프 반환"""
    if not allowed_types:
        return kg
    allowed_nodes = [
        node_id
        for node_id, attrs in kg.nodes(data=True)
        if attrs.get("type") in allowed_types
    ]
    return kg.subgraph(allowed_nodes).copy()


def visualize_kg(
    kg: nx.MultiDiGraph,
    output_path: Path,
    hide_labels: bool = False,
    layout: str = "spring",
):
    """NetworkX 그래프를 Matplotlib으로 그려 저장"""
    if kg.number_of_nodes() == 0:
        raise ValueError("그래프에 노드가 없습니다.")

    layout_funcs = {
        "spring": lambda g: nx.spring_layout(g, seed=42, k=0.5),
        "kamada_kawai": lambda g: nx.kamada_kawai_layout(g),
        "planar": lambda g: nx.planar_layout(g),
        "spectral": lambda g: nx.spectral_layout(g),
    }
    if layout not in layout_funcs:
        raise ValueError(f"지원하지 않는 레이아웃: {layout}")

    pos = layout_funcs[layout](kg)

    color_map = {
        "Table": "#1f77b4",
        "Row": "#2ca02c",
        "Column": "#ff7f0e",
        "Header": "#9467bd",
        "Cell": "#7f7f7f",
        "Value": "#d62728",
    }
    node_colors = [
        color_map.get(attrs.get("type"), "#17becf")
        for _, attrs in kg.nodes(data=True)
    ]

    labels = (
        None
        if hide_labels
        else {node: attrs.get("name", "") for node, attrs in kg.nodes(data=True)}
    )

    plt.figure(figsize=(16, 12))
    nx.draw_networkx(
        kg,
        pos=pos,
        with_labels=not hide_labels,
        labels=labels,
        node_color=node_colors,
        node_size=500,
        font_size=8,
        edge_color="#bbbbbb",
        arrows=True,
        arrowsize=12,
    )
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="표 데이터를 지식 그래프로 시각화"
    )
    parser.add_argument(
        "--tables-json",
        type=Path,
        required=True,
        help="scripts/test_single_pdf_extraction.py 결과 JSON 경로",
    )
    parser.add_argument(
        "--table-id",
        type=str,
        default=None,
        help="시각화할 table_id (없으면 index 사용)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="table_id를 지정하지 않을 때 사용할 표 인덱스",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="시각화 이미지를 저장할 경로 (PNG 추천)",
    )
    parser.add_argument(
        "--node-types",
        type=str,
        nargs="+",
        default=None,
        help="특정 노드 타입만 표시 (예: Header Value)",
    )
    parser.add_argument(
        "--hide-labels",
        action="store_true",
        help="노드 라벨을 숨겨 복잡한 그래프를 간단히 표시",
    )
    parser.add_argument(
        "--layout",
        type=str,
        default="spring",
        choices=["spring", "kamada_kawai", "planar", "spectral"],
        help="그래프 레이아웃 알고리즘",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tables = load_tables(args.tables_json)
    table = select_table(tables, args.table_id, args.index)

    converter = TableToKnowledgeGraph()
    kg = converter.convert_table_to_kg(table, table.get("table_id", "unknown"))
    filtered_kg = filter_node_types(kg, args.node_types)

    if filtered_kg.number_of_nodes() == 0:
        raise ValueError("선택한 노드 타입으로 구성된 노드가 없습니다.")

    visualize_kg(
        filtered_kg,
        args.output,
        hide_labels=args.hide_labels,
        layout=args.layout,
    )

    print(
        f"그래프 시각화를 생성했습니다: {args.output} "
        f"(노드 {filtered_kg.number_of_nodes()}개, 엣지 {filtered_kg.number_of_edges()}개)"
    )


if __name__ == "__main__":
    main()
