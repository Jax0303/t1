#!/bin/bash
# 다음 단계 작업들을 모두 실행하는 스크립트

set -e

echo "=========================================="
echo "다음 단계 작업 실행 시작"
echo "=========================================="

BASE_DIR="outputs/dart_kg_experiment_all"
VIZ_DIR="visualizations"

# 디렉터리 확인
if [ ! -d "$BASE_DIR" ]; then
    echo "오류: $BASE_DIR 디렉터리를 찾을 수 없습니다."
    exit 1
fi

# 1. Graphviz 서브그래프 생성
echo ""
echo "1. Graphviz 서브그래프 생성 중..."
if command -v dot &> /dev/null; then
    python scripts/generate_graphviz_subgraphs.py --format png --output-dir "$VIZ_DIR/graphviz" || echo "⚠ Graphviz 생성 실패 (dot 실행 파일 확인 필요)"
else
    echo "⚠ Graphviz dot이 설치되지 않아 건너뜁니다."
    echo "  설치: sudo apt-get install graphviz (Ubuntu/Debian)"
fi

# 2. PyKEEN 임베딩 기반 유사 표 검색
echo ""
echo "2. PyKEEN 임베딩 생성 및 유사 표 검색 중..."
if python -c "import pykeen" 2>/dev/null; then
    python scripts/pykeen_similarity_search.py --base-dir "$BASE_DIR" --model TransE --dim 50 --epochs 10 || echo "⚠ PyKEEN 실행 실패"
else
    echo "⚠ PyKEEN이 설치되지 않아 건너뜁니다."
    echo "  설치: pip install pykeen"
fi

# 3. 빈 셀 추론 연구
echo ""
echo "3. 빈 셀 추론 연구 중..."
python scripts/cell_inference_research.py --base-dir "$BASE_DIR" || echo "⚠ 빈 셀 추론 실패"

# 4. 고급 KG 실험 (선택사항)
echo ""
echo "4. 고급 KG 실험 실행 중..."
if [ -f "scripts/advanced_kg_experiments.py" ]; then
    python scripts/advanced_kg_experiments.py || echo "⚠ 고급 실험 실패"
else
    echo "⚠ 고급 실험 스크립트를 찾을 수 없습니다."
fi

echo ""
echo "=========================================="
echo "모든 작업 완료!"
echo "=========================================="
echo ""
echo "생성된 파일:"
echo "- $VIZ_DIR/graphviz/: Graphviz 그래프"
echo "- $VIZ_DIR/pykeen/: PyKEEN 임베딩 및 리포트"
echo "- $VIZ_DIR/cell_inference/: 빈 셀 추론 결과"
echo "- $VIZ_DIR/neo4j/visualization_queries.cypher: Neo4j 시각화 쿼리"
echo ""
echo "Neo4j Browser에서 visualization_queries.cypher 파일을 열어 실행하세요."

