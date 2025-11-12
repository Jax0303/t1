#!/bin/bash
# Ollama 설치 스크립트

echo "=========================================="
echo "Ollama 설치 스크립트"
echo "=========================================="

# Ollama 설치 확인
if command -v ollama &> /dev/null; then
    echo "✓ Ollama가 이미 설치되어 있습니다."
    ollama --version
else
    echo "Ollama 설치 중..."
    
    # 공식 설치 스크립트 실행
    curl -fsSL https://ollama.com/install.sh | sh
    
    if [ $? -eq 0 ]; then
        echo "✓ Ollama 설치 완료"
    else
        echo "❌ 설치 실패. 수동 설치가 필요할 수 있습니다."
        echo ""
        echo "수동 설치 방법:"
        echo "1. https://ollama.com/download 에서 다운로드"
        echo "2. 또는 다음 명령어 실행:"
        echo "   curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
fi

# Ollama 서비스 시작
echo ""
echo "Ollama 서비스 시작 중..."
ollama serve > /tmp/ollama.log 2>&1 &
sleep 3

# 서비스 확인
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama 서비스가 실행 중입니다."
else
    echo "⚠ Ollama 서비스 시작 실패. 수동으로 시작하세요:"
    echo "   ollama serve"
fi

# 모델 다운로드
echo ""
echo "모델 다운로드 중 (llama3.2)..."
ollama pull llama3.2

if [ $? -eq 0 ]; then
    echo "✓ 모델 다운로드 완료"
else
    echo "⚠ 모델 다운로드 실패. 수동으로 다운로드하세요:"
    echo "   ollama pull llama3.2"
fi

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "다음 명령어로 테스트하세요:"
echo "  python run_rag_qa_with_ollama.py"

