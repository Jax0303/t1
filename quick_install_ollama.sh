#!/bin/bash
echo "Ollama 설치를 시작합니다..."
echo "주의: sudo 권한이 필요할 수 있습니다."
echo ""
echo "설치 스크립트 실행:"
curl -fsSL https://ollama.com/install.sh | sh

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ 설치 완료!"
    echo ""
    echo "다음 단계:"
    echo "1. Ollama 서비스 시작: ollama serve &"
    echo "2. 모델 다운로드: ollama pull llama3.2"
    echo "3. 테스트 실행: python run_rag_qa_with_ollama.py"
else
    echo ""
    echo "❌ 설치 실패. 수동 설치가 필요합니다."
    echo "자세한 내용은 OLLAMA_QA_TEST_GUIDE.md를 참고하세요."
fi
