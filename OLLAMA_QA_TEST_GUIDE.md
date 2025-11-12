# Ollama 설치 및 QA 테스트 가이드

## Ollama 설치

### 방법 1: 공식 설치 스크립트 (권장)

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**주의**: sudo 권한이 필요할 수 있습니다.

### 방법 2: 수동 설치

1. https://ollama.com/download 에서 운영체제에 맞는 파일 다운로드
2. 설치 파일 실행
3. 또는 바이너리를 직접 다운로드:
   ```bash
   curl -L https://ollama.com/download/ollama-linux-amd64 -o /tmp/ollama
   chmod +x /tmp/ollama
   sudo mv /tmp/ollama /usr/local/bin/ollama
   ```

## Ollama 서비스 시작

```bash
# 백그라운드에서 실행
ollama serve &

# 또는 포그라운드에서 실행 (로그 확인용)
ollama serve
```

## 모델 다운로드

```bash
# 기본 모델 (영어)
ollama pull llama3.2

# 또는 더 작은 모델 (빠른 테스트용)
ollama pull llama3.2:1b

# 한국어 지원 모델 (권장)
ollama pull qwen2.5
```

## 서비스 확인

```bash
# 서비스 상태 확인
curl http://localhost:11434/api/tags

# 모델 목록 확인
ollama list
```

## RAG QA 테스트 실행

Ollama가 설치되고 실행 중이면 다음 명령어로 테스트를 실행할 수 있습니다:

```bash
cd /home/user/t1
source venv/bin/activate
python run_rag_qa_with_ollama.py
```

## 테스트 스크립트 기능

`run_rag_qa_with_ollama.py` 스크립트는 다음을 수행합니다:

1. ✅ Ollama 서비스 실행 확인
2. ✅ 필요한 모델 다운로드 확인 (없으면 자동 다운로드 시도)
3. ✅ HWP 표 데이터 로드 및 변환
4. ✅ RAG 지식 베이스 구축
5. ✅ 예제 쿼리 실행 및 답변 생성
6. ✅ 결과를 JSON 파일로 저장

## 예제 쿼리

스크립트는 다음 쿼리들을 테스트합니다:

- "표에서 취업규칙과 관련된 내용을 찾아주세요"
- "표에서 근로조건에 대한 정보를 알려주세요"
- "표에서 필수 항목을 찾아주세요"
- "표에서 인사위원회 관련 정보를 찾아주세요"
- "표에서 휴직 관련 내용을 알려주세요"
- "표에서 제13조 관련 내용을 찾아주세요"
- "표에서 주40시간제 관련 내용을 찾아주세요"
- "표에서 근로기준법 관련 내용을 찾아주세요"

## 결과 파일

테스트 결과는 `rag_results_ollama.json` 파일에 저장됩니다.

## 문제 해결

### Ollama 서비스가 시작되지 않는 경우

```bash
# 프로세스 확인
ps aux | grep ollama

# 포트 확인
netstat -tuln | grep 11434

# 로그 확인
cat /tmp/ollama.log
```

### 모델 다운로드 실패

- 인터넷 연결 확인
- 디스크 공간 확인
- 수동으로 다운로드: `ollama pull llama3.2`

### RAG 시스템 초기화 실패

- 필요한 Python 패키지 설치 확인: `pip install langchain-ollama langchain-community`
- Ollama 서비스 실행 확인: `curl http://localhost:11434/api/tags`

## 빠른 시작 (전체 과정)

```bash
# 1. Ollama 설치 (sudo 권한 필요)
curl -fsSL https://ollama.com/install.sh | sh

# 2. Ollama 서비스 시작
ollama serve &

# 3. 모델 다운로드
ollama pull llama3.2

# 4. RAG QA 테스트 실행
cd /home/user/t1
source venv/bin/activate
python run_rag_qa_with_ollama.py
```

