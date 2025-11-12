# 빠른 시작 가이드

## 1. 환경 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# NLTK 데이터 다운로드
python -c "import nltk; nltk.download('punkt')"

# 데이터 디렉토리 생성
python setup_data_dirs.py
```

## 2. 데이터 준비

### HWPX 파일
- `data/hwpx/` 폴더에 `.hwpx` 파일들을 넣어주세요

### PDF 파일  
- `data/pdf/` 폴더에 동일한 문서의 `.pdf` 파일들을 넣어주세요
- HWPX와 PDF는 동일한 문서의 다른 형식이어야 합니다

### Ground Truth (정답 데이터)
- `data/ground_truth/` 폴더에 JSON 형식의 정답 표 데이터를 넣어주세요
- 파일명은 원본 문서명과 매칭되어야 합니다
- 형식은 `example.json` 참고

### 평가 질문
- `data/questions.json` 파일을 수정하여 질문-답변 쌍을 추가하세요
- 각 질문에는 `question`, `answer`, `relevant_tables` 필드가 필요합니다

## 3. 설정 확인

`config.yaml` 파일에서 다음을 확인하세요:
- API 키가 올바르게 설정되어 있는지
- 모델명이 원하는 것으로 설정되어 있는지 (기본: gemini-2.0-flash-exp)
- 데이터셋 경로가 올바른지

## 4. 실험 실행

```bash
python run_experiment.py
```

## 5. 결과 확인

실험 결과는 `results.json` 파일에 저장됩니다:
- 표 추출 성능 (F1-score, 처리 속도)
- RAG 성능 (EM Score, Hit@K, BLEU Score)
- 베이스라인 대비 개선율
- 목표 달성 여부

## 문제 해결

### HWPX 파일을 찾을 수 없음
- `data/hwpx/` 폴더에 `.hwpx` 파일이 있는지 확인하세요
- 파일명에 공백이나 특수문자가 없는지 확인하세요

### PDF 추출 오류
- `camelot-py`는 `ghostscript`와 `tesseract`가 필요할 수 있습니다
- 시스템에 설치되어 있는지 확인하세요:
  ```bash
  # Ubuntu/Debian
  sudo apt-get install ghostscript tesseract-ocr
  
  # macOS
  brew install ghostscript tesseract
  ```

### API 오류
- Gemini API 키가 유효한지 확인하세요
- API 할당량을 확인하세요
- 모델명이 올바른지 확인하세요 (gemini-2.0-flash-exp 또는 gemini-2.5-pro)

### 메모리 부족
- 대용량 문서의 경우 배치 처리를 고려하세요
- `config.yaml`에서 청크 크기를 조정하세요

