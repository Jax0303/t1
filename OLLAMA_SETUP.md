# Ollama 설치 가이드

## 설치 방법

### 방법 1: Snap 사용 (권장)
```bash
sudo snap install ollama
```

### 방법 2: 공식 설치 스크립트
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 방법 3: 수동 설치
```bash
# 바이너리 다운로드
curl -L https://ollama.com/download/ollama-linux-amd64 -o /tmp/ollama
chmod +x /tmp/ollama
sudo mv /tmp/ollama /usr/local/bin/
```

## 서비스 시작

```bash
# 백그라운드 실행
ollama serve &

# 또는 systemd 서비스로 등록 (선택)
```

## 모델 다운로드

```bash
# 기본 모델 (llama3.2)
ollama pull llama3.2

# 또는 더 작은 모델 (빠른 테스트용)
ollama pull llama3.2:1b

# 한국어 지원 모델
ollama pull qwen2.5
```

## 서비스 확인

```bash
# 서비스 상태 확인
curl http://localhost:11434/api/tags

# 모델 목록 확인
ollama list
```

## 실험에서 사용

설치 후 `run_hwp_comparison.py`를 다시 실행하면 자동으로 Ollama로 전환됩니다.

config.yaml에 이미 설정되어 있음:
```yaml
ollama:
  base_url: "http://localhost:11434"
  model: "llama3.2"
```

