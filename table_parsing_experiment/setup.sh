#!/bin/bash
# 실험 환경 설치 스크립트

echo "========================================="
echo "Table Parsing Experiment Setup"
echo "========================================="

# 1. 가상 환경 생성
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 2. 가상 환경 활성화
echo "Activating virtual environment..."
source venv/bin/activate

# 3. pip 업그레이드
echo "Upgrading pip..."
pip install --upgrade pip

# 4. PyTorch 설치 (CPU 버전)
echo "Installing PyTorch (CPU version)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 5. 기타 의존성 설치
echo "Installing other dependencies..."
pip install transformers pillow opencv-python numpy pandas matplotlib seaborn
pip install scikit-learn scipy lxml beautifulsoup4 tqdm pyyaml tabulate
pip install datasets huggingface-hub

# 6. OCR 엔진 설치
echo "Installing OCR engines..."
pip install pytesseract easyocr

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "Note: Tesseract OCR도 시스템에 설치해야 합니다:"
echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-kor"
echo "  macOS: brew install tesseract tesseract-lang"
echo ""
echo "실행 방법:"
echo "  source venv/bin/activate"
echo "  python demo.py"
echo ""
