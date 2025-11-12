#!/usr/bin/env python3
"""
데이터 디렉토리 구조 생성 스크립트
"""
import os
from pathlib import Path

def create_data_directories():
    """필요한 데이터 디렉토리 생성"""
    directories = [
        "data/hwpx",
        "data/pdf",
        "data/ground_truth",
        "chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ 디렉토리 생성: {directory}")
    
    # 예제 질문 파일 생성
    example_questions = {
        "questions": [
            {
                "question": "표에서 총합은 얼마인가요?",
                "answer": "정답 예시",
                "relevant_tables": ["table_1"]
            }
        ]
    }
    
    import json
    questions_path = Path("data/questions.json")
    if not questions_path.exists():
        with open(questions_path, 'w', encoding='utf-8') as f:
            json.dump(example_questions, f, ensure_ascii=False, indent=2)
        print(f"✓ 예제 파일 생성: {questions_path}")
    
    print("\n데이터 디렉토리 준비 완료!")
    print("\n다음 단계:")
    print("1. data/hwpx/ 폴더에 HWPX 파일들을 넣어주세요")
    print("2. data/pdf/ 폴더에 동일한 문서의 PDF 파일들을 넣어주세요")
    print("3. data/ground_truth/ 폴더에 정답 표 데이터를 넣어주세요")
    print("4. data/questions.json 파일을 수정하여 평가 질문을 추가하세요")

if __name__ == "__main__":
    create_data_directories()

