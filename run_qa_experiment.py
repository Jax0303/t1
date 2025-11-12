#!/usr/bin/env python3
"""
질의응답 데이터셋 실험 실행 스크립트
"""
import yaml
from src.experiment import ExperimentRunner
from src.data import DatasetLoader

if __name__ == "__main__":
    # 설정 로드
    with open("config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 데이터셋 로더
    loader = DatasetLoader()
    
    # 질의응답 데이터셋 로드
    print("=" * 60)
    print("질의응답 데이터셋 로드")
    print("=" * 60)
    
    qa_dataset = loader.load_qa_dataset("dataset1/149.표 정보 질의 응답 데이터")
    
    # 질문-답변 쌍 추출
    training_qa = loader.extract_qa_pairs(qa_dataset['training']['label'])
    validation_qa = loader.extract_qa_pairs(qa_dataset['validation']['label'])
    
    print(f"\nTraining 질문-답변 쌍: {len(training_qa)}개")
    print(f"Validation 질문-답변 쌍: {len(validation_qa)}개")
    
    # 표 라벨 추출
    training_labels = loader.extract_table_labels(qa_dataset['training']['label'])
    validation_labels = loader.extract_table_labels(qa_dataset['validation']['label'])
    
    print(f"\nTraining 표 라벨: {len(training_labels)}개 문서")
    print(f"Validation 표 라벨: {len(validation_labels)}개 문서")
    
    # 실험 실행 (기존 ExperimentRunner 사용)
    print("\n" + "=" * 60)
    print("표 추출 및 RAG 실험 시작")
    print("=" * 60)
    
    runner = ExperimentRunner()
    
    # 질문 데이터를 파일로 저장 (실험에서 사용)
    import json
    questions_path = "data/questions_qa.json"
    with open(questions_path, 'w', encoding='utf-8') as f:
        json.dump(training_qa + validation_qa, f, ensure_ascii=False, indent=2)
    
    # 설정 업데이트
    config['dataset']['questions_path'] = questions_path
    
    # 실험 실행
    # runner.run_full_experiment()

