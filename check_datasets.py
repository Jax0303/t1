#!/usr/bin/env python3
"""
데이터셋 확인 및 검증 스크립트
"""
import json
from pathlib import Path
from src.data import DatasetLoader

def main():
    print("=" * 60)
    print("데이터셋 확인 및 검증")
    print("=" * 60)
    
    loader = DatasetLoader()
    
    # Dataset 1: 질의응답 데이터
    print("\n[Dataset 1: 질의응답 데이터]")
    try:
        qa_dataset = loader.load_qa_dataset("dataset1/149.표 정보 질의 응답 데이터")
        
        print(f"Training 원천데이터: {len(qa_dataset['training']['source'])}개 항목")
        print(f"Training 라벨링데이터: {len(qa_dataset['training']['label'])}개 항목")
        print(f"Validation 원천데이터: {len(qa_dataset['validation']['source'])}개 항목")
        print(f"Validation 라벨링데이터: {len(qa_dataset['validation']['label'])}개 항목")
        
        # 질문-답변 쌍 추출
        training_qa = loader.extract_qa_pairs(qa_dataset['training']['label'])
        validation_qa = loader.extract_qa_pairs(qa_dataset['validation']['label'])
        
        print(f"\nTraining 질문-답변 쌍: {len(training_qa)}개")
        print(f"Validation 질문-답변 쌍: {len(validation_qa)}개")
        
        if training_qa:
            print(f"\n첫 번째 질문 예시:")
            qa = training_qa[0]
            question = qa.get('question', 'N/A')
            answer = qa.get('answer', 'N/A')
            print(f"  질문: {str(question)[:100]}")
            print(f"  답변: {str(answer)[:100]}")
        
        # 표 라벨 추출
        training_labels = loader.extract_table_labels(qa_dataset['training']['label'])
        validation_labels = loader.extract_table_labels(qa_dataset['validation']['label'])
        
        print(f"\nTraining 표 라벨: {len(training_labels)}개 문서")
        print(f"Validation 표 라벨: {len(validation_labels)}개 문서")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    
    # Dataset 2: 다중 형식 데이터
    print("\n[Dataset 2: 다중 형식 데이터]")
    try:
        multi_format_data = loader.load_multi_format_dataset("dataset2")
        
        print(f"HWP 파일: {len(multi_format_data['hwp'])}개")
        print(f"HWPX 파일: {len(multi_format_data['hwpx'])}개")
        print(f"PDF 파일: {len(multi_format_data['pdf'])}개")
        
        for ext, files in multi_format_data.items():
            if files:
                print(f"\n{ext.upper()} 파일 목록:")
                for f in files:
                    print(f"  - {Path(f).name}")
        
        # 파일 매칭
        matched_groups = loader.match_files_by_content(multi_format_data)
        print(f"\n매칭된 파일 그룹: {len(matched_groups)}개")
        for group in matched_groups:
            print(f"\n  그룹: {group['base_name']}")
            for ext, file_path in group['files'].items():
                print(f"    {ext.upper()}: {Path(file_path).name}")
        
    except Exception as e:
        print(f"오류: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("데이터셋 확인 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()

