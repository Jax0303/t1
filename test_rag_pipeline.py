"""
RAG 파이프라인 테스트 스크립트
032.표 이미지-텍스트 쌍 데이터와 149.표 정보 질의 응답 데이터를 사용하여
Gemini 2.5 Pro로 RAG 시스템 테스트
"""
import os
import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
import yaml
from tqdm import tqdm

from src.rag.rag_system import TableRAGSystem


def parse_html_table_to_dataframe(table_elem) -> Optional[pd.DataFrame]:
    """
    BeautifulSoup table 요소를 pandas DataFrame으로 변환
    
    Args:
        table_elem: BeautifulSoup table 요소
        
    Returns:
        DataFrame 또는 None
    """
    try:
        tr_elements = table_elem.find_all('tr')
        
        if not tr_elements:
            return None
        
        # 모든 행 파싱
        parsed_rows = []
        for tr in tr_elements:
            row_cells = []
            cell_elements = tr.find_all(['td', 'th'])
            
            for cell in cell_elements:
                # 셀 텍스트 추출 (공백 정리)
                cell_text = cell.get_text(separator=' ', strip=True)
                
                # colspan 처리
                colspan = int(cell.get('colspan', 1))
                for _ in range(colspan):
                    row_cells.append(cell_text)
            
            if row_cells:
                parsed_rows.append(row_cells)
        
        if not parsed_rows:
            return None
        
        # 행 길이 맞추기 (가장 긴 행 기준)
        max_cols = max(len(row) for row in parsed_rows) if parsed_rows else 0
        if max_cols == 0:
            return None
            
        for row in parsed_rows:
            while len(row) < max_cols:
                row.append("")
        
        # 첫 번째 행을 헤더로 사용
        headers = parsed_rows[0]
        data_rows = parsed_rows[1:] if len(parsed_rows) > 1 else []
        
        # 헤더가 비어있거나 모두 공백이면 숫자로 생성
        if not headers or all(not str(h).strip() for h in headers):
            headers = [f"열{i+1}" for i in range(max_cols)]
        
        # DataFrame 생성
        if data_rows:
            df = pd.DataFrame(data_rows, columns=headers[:max_cols])
        else:
            # 데이터 행이 없으면 헤더만 있는 빈 DataFrame
            df = pd.DataFrame(columns=headers[:max_cols])
        
        return df
        
    except Exception as e:
        # 오류 발생 시 상세 정보 출력하지 않고 None 반환
        return None


def load_html_tables(html_dir: str, limit: Optional[int] = None) -> List[Dict]:
    """
    HTML 파일에서 표를 추출하여 DataFrame 리스트로 반환
    
    Args:
        html_dir: HTML 파일이 있는 디렉토리
        limit: 최대 처리할 파일 수 (None이면 모두 처리)
        
    Returns:
        표 정보 리스트
    """
    html_files = list(Path(html_dir).glob("*.html"))
    if limit:
        html_files = html_files[:limit]
    
    tables = []
    
    print(f"\n[HTML 테이블 추출] 총 {len(html_files)}개 파일 처리 중...")
    
    for html_file in tqdm(html_files, desc="HTML 파싱"):
        try:
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            table_elements = soup.find_all('table')
            
            for idx, table_elem in enumerate(table_elements):
                try:
                    # HTML 테이블을 DataFrame으로 변환
                    df = parse_html_table_to_dataframe(table_elem)
                    
                    if df is not None and not df.empty:
                        tables.append({
                            'table_id': f"{html_file.stem}_table{idx}",
                            'dataframe': df,
                            'source_file': str(html_file),
                            'extraction_method': 'html_parser'
                        })
                except Exception as e:
                    print(f"  경고: {html_file}의 테이블 {idx} 파싱 실패: {e}")
                    continue
                    
        except Exception as e:
            print(f"  경고: {html_file} 읽기 실패: {e}")
            continue
    
    print(f"  추출 완료: {len(tables)}개 표")
    return tables


def load_qa_data(json_file: str) -> List[Dict]:
    """
    JSON 파일에서 질의응답 데이터 로드
    
    Args:
        json_file: JSON 파일 경로
        
    Returns:
        질의응답 데이터 리스트
    """
    print(f"\n[질의응답 데이터 로드] {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        qa_pairs = []
        
        if 'data' in data:
            for doc in data['data']:
                if 'paragraphs' in doc:
                    for para in doc['paragraphs']:
                        if 'qas' in para:
                            context = para.get('context', '')
                            table_title = para.get('table_title', '')
                            
                            for qa in para['qas']:
                                if not qa.get('is_impossible', False):
                                    qa_pairs.append({
                                        'question': qa['question'],
                                        'answer': qa['answer']['text'],
                                        'context': context,
                                        'table_title': table_title,
                                        'question_id': qa.get('question_id', ''),
                                        'doc_id': doc.get('doc_id', ''),
                                        'doc_title': doc.get('doc_title', '')
                                    })
        
        print(f"  로드 완료: {len(qa_pairs)}개 질의응답 쌍")
        return qa_pairs
        
    except Exception as e:
        print(f"  오류: JSON 파일 로드 실패: {e}")
        return []


def extract_tables_from_json_context(json_file: str) -> List[Dict]:
    """
    JSON 파일의 context에서 HTML 테이블을 추출하여 DataFrame 리스트로 반환
    
    Args:
        json_file: JSON 파일 경로
        
    Returns:
        표 정보 리스트
    """
    print(f"\n[JSON context에서 표 추출] {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = []
        
        if 'data' in data:
            for doc_idx, doc in enumerate(data['data']):
                if 'paragraphs' in doc:
                    for para_idx, para in enumerate(doc['paragraphs']):
                        context = para.get('context', '')
                        table_title = para.get('table_title', '')
                        
                        if '<table>' in context:
                            try:
                                # HTML 테이블 추출
                                soup = BeautifulSoup(context, 'html.parser')
                                table_elements = soup.find_all('table')
                                
                                for table_idx, table_elem in enumerate(table_elements):
                                    try:
                                        df = parse_html_table_to_dataframe(table_elem)
                                        
                                        if df is not None and not df.empty:
                                            tables.append({
                                                'table_id': f"doc{doc_idx}_para{para_idx}_table{table_idx}",
                                                'dataframe': df,
                                                'source_file': json_file,
                                                'table_title': table_title,
                                                'doc_id': doc.get('doc_id', ''),
                                                'doc_title': doc.get('doc_title', ''),
                                                'extraction_method': 'json_html_parser'
                                            })
                                    except Exception as e:
                                        print(f"  경고: 테이블 파싱 실패: {e}")
                                        continue
                                        
                            except Exception as e:
                                print(f"  경고: context 파싱 실패: {e}")
                                continue
        
        print(f"  추출 완료: {len(tables)}개 표")
        return tables
        
    except Exception as e:
        print(f"  오류: JSON 파일 로드 실패: {e}")
        return []


def test_rag_pipeline(
    html_dir: str,
    qa_json_file: str,
    api_key: str,
    model_name: str = "gemini-2.5-pro",
    num_test_questions: int = 10
):
    """
    RAG 파이프라인 테스트
    
    Args:
        html_dir: HTML 테이블 파일 디렉토리
        qa_json_file: 질의응답 JSON 파일 경로
        api_key: Gemini API 키
        model_name: 사용할 모델명 (gemini-2.5-pro)
        num_test_questions: 테스트할 질문 수
    """
    print("=" * 80)
    print("RAG 파이프라인 테스트 시작")
    print("=" * 80)
    print(f"모델: {model_name}")
    print(f"HTML 디렉토리: {html_dir}")
    print(f"QA JSON 파일: {qa_json_file}")
    print("=" * 80)
    
    # 1. HTML 테이블 추출 (샘플만)
    html_tables = load_html_tables(html_dir, limit=50)
    
    # 2. JSON context에서 표 추출
    json_tables = extract_tables_from_json_context(qa_json_file)
    
    # 3. 모든 표 합치기
    all_tables = html_tables + json_tables
    
    if not all_tables:
        print("\n오류: 추출된 표가 없습니다.")
        return
    
    print(f"\n총 {len(all_tables)}개 표로 지식 베이스 구축 중...")
    
    # 4. RAG 시스템 초기화 및 지식 베이스 구축
    rag_system = TableRAGSystem(
        api_key=api_key,
        model_name=model_name,
        provider="gemini"
    )
    
    try:
        rag_system.build_knowledge_base(all_tables)
        print("지식 베이스 구축 완료!")
    except Exception as e:
        print(f"\n오류: 지식 베이스 구축 실패: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 5. 질의응답 데이터 로드
    qa_pairs = load_qa_data(qa_json_file)
    
    if not qa_pairs:
        print("\n경고: 질의응답 데이터가 없습니다. 샘플 질문으로 테스트합니다.")
        # 샘플 질문들
        sample_questions = [
            "표에서 가장 큰 값은 무엇인가요?",
            "표의 행 수와 열 수는 얼마인가요?",
            "표의 헤더는 무엇인가요?"
        ]
        
        print("\n[샘플 질문 테스트]")
        for question in sample_questions:
            try:
                result = rag_system.query(question)
                print(f"\n질문: {question}")
                print(f"답변: {result['answer']}")
                print(f"관련 문서 수: {len(result['source_documents'])}")
            except Exception as e:
                print(f"  오류: {e}")
    else:
        # 실제 질의응답 데이터로 테스트
        test_questions = qa_pairs[:num_test_questions]
        
        print(f"\n[질의응답 테스트] {len(test_questions)}개 질문 테스트 중...")
        print("=" * 80)
        
        correct = 0
        total = len(test_questions)
        
        for idx, qa in enumerate(test_questions, 1):
            question = qa['question']
            expected_answer = qa['answer']
            
            print(f"\n[{idx}/{total}] 질문: {question}")
            print(f"예상 답변: {expected_answer}")
            
            try:
                result = rag_system.query(question)
                predicted_answer = result['answer']
                
                print(f"생성 답변: {predicted_answer}")
                print(f"관련 문서 수: {len(result['source_documents'])}")
                
                # 간단한 정확도 체크 (답변에 예상 답변이 포함되어 있는지)
                if expected_answer.lower() in predicted_answer.lower() or \
                   predicted_answer.lower() in expected_answer.lower():
                    correct += 1
                    print("✓ 정답 일치")
                else:
                    print("✗ 정답 불일치")
                    
            except Exception as e:
                print(f"  오류: {e}")
                import traceback
                traceback.print_exc()
        
        accuracy = correct / total if total > 0 else 0
        print("\n" + "=" * 80)
        print(f"테스트 결과: {correct}/{total} 정답 ({accuracy*100:.1f}%)")
        print("=" * 80)
    
    # 6. 정리
    print("\n지식 베이스 정리 중...")
    rag_system.clear_knowledge_base()
    print("완료!")


def main():
    """메인 함수"""
    # 설정 파일 로드
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print(f"경고: {config_path} 파일이 없습니다. config.yaml.example을 참고하여 생성하세요.")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("오류: GEMINI_API_KEY 환경변수를 설정하거나 config.yaml 파일을 생성하세요.")
            return
        model_name = "gemini-2.5-pro"
    else:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        api_key = config['api']['gemini']['api_key']
        model_name = config['api']['gemini'].get('model', 'gemini-2.5-pro')
    
    # 데이터 경로 설정
    html_dir = "/home/user/t1/test_data/032"
    qa_json_file = "/home/user/t1/test_data/149/1.Training_[TQA] эСЬ ьаХы│┤ ьзИьЭШьЭСыЛ╡_output_ыЭ╝ы▓иызБыН░ьЭ┤эД░_2023052317.json"
    
    # 경로 확인
    if not os.path.exists(html_dir):
        print(f"오류: HTML 디렉토리를 찾을 수 없습니다: {html_dir}")
        return
    
    if not os.path.exists(qa_json_file):
        # 다른 JSON 파일 찾기
        json_files = list(Path("/home/user/t1/test_data/149").glob("*.json"))
        if json_files:
            qa_json_file = str(json_files[0])
            print(f"경로 변경: {qa_json_file}")
        else:
            print(f"오류: QA JSON 파일을 찾을 수 없습니다.")
            return
    
    # RAG 파이프라인 테스트 실행
    test_rag_pipeline(
        html_dir=html_dir,
        qa_json_file=qa_json_file,
        api_key=api_key,
        model_name=model_name,
        num_test_questions=10
    )


if __name__ == "__main__":
    main()

