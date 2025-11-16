"""
지식 그래프 기반 RAG 시스템
표에서 추출한 지식 그래프를 활용하여 더 정확한 질의응답 제공
"""
import os
from typing import List, Dict, Optional, Set
import pandas as pd
import networkx as nx
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .table_to_kg import TableToKnowledgeGraph


class KnowledgeGraphRAGSystem:
    """지식 그래프 기반 RAG 시스템"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", provider: str = "openai",
                 ollama_base_url: str = "http://localhost:11434", use_korean_embedding: bool = True):
        """
        Args:
            api_key: API 키
            model_name: 사용할 모델명
            provider: "openai", "gemini" 또는 "ollama"
            ollama_base_url: Ollama API 기본 URL
            use_korean_embedding: 한국어 최적화 임베딩 사용 여부
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider.lower()
        self.ollama_base_url = ollama_base_url
        self.use_korean_embedding = use_korean_embedding
        
        # 임베딩 모델 초기화
        self.embeddings = self._init_embeddings()
        
        # LLM 초기화
        self.llm = self._init_llm()
        
        # 지식 그래프 저장소
        self.kg_store: Dict[str, nx.MultiDiGraph] = {}
        self.kg_converter = TableToKnowledgeGraph()
        
        # 벡터 스토어 (텍스트 기반 검색용)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def _init_embeddings(self):
        """임베딩 모델 초기화"""
        if self.use_korean_embedding:
            try:
                from sentence_transformers import SentenceTransformer
                # 한국어 최적화 모델 사용
                model = SentenceTransformer('jhgan/ko-sroberta-multitask')
                print("한국어 최적화 임베딩 모델 로드 완료: ko-sroberta-multitask")
                
                # LangChain 래퍼 생성
                class KoreanEmbeddings:
                    def __init__(self, model):
                        self.model = model
                    
                    def embed_documents(self, texts):
                        return self.model.encode(texts, convert_to_numpy=True).tolist()
                    
                    def embed_query(self, text):
                        return self.model.encode([text], convert_to_numpy=True)[0].tolist()
                
                return KoreanEmbeddings(model)
            except Exception as e:
                print(f"한국어 임베딩 모델 로드 실패 ({e}). 기본 임베딩 사용.")
        
        # 기본 임베딩 (OpenAI, Gemini 등)
        try:
            if self.provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                return OpenAIEmbeddings(
                    openai_api_key=self.api_key,
                    model="text-embedding-3-small"
                )
            elif self.provider == "gemini":
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                return GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=self.api_key
                )
            elif self.provider == "ollama":
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                    return OllamaEmbeddings(
                        model=self.model_name,
                        base_url=self.ollama_base_url
                    )
                except ImportError:
                    from langchain_core.embeddings import FakeEmbeddings
                    return FakeEmbeddings(size=384)
            else:
                from langchain_core.embeddings import FakeEmbeddings
                return FakeEmbeddings(size=384)
        except Exception as e:
            print(f"임베딩 초기화 실패: {e}. FakeEmbeddings 사용.")
            from langchain_core.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384)
    
    def _init_llm(self):
        """LLM 초기화"""
        try:
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=self.model_name,
                    openai_api_key=self.api_key,
                    temperature=0.0,
                    max_retries=3,
                    timeout=60
                )
            elif self.provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=self.model_name,
                    google_api_key=self.api_key,
                    temperature=0.0,
                    max_retries=3,
                    timeout=60
                )
            elif self.provider == "ollama":
                try:
                    from langchain_community.llms import Ollama
                    return Ollama(
                        model=self.model_name,
                        base_url=self.ollama_base_url,
                        temperature=0.0
                    )
                except ImportError:
                    from langchain_ollama import ChatOllama
                    return ChatOllama(
                        model=self.model_name,
                        base_url=self.ollama_base_url,
                        temperature=0.0
                    )
            else:
                raise ValueError(f"지원하지 않는 provider: {self.provider}")
        except Exception as e:
            raise ValueError(f"LLM 초기화 실패 ({self.provider}): {str(e)}")
    
    def build_knowledge_base(self, tables: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        표 데이터로부터 지식 베이스 구축 (지식 그래프 + 벡터 스토어)
        
        Args:
            tables: 추출된 표 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        documents = []
        
        print("지식 그래프 구축 중...")
        for table_info in tables:
            table_id = table_info.get('table_id', 'unknown')
            source_file = table_info.get('source_file', 'unknown')
            
            # 표 데이터를 지식 그래프로 변환
            if 'rows' in table_info:
                # hwp5-table-extractor 형식
                kg = self.kg_converter.convert_table_to_kg(table_info, table_id)
            elif 'dataframe' in table_info:
                # DataFrame 형식 - rows로 변환
                df = table_info['dataframe']
                table_data = self._dataframe_to_table_data(df, table_id)
                kg = self.kg_converter.convert_table_to_kg(table_data, table_id)
            else:
                continue
            
            self.kg_store[table_id] = kg
            
            # 지식 그래프를 텍스트로 변환하여 벡터 스토어에 추가
            kg_text = self._kg_to_text(kg, table_id)
            
            # 청킹
            chunks = self._chunk_text(kg_text, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'table_id': table_id,
                        'source_file': source_file,
                        'chunk_index': chunk_idx,
                        'extraction_method': table_info.get('extraction_method', 'unknown'),
                        'has_kg': True
                    }
                )
                documents.append(doc)
        
        print(f"  {len(self.kg_store)}개 표의 지식 그래프 구축 완료")
        
        # 벡터 스토어 생성
        persist_directory = "./chroma_db_kg"
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        except Exception as e:
            if "readonly" in str(e).lower() or "permission" in str(e).lower():
                print(f"  경고: 디스크 저장 실패. 메모리 모드로 전환합니다.")
                self.vector_store = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings
                )
            else:
                raise
        
        # Retriever 생성
        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        
        # QA 체인 생성
        prompt_template = """다음 컨텍스트는 문서에서 추출한 표 정보와 지식 그래프 구조입니다.
표의 구조, 헤더 계층, 셀 간의 관계를 바탕으로 질문에 정확하게 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        self.qa_chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        self._retriever_for_sources = self.retriever
    
    def _dataframe_to_table_data(self, df: pd.DataFrame, table_id: str) -> Dict:
        """DataFrame을 표 데이터 형식으로 변환"""
        from src.utils.table_converter import dataframe_to_table_data
        return dataframe_to_table_data(df, table_id)
    
    def _kg_to_text(self, kg: nx.MultiDiGraph, table_id: str) -> str:
        """지식 그래프를 텍스트로 변환"""
        text_parts = [f"표 ID: {table_id}"]
        text_parts.append(f"지식 그래프 노드 수: {kg.number_of_nodes()}, 엣지 수: {kg.number_of_edges()}")
        
        # 엔티티 타입별 정보
        entity_types = {}
        for node_id, attrs in kg.nodes(data=True):
            entity_type = attrs.get('type', 'Unknown')
            if entity_type not in entity_types:
                entity_types[entity_type] = []
            entity_types[entity_type].append({
                'id': node_id,
                'name': attrs.get('name', ''),
                **{k: v for k, v in attrs.items() if k not in ['type', 'name']}
            })
        
        # 각 엔티티 타입별 정보 추가
        for entity_type, entities in entity_types.items():
            text_parts.append(f"\n{entity_type} 엔티티 ({len(entities)}개):")
            for entity in entities[:10]:  # 최대 10개만
                name = entity.get('name', entity.get('id', ''))
                text_parts.append(f"  - {name}")
                # 속성 추가
                for key, value in entity.items():
                    if key not in ['id', 'name', 'type']:
                        text_parts.append(f"    {key}: {value}")
        
        # 관계 정보 추가
        relations_by_type = {}
        for source, target, attrs in kg.edges(data=True):
            rel_type = attrs.get('type', 'related_to')
            if rel_type not in relations_by_type:
                relations_by_type[rel_type] = []
            
            source_name = kg.nodes[source].get('name', source)
            target_name = kg.nodes[target].get('name', target)
            relations_by_type[rel_type].append(f"{source_name} -> {target_name}")
        
        text_parts.append("\n관계:")
        for rel_type, relations in list(relations_by_type.items())[:5]:  # 최대 5개 타입만
            text_parts.append(f"  {rel_type} ({len(relations)}개):")
            for rel in relations[:5]:  # 각 타입당 최대 5개만
                text_parts.append(f"    - {rel}")
        
        return "\n".join(text_parts)
    
    def _chunk_text(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """텍스트를 청크로 분할"""
        chunks = []
        words = text.split()
        
        if len(words) <= chunk_size:
            return [text]
        
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
            
            start = end - chunk_overlap
        
        return chunks
    
    def query_with_kg(self, question: str, use_kg: bool = True) -> Dict:
        """
        지식 그래프를 활용한 질의응답
        
        Args:
            question: 사용자 질문
            use_kg: 지식 그래프 사용 여부
            
        Returns:
            답변과 관련 문서 정보
        """
        if self.qa_chain is None:
            raise ValueError("지식 베이스가 구축되지 않았습니다. build_knowledge_base()를 먼저 호출하세요.")
        
        # 벡터 검색으로 관련 문서 찾기
        source_documents = self._retriever_for_sources.invoke(question)
        
        # 지식 그래프에서 관련 엔티티 찾기
        kg_context = ""
        if use_kg:
            kg_context = self._extract_kg_context(question, source_documents)
        
        # 컨텍스트 결합
        text_context = "\n\n".join(doc.page_content for doc in source_documents)
        if kg_context:
            full_context = f"{text_context}\n\n지식 그래프 정보:\n{kg_context}"
        else:
            full_context = text_context
        
        # 답변 생성
        prompt_template = """다음 컨텍스트는 문서에서 추출한 표 정보와 지식 그래프 구조입니다.
표의 구조, 헤더 계층, 셀 간의 관계를 바탕으로 질문에 정확하게 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        chain = PROMPT | self.llm | StrOutputParser()
        answer = chain.invoke({"context": full_context, "question": question})
        
        return {
            'answer': answer,
            'source_documents': source_documents,
            'kg_context': kg_context if use_kg else None
        }
    
    def _extract_kg_context(self, question: str, source_documents: List[Document]) -> str:
        """질문과 관련된 지식 그래프 컨텍스트 추출"""
        kg_contexts = []
        
        # 관련 문서의 표 ID 추출
        table_ids = set()
        for doc in source_documents:
            table_id = doc.metadata.get('table_id')
            if table_id and table_id in self.kg_store:
                table_ids.add(table_id)
        
        # 각 표의 지식 그래프에서 관련 엔티티 찾기
        for table_id in table_ids:
            kg = self.kg_store[table_id]
            
            # 질문 키워드와 매칭되는 엔티티 찾기
            question_lower = question.lower()
            relevant_entities = []
            
            for node_id, attrs in kg.nodes(data=True):
                name = attrs.get('name', '').lower()
                if any(keyword in name for keyword in question_lower.split() if len(keyword) > 2):
                    relevant_entities.append(node_id)
            
            # 관련 엔티티의 이웃 노드 찾기
            if relevant_entities:
                context_parts = [f"표 {table_id}의 관련 정보:"]
                visited = set()
                
                for entity_id in relevant_entities[:3]:  # 최대 3개 엔티티만
                    if entity_id in visited:
                        continue
                    visited.add(entity_id)
                    
                    entity_info = kg.nodes[entity_id]
                    context_parts.append(f"  - {entity_info.get('name', entity_id)} ({entity_info.get('type', 'Unknown')})")
                    
                    # 이웃 노드 찾기
                    neighbors = list(kg.neighbors(entity_id))[:3]  # 최대 3개 이웃만
                    for neighbor_id in neighbors:
                        neighbor_info = kg.nodes[neighbor_id]
                        edge_data = kg.get_edge_data(entity_id, neighbor_id)
                        if edge_data:
                            rel_type = list(edge_data.values())[0].get('type', 'related_to')
                            context_parts.append(f"    -> {neighbor_info.get('name', neighbor_id)} ({rel_type})")
                
                kg_contexts.append("\n".join(context_parts))
        
        return "\n\n".join(kg_contexts)
    
    def get_kg_statistics(self) -> Dict:
        """지식 그래프 통계 정보"""
        stats = {
            'total_tables': len(self.kg_store),
            'total_nodes': 0,
            'total_edges': 0,
            'tables': {}
        }
        
        for table_id, kg in self.kg_store.items():
            kg_stats = self.kg_converter.get_graph_statistics()
            stats['total_nodes'] += kg_stats['num_nodes']
            stats['total_edges'] += kg_stats['num_edges']
            stats['tables'][table_id] = kg_stats
        
        return stats

