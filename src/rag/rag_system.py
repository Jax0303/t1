"""
RAG 시스템 구현
표 데이터를 임베딩하고 검색하여 질문에 답변
"""
import os
from typing import List, Dict, Optional
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import chromadb
from chromadb.config import Settings


class TableRAGSystem:
    """표 기반 RAG 시스템"""
    
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini", provider: str = "openai", 
                 ollama_base_url: str = "http://localhost:11434"):
        """
        Args:
            api_key: API 키 (OpenAI, Gemini 또는 Ollama)
            model_name: 사용할 모델명
            provider: "openai", "gemini" 또는 "ollama"
            ollama_base_url: Ollama API 기본 URL (provider가 ollama인 경우)
        """
        self.api_key = api_key
        self.model_name = model_name
        self.provider = provider.lower()
        self.ollama_base_url = ollama_base_url
        
        # 임베딩 모델 초기화
        try:
            if self.provider == "openai":
                from langchain_openai import OpenAIEmbeddings
                self.embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    model="text-embedding-3-small"
                )
            elif self.provider == "gemini":
                # Gemini 임베딩
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                self.embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/text-embedding-004",
                    google_api_key=api_key
                )
            elif self.provider == "ollama":
                # Ollama 임베딩 (로컬 모델)
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                    self.embeddings = OllamaEmbeddings(
                        model=model_name,
                        base_url=ollama_base_url
                    )
                except ImportError:
                    # Ollama 임베딩이 없으면 FakeEmbeddings 사용
                    from langchain_core.embeddings import FakeEmbeddings
                    self.embeddings = FakeEmbeddings(size=384)
            else:
                # 기본: FakeEmbeddings
                from langchain_core.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=384)
        except Exception as e:
            print(f"경고: {self.provider} 임베딩 사용 불가 ({str(e)[:100]}). 간단한 임베딩 방식 사용.")
            # 간단한 임베딩 (FakeEmbeddings - 빠른 테스트용)
            try:
                from langchain.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=384)
            except ImportError:
                from langchain_core.embeddings import FakeEmbeddings
                self.embeddings = FakeEmbeddings(size=384)
        
        # LLM 초기화
        try:
            if self.provider == "openai":
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model=model_name,
                    openai_api_key=api_key,
                    temperature=0.0,
                    max_retries=3,
                    timeout=60
                )
            elif self.provider == "gemini":
                # Gemini LLM
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model=model_name,
                    google_api_key=api_key,
                    temperature=0.0,
                    max_retries=3,
                    timeout=60
                )
            elif self.provider == "ollama":
                # Ollama LLM (로컬 모델)
                try:
                    from langchain_community.llms import Ollama
                    self.llm = Ollama(
                        model=model_name,
                        base_url=ollama_base_url,
                        temperature=0.0
                    )
                except ImportError:
                    # langchain_community가 없으면 langchain_ollama 사용
                    try:
                        from langchain_ollama import ChatOllama
                        self.llm = ChatOllama(
                            model=model_name,
                            base_url=ollama_base_url,
                            temperature=0.0
                        )
                    except ImportError:
                        raise ValueError("Ollama 지원을 위해 langchain-community 또는 langchain-ollama가 필요합니다.")
            else:
                raise ValueError(f"지원하지 않는 provider: {self.provider}")
        except Exception as e:
            # API 키 오류나 한도 초과 시 ollama로 폴백
            if "quota" in str(e).lower() or "limit" in str(e).lower() or "403" in str(e) or "429" in str(e):
                print(f"경고: {self.provider} API 한도 초과 또는 오류. Ollama로 전환 시도...")
                try:
                    from langchain_community.llms import Ollama
                    self.llm = Ollama(model="llama3.2", base_url="http://localhost:11434", temperature=0.0)
                    self.provider = "ollama"
                    print("Ollama로 전환 성공")
                except:
                    raise ValueError(f"LLM 초기화 실패 ({self.provider}): {str(e)}")
            else:
                raise ValueError(f"LLM 초기화 실패 ({self.provider}): {str(e)}")
        
        # 벡터 스토어 초기화
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
    
    def build_knowledge_base(self, tables: List[Dict], chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        표 데이터로부터 지식 베이스 구축
        
        Args:
            tables: 추출된 표 리스트
            chunk_size: 청크 크기
            chunk_overlap: 청크 오버랩
        """
        documents = []
        
        for table_info in tables:
            df = table_info['dataframe']
            table_id = table_info.get('table_id', 'unknown')
            source_file = table_info.get('source_file', 'unknown')
            
            # 표를 텍스트로 변환 (구조 정보 보존)
            table_text = self._table_to_text(df, table_id)
            
            # 청킹
            chunks = self._chunk_text(table_text, chunk_size, chunk_overlap)
            
            for chunk_idx, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'table_id': table_id,
                        'source_file': source_file,
                        'chunk_index': chunk_idx,
                        'extraction_method': table_info.get('extraction_method', 'unknown')
                    }
                )
                documents.append(doc)
        
        # ChromaDB에 저장 (권한 문제 대비)
        persist_directory = "./chroma_db"
        try:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory
            )
        except Exception as e:
            if "readonly" in str(e).lower() or "permission" in str(e).lower():
                # 메모리 모드로 전환
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
        
        # QA 체인 생성 (LangChain 1.0+ 방식)
        prompt_template = """다음 컨텍스트는 문서에서 추출한 표 정보입니다. 
표의 구조와 내용을 바탕으로 질문에 정확하게 답변해주세요.

컨텍스트:
{context}

질문: {question}

답변:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # LangChain 1.0+ 방식으로 체인 구성
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
        
        # source_documents를 가져오기 위한 별도 retriever 저장
        self._retriever_for_sources = self.retriever
    
    def _table_to_text(self, df: pd.DataFrame, table_id: str) -> str:
        """DataFrame을 구조화된 텍스트로 변환"""
        text_parts = [f"표 ID: {table_id}"]
        text_parts.append(f"행 수: {len(df)}, 열 수: {len(df.columns)}")
        
        # 헤더
        headers = " | ".join(str(col) for col in df.columns)
        text_parts.append(f"헤더: {headers}")
        
        # 데이터 행
        text_parts.append("데이터:")
        for idx, row in df.iterrows():
            row_data = " | ".join(str(val) for val in row.values)
            text_parts.append(f"행 {idx}: {row_data}")
        
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
    
    def query(self, question: str) -> Dict:
        """
        질문에 대한 답변 검색
        
        Args:
            question: 사용자 질문
            
        Returns:
            답변과 관련 문서 정보
        """
        if self.qa_chain is None:
            raise ValueError("지식 베이스가 구축되지 않았습니다. build_knowledge_base()를 먼저 호출하세요.")
        
        # 답변 생성
        answer = self.qa_chain.invoke(question)
        
        # 관련 문서 검색
        source_documents = self._retriever_for_sources.invoke(question)
        
        return {
            'answer': answer,
            'source_documents': source_documents
        }
    
    def clear_knowledge_base(self):
        """지식 베이스 초기화"""
        if self.vector_store:
            # ChromaDB 데이터 삭제
            persist_directory = "./chroma_db"
            import shutil
            if os.path.exists(persist_directory):
                shutil.rmtree(persist_directory)
            self.vector_store = None
            self.retriever = None
            self.qa_chain = None

