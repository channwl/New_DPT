import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Tuple, Dict, Any, Optional
import os
import re
import csv
import time
from dotenv import load_dotenv
import openai
import tempfile
import uuid  # 고유 ID 생성용

# 환경 변수 로드 및 검증 개선
time.sleep(1)  # 환경 변수 불러오기 전에 1초 대기
openai.api_key = st.secrets["openai"]["API_KEY"] #OpenAI API 키를 st.secrets에서 가져와 api_key 변수에 저장
api_key = openai.api_key  # 변수에 저장하여 나중에 사용

# 모듈화: PDF 처리 기능을 클래스로 분리, PDFProcessor 클래스 (PDF 처리)
class PDFProcessor:
    @staticmethod # 클래스에서 객체를 생성하지 않고, 클래스 이름으로 바로 호출할 수 있다.
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        """PDF 파일을 Document 객체 리스트로 변환"""
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            for d in documents:
                d.metadata['file_path'] = pdf_path
            return documents
        except Exception as e:
            st.error(f"PDF 로드 중 오류 발생: {e}") #사용자 친화적 시스템 : Streamlit에서 에러 표시해주기
            return []

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        """Document를 더 작은 단위로 분할"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    @staticmethod
    def save_to_vector_store(documents: List[Document], index_name: str = "faiss_index") -> bool:
        """Document를 벡터 DB에 저장"""
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local(index_name)
            return True
        except Exception as e:
            st.error(f"벡터 저장소 생성 중 오류 발생: {e}")
            return False

    @staticmethod
    def process_uploaded_files(uploaded_files) -> bool:
        """업로드된 PDF 파일 처리 작업 통합"""
        if not uploaded_files:
            st.error("업로드된 파일이 없습니다.")
            return False
        
        all_documents = []
        
        # 업로드된 모든 파일 처리
        for uploaded_file in uploaded_files:
            # 임시 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name
            
            # PDF 문서 추출
            documents = PDFProcessor.pdf_to_documents(temp_path)
            if documents:
                all_documents.extend(documents)
                st.success(f"{uploaded_file.name} 파일 처리 완료")
            else:
                st.warning(f"{uploaded_file.name} 파일 처리 실패")
            
            # 임시 파일 삭제
            os.unlink(temp_path)
        
        if not all_documents:
            st.error("모든 파일 처리에 실패했습니다.")
            return False
        
        # 문서 분할
        smaller_documents = PDFProcessor.chunk_documents(all_documents)
        
        # 세션에 고유한 인덱스 이름 생성 또는 사용
        if "index_name" not in st.session_state:
            st.session_state.index_name = f"faiss_index_{uuid.uuid4().hex[:8]}"
        
        # 벡터 저장소에 저장
        success = PDFProcessor.save_to_vector_store(smaller_documents, st.session_state.index_name)
        
        if success:
            st.success(f"총 {len(all_documents)}개의 문서, {len(smaller_documents)}개의 청크가 처리되었습니다.")
        
        return success

# 모듈화: RAG 시스템 기능을 클래스로 분리
class RAGSystem:
    def __init__(self, api_key: str, index_name: str = "faiss_index"):
        self.api_key = api_key
        self.index_name = index_name
        
    def get_rag_chain(self) -> Runnable:
        """RAG 체인 생성"""
        template = """
        아래 컨텍스트를 바탕으로 질문에 답해주세요:

        **사용자가 학과 관련 질문을 하면, 아래 규칙을 따릅니다.**

        이 프롬포트는, 학과 챗봇을 위한 프롬포팅이야.

        1. 응답은 최대 5문장 이내로 작성합니다.
        2. 명확한 답변이 어려울 경우 **"잘 모르겠습니다."**라고 답변합니다.
        3. 공손하고 이해하기 쉬운 표현을 사용합니다.
        4. 질문에 **'디지털경영전공'이라는 단어가 없더라도**, 관련 정보를 PDF에서 찾아 답변합니다.
        5. 사용자의 질문 의도를 정확히 파악하여, **가장 관련성이 높은 정보**를 제공합니다.
        6. 학생이 추가 질문을 할 수 있도록 부드러운 마무리 문장을 사용합니다. (예: “더 궁금한 점이 있으면 편하게 물어봐 !”)
        7. 대화 흐름을 유지하기 위해, 학생의 질문 의도를 고려하고 **적절한 후속 질문을 던져 더 깊이 있는 대화를 유도합니다.
        8. 내용을 사용자가 알아보기 쉽게 정리해서 나열합니다.
        

        컨텍스트: {context}

        질문: {question}

        응답:
        """


        custom_rag_prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)

        return custom_rag_prompt | model | StrOutputParser()
    
    @st.cache_resource
    def get_vector_db(_self, index_name):  # 첫 번째 인자로 self를 받되 사용하지 않도록 _self로 이름 변경
        """벡터 DB 로드"""
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"벡터 DB 로드 중 오류 발생: {e}")
            return None
    
    def process_question(self, user_question: str) -> Tuple[str, List[Document]]:
        """사용자 질문에 대한 RAG 처리"""
        vector_db = self.get_vector_db(self.index_name)
        if not vector_db:
            return "시스템 오류가 발생했습니다. PDF 파일을 다시 업로드해주세요.", []
            
        # 관련 문서 검색
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        retrieve_docs = retriever.invoke(user_question)
        
        # RAG 체인 호출
        chain = self.get_rag_chain()
        
        try:
            response = chain.invoke({"question": user_question, "context": retrieve_docs})
            return response, retrieve_docs
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")
            return "질문 처리 중 오류가 발생했습니다. 다시 시도해주세요.", []

# 모듈화: UI 관련 기능을 클래스로 분리
class ChatbotUI:
    @staticmethod
    def natural_sort_key(s):
        """파일명 자연 정렬 키 생성"""
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
    
    @staticmethod
    def save_feedback(questions: List[Dict], feedbacks: List[Dict]) -> bool:
        """사용자 질문 및 피드백을 CSV로 저장"""
        if not questions and not feedbacks:
            st.warning("저장할 질문 또는 피드백 데이터가 없습니다.")
            return False
            
        try:
            # 질문과 피드백 형식 통일 (딕셔너리/문자열 모두 처리)
            formatted_questions = []
            for q in questions:
                if isinstance(q, dict) and "질문" in q:
                    formatted_questions.append(q["질문"])
                elif isinstance(q, str):
                    formatted_questions.append(q)
                    
            formatted_feedbacks = []
            for f in feedbacks:
                if isinstance(f, dict) and "피드백" in f:
                    formatted_feedbacks.append(f["피드백"])
                elif isinstance(f, str):
                    formatted_feedbacks.append(f)
            
            # 길이 맞추기
            max_length = max(len(formatted_questions), len(formatted_feedbacks))
            formatted_questions.extend([""] * (max_length - len(formatted_questions)))
            formatted_feedbacks.extend([""] * (max_length - len(formatted_feedbacks)))
            
            # CSV 저장
            with open("questions_and_feedback.csv", mode="w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["질문", "피드백"])
                for q, f in zip(formatted_questions, formatted_feedbacks):
                    writer.writerow([q, f])
            return True
            
        except Exception as e:
            st.error(f"피드백 저장 중 오류 발생: {e}")
            return False

def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon="🤖",
        page_title="디지털경영전공 챗봇")

    # 세션 상태 초기화를 한곳에서 처리
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = []
    # PDF 처리 상태 추적용 변수
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    # 인덱스 이름 저장용 변수
    if "index_name" not in st.session_state:
        st.session_state.index_name = f"faiss_index_{uuid.uuid4().hex[:8]}"

    # UI 초기화
    st.header("디지털경영전공 챗봇")

    # 레이아웃
    left_column, mid_column, right_column = st.columns([1, 2, 1])
    
    # 왼쪽 열 - PDF 업로드 및 처리
    with left_column:
        st.subheader("PDF 업로드")
        uploaded_files = st.file_uploader(
            "PDF 파일을 업로드해주세요 (여러 파일 선택 가능)",
            type=["pdf"],
            accept_multiple_files=True
        )
        
        if st.button("업로드한 PDF 처리하기", disabled=not uploaded_files):
            with st.spinner("PDF 파일 처리 중..."):
                success = PDFProcessor.process_uploaded_files(uploaded_files)
                if success:
                    st.session_state.pdf_processed = True
                    st.success("모든 PDF 파일 처리가 완료되었습니다!")
                else:
                    st.session_state.pdf_processed = False
                    st.error("PDF 파일 처리 중 오류가 발생했습니다.")
        
        # 사용 설명서
        with st.expander("사용 방법"):
            st.markdown("""
            1. 왼쪽에서 PDF 파일을 업로드합니다 (여러 파일 가능).
            2. '업로드한 PDF 처리하기' 버튼을 클릭합니다.
            3. 중앙의 입력창에 질문을 입력합니다.
            4. 챗봇은 업로드한 PDF 파일의 내용을 기반으로 답변합니다.
            5. 새로운 PDF로 변경하려면 다시 업로드하고 처리하세요.
            """)

    # 중앙 열 - 채팅 인터페이스
    with mid_column:
        # 대화 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 메시지 입력 및 처리
        prompt = st.chat_input("PDF 내용에 대해 궁금한 점을 질문해 주세요.")
        
        if prompt:
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # PDF 처리 상태 확인 로직 추가
            if not st.session_state.pdf_processed:
                with st.chat_message("assistant"):
                    assistant_response = "먼저 왼쪽에서 PDF 파일을 업로드하고 처리해주세요."
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                # RAG 시스템 초기화
                rag_system = RAGSystem(api_key, st.session_state.index_name)
                
                # 질문 처리 및 응답
                with st.spinner("질문에 대한 답변을 생성 중입니다..."):
                    try:
                        response, context = rag_system.process_question(prompt)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            if context:
                                # 관련 문서 표시 방식 개선
                                with st.expander("관련 문서 보기"):
                                    for idx, document in enumerate(context, 1):
                                        st.subheader(f"관련 문서 {idx}")
                                        st.write(document.page_content)
                                        
                                        # 메타데이터 표시 (파일 정보 등)
                                        if document.metadata and 'file_path' in document.metadata:
                                            file_name = os.path.basename(document.metadata['file_path'])
                                            st.caption(f"출처: {file_name}")
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"질문을 처리하는 중 오류가 발생했습니다: {str(e)}")

    # 오른쪽 열 - 피드백 및 추가 질문
    with right_column:
        # 섹션 제목 추가로 UI 명확성 향상
        st.subheader("추가 질문 및 피드백")
        
        # 추가 질문 섹션
        st.text("추가 질문")
        user_question = st.text_input(
            "챗봇을 통해 정보를 얻지 못하였거나 추가적으로 궁금한 질문을 남겨주세요!",
            placeholder="과목 변경 or 행사 문의"
        )

        # 버튼에 key 추가로 중복 방지
        if st.button("질문 제출", key="submit_question"):
            if user_question:
                st.session_state.user_questions.append({"질문": user_question})
                st.success("질문이 제출되었습니다.")
                # 입력 필드 초기화를 위한 페이지 새로고침
                st.experimental_rerun()
            else:
                st.warning("질문을 입력해주세요.")

        # 피드백 섹션
        st.text("")
        st.text("응답 피드백")
        feedback = st.radio("응답이 만족스러우셨나요?", ("만족", "불만족"))

        if feedback == "만족":
            st.success("감사합니다! 도움이 되어 기쁩니다.")
        elif feedback == "불만족":
            st.warning("불만족하신 부분을 개선하기 위해 노력하겠습니다.")
            
            # 불만족 사유 입력
            reason = st.text_area("불만족한 부분이 무엇인지 말씀해 주세요.")

            # 버튼에 key 추가로 중복 방지
            if st.button("피드백 제출", key="submit_feedback"):
                if reason:
                    st.session_state.user_feedback.append({"피드백": reason})
                    st.success("피드백이 제출되었습니다.")
                    # 입력 필드 초기화를 위한 페이지 새로고침
                    st.experimental_rerun()
                else:
                    st.warning("불만족 사유를 입력해 주세요.")

        # 질문 및 피드백 CSV 저장
        st.text("")
        
        ui = ChatbotUI()  # UI 클래스 초기화
        if st.button("질문 및 피드백 등록하기"):
            # 개선된 피드백 저장 함수 사용
            if ui.save_feedback(st.session_state.user_questions, st.session_state.user_feedback):
                st.success("질문과 피드백이 등록되었습니다.")
                # 등록 후 목록 초기화
                st.session_state.user_questions = []
                st.session_state.user_feedback = []
                time.sleep(1)
                st.experimental_rerun()

        # 문의 정보
        st.text("")
        st.text("")
        # 텍스트를 markdown으로 변경하여 가독성 향상
        st.markdown("""
        고려대학교 세종캠퍼스 디지털경영전공 홈페이지를 참고하거나,
        디지털경영전공 사무실(044-860-1560)에 전화하여 문의사항을 접수하세요.
        """)

if __name__ == "__main__":
    main()

# start : streamlit run app.py
# stop : ctrl + c
