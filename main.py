import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import Runnable
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyMuPDFLoader
from typing import List, Tuple, Dict, Any, Optional  # [변경] 추가 타입 임포트
import os
import re
import csv
import time  # [추가] time 모듈 추가
from dotenv import load_dotenv

# [변경] 환경 변수 로드 및 검증 개선
load_dotenv()
api_key = os.getenv("YOURKEY")
if not api_key:
    raise ValueError("API 키가 설정되지 않았습니다. .env 파일을 확인해주세요.")

# [변경] 모듈화: PDF 처리 기능을 클래스로 분리
class PDFProcessor:
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        """PDF 파일을 Document 객체 리스트로 변환"""
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            for d in documents:
                d.metadata['file_path'] = pdf_path
            return documents
        except Exception as e:  # [추가] 예외 처리 추가
            st.error(f"PDF 로드 중 오류 발생: {e}")
            return []

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        """Document를 더 작은 단위로 분할"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    @staticmethod
    def save_to_vector_store(documents: List[Document]) -> bool:  # [변경] 성공/실패 여부 반환
        """Document를 벡터 DB에 저장"""
        try:
            # [변경] API 키를 명시적으로 전달
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local("faiss_index")
            return True
        except Exception as e:  # [추가] 예외 처리 추가
            st.error(f"벡터 저장소 생성 중 오류 발생: {e}")
            return False

    @staticmethod
    def process_pdf(pdf_path: str) -> bool:  # [추가] PDF 처리를 통합하는 메서드 추가
        """PDF 파일 처리 작업 통합"""
        if not os.path.exists(pdf_path):
            st.error(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")
            return False
            
        documents = PDFProcessor.pdf_to_documents(pdf_path)
        if not documents:
            return False
            
        smaller_documents = PDFProcessor.chunk_documents(documents)
        return PDFProcessor.save_to_vector_store(smaller_documents)

# [변경] 모듈화: RAG 시스템 기능을 클래스로 분리
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def get_rag_chain(self) -> Runnable:
        """RAG 체인 생성"""
        template = """
        아래 컨텍스트를 바탕으로 질문에 답해주세요:
        - 질문에 대한 응답은 5줄 이내로 간결하게 작성해주세요.
        - 애매하거나 모르는 내용은 "잘 모르겠습니다"라고 답변해주세요.
        - 공손한 표현을 사용해주세요.

        컨텍스트: {context}

        질문: {question}

        응답:"""

        custom_rag_prompt = PromptTemplate.from_template(template)
        # [변경] API 키를 명시적으로 전달
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)

        return custom_rag_prompt | model | StrOutputParser()
    
    # [변경] 캐싱 데코레이터 수정 (@st.cache_data 제거, @st.cache_resource만 사용)
    @st.cache_resource
    def get_vector_db(self):
        """벡터 DB 로드"""
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=self.api_key)
            return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:  # [추가] 예외 처리 추가
            st.error(f"벡터 DB 로드 중 오류 발생: {e}")
            return None
    
    def process_question(self, user_question: str) -> Tuple[str, List[Document]]:  # [변경] 반환 타입 명시
        """사용자 질문에 대한 RAG 처리"""
        vector_db = self.get_vector_db()
        if not vector_db:  # [추가] DB 로드 실패 시 처리
            return "시스템 오류가 발생했습니다. 잠시 후 다시 시도해주세요.", []
            
        # 관련 문서 검색
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        retrieve_docs = retriever.invoke(user_question)
        
        # RAG 체인 호출
        chain = self.get_rag_chain()
        
        try:
            response = chain.invoke({"question": user_question, "context": retrieve_docs})
            return response, retrieve_docs
        except Exception as e:  # [추가] 예외 처리 추가
            st.error(f"응답 생성 중 오류 발생: {e}")
            return "질문 처리 중 오류가 발생했습니다. 다시 시도해주세요.", []

# [추가] 모듈화: UI 관련 기능을 클래스로 분리
class ChatbotUI:
    @staticmethod
    def create_buttons(options):
        """버튼 생성"""
        for option in options:
            if isinstance(option, tuple):
                label, value = option
            else:
                label, value = option, option
                
            if st.button(label):
                st.session_state.selected_category = value
                return True  # [변경] 버튼 클릭 시 True 반환하도록 수정
        return False
                
    @staticmethod
    def natural_sort_key(s):
        """파일명 자연 정렬 키 생성"""
        return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', s)]
    
    @staticmethod
    def save_feedback(questions: List[Dict], feedbacks: List[Dict]) -> bool:  # [변경] 성공/실패 여부 반환
        """사용자 질문 및 피드백을 CSV로 저장"""
        if not questions and not feedbacks:
            st.warning("저장할 질문 또는 피드백 데이터가 없습니다.")
            return False
            
        try:
            # [변경] 질문과 피드백 형식 통일 (딕셔너리/문자열 모두 처리)
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
            
        except Exception as e:  # [추가] 예외 처리 추가
            st.error(f"피드백 저장 중 오류 발생: {e}")
            return False

def main():
    st.set_page_config(
        initial_sidebar_state="expanded",
        layout="wide",
        page_icon="🤖",
        page_title="디지털경영전공 챗봇")

    # [변경] 세션 상태 초기화를 한곳에서 처리
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_category" not in st.session_state:
        st.session_state.selected_category = None
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = []
    # [추가] PDF 처리 상태 추적용 변수
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False

    # UI 초기화
    st.header("디지털경영전공 챗봇")
    st.text("질문하고싶은 카테고리를 선택해주세요")

    # [추가] RAG 시스템 및 UI 클래스 초기화
    rag_system = RAGSystem(api_key)
    ui = ChatbotUI()

    # 레이아웃
    left1_column, left2_column, mid_column, right_column = st.columns([0.3, 0.3, 1, 0.9])
    
    # 왼쪽 첫 번째 열 - 카테고리
    with left1_column:
        st.text("디지털경영학과")
        
        categories = [
            "학과 정보", "전공 과목", "교내 장학금", "학교 행사",
            "소모임", "비교과", "교환 학생"]

        # [변경] 버튼 클릭 결과에 따른 처리 개선
        if ui.create_buttons(categories) and st.session_state.selected_category:
            # [추가] 스피너 추가로 사용자에게 처리 중임을 알림
            with st.spinner(f"{st.session_state.selected_category} 정보를 준비 중입니다..."):
                pdf_path = f"{st.session_state.selected_category}.pdf"
                st.session_state.pdf_processed = PDFProcessor.process_pdf(pdf_path)

    # 왼쪽 두 번째 열 - 학년별
    with left2_column:
        st.text("학년별")

        grade_levels = [
            ("20학번 이전", "20이전"), ("21학번", "21"),
            ("22학번", "22"), ("23학번", "23"), ("24학번", "24")]

        # [변경] 버튼 클릭 결과에 따른 처리 개선
        if ui.create_buttons(grade_levels) and st.session_state.selected_category:
            # [추가] 스피너 추가로 사용자에게 처리 중임을 알림
            with st.spinner(f"{st.session_state.selected_category} 정보를 준비 중입니다..."):
                pdf_path = f"{st.session_state.selected_category}.pdf"
                st.session_state.pdf_processed = PDFProcessor.process_pdf(pdf_path)

    # 중앙 열 - 채팅 인터페이스
    with mid_column:
        # 대화 히스토리 표시
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 사용자 메시지 입력 및 처리
        prompt = st.chat_input("선택하신 카테고리에서 궁금한 점을 질문해 주세요.")
        
        if prompt:
            # 사용자 메시지 표시
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # [추가] 카테고리 선택 여부 확인 로직 추가
            if not st.session_state.selected_category:
                with st.chat_message("assistant"):
                    assistant_response = "먼저 왼쪽에서 카테고리를 선택해주세요."
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            # [추가] PDF 처리 상태 확인 로직 추가
            elif not st.session_state.pdf_processed:
                with st.chat_message("assistant"):
                    assistant_response = "데이터 처리 중 문제가 발생했습니다. 다시 카테고리를 선택해주세요."
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                # 질문 처리 및 응답
                # [추가] 스피너 추가로 사용자에게 처리 중임을 알림
                with st.spinner("질문에 대한 답변을 생성 중입니다..."):
                    try:
                        response, context = rag_system.process_question(prompt)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            if context:
                                # [개선] 관련 문서 표시 방식 개선
                                with st.expander("관련 문서 보기"):
                                    for idx, document in enumerate(context, 1):
                                        st.subheader(f"관련 문서 {idx}")
                                        st.write(document.page_content)
                        
                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"질문을 처리하는 중 오류가 발생했습니다: {str(e)}")

    # 오른쪽 열 - 피드백 및 추가 질문
    with right_column:
        # [변경] 섹션 제목 추가로 UI 명확성 향상
        st.subheader("추가 질문 및 피드백")
        
        # 추가 질문 섹션
        st.text("추가 질문")
        user_question = st.text_input(
            "챗봇을 통해 정보를 얻지 못하였거나 추가적으로 궁금한 질문을 남겨주세요!",
            placeholder="과목 변경 or 행사 문의"
        )

        # [변경] 버튼에 key 추가로 중복 방지
        if st.button("질문 제출", key="submit_question"):
            if user_question:
                st.session_state.user_questions.append({"질문": user_question})
                st.success("질문이 제출되었습니다.")
                # [추가] 입력 필드 초기화를 위한 페이지 새로고침
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

            # [변경] 버튼에 key 추가로 중복 방지
            if st.button("피드백 제출", key="submit_feedback"):
                if reason:
                    st.session_state.user_feedback.append({"피드백": reason})
                    st.success("피드백이 제출되었습니다.")
                    # [추가] 입력 필드 초기화를 위한 페이지 새로고침
                    st.experimental_rerun()
                else:
                    st.warning("불만족 사유를 입력해 주세요.")

        # 질문 및 피드백 CSV 저장
        st.text("")
        if st.button("질문 및 피드백 등록하기"):
            # [변경] 개선된 피드백 저장 함수 사용
            if ui.save_feedback(st.session_state.user_questions, st.session_state.user_feedback):
                st.success("질문과 피드백이 등록되었습니다.")
                # [추가] 등록 후 목록 초기화
                st.session_state.user_questions = []
                st.session_state.user_feedback = []
                time.sleep(1)
                st.experimental_rerun()

        # 문의 정보
        st.text("")
        st.text("")
        # [변경] 텍스트를 markdown으로 변경하여 가독성 향상
        st.markdown("""
        고려대학교 세종캠퍼스 디지털경영전공 홈페이지를 참고하거나,
        디지털경영전공 사무실(044-860-1560)에 전화하여 문의사항을 접수하세요.
        """)

if __name__ == "__main__":
    main()

# start : streamlit run end.py
# stop : ctrl + c
