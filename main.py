import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_openai import OpenAIEmbeddings
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
import tempfile
import uuid

# Anthropic API 키 로드
time.sleep(1)  # 환경 변수 불러오기 전에 1초 대기
anthropic_api_key = st.secrets["anthropic"]["API_KEY"]

# PDF 처리 기능 클래스
class PDFProcessor:
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            for d in documents:
                d.metadata['file_path'] = pdf_path
            return documents
        except Exception as e:
            st.error(f"PDF 로드 중 오류 발생: {e}")
            return []

    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return text_splitter.split_documents(documents)

    @staticmethod
    def save_to_vector_store(documents: List[Document], index_name: str = "faiss_index") -> bool:
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["openai"]["API_KEY"])
            vector_store = FAISS.from_documents(documents, embedding=embeddings)
            vector_store.save_local(index_name)
            return True
        except Exception as e:
            st.error(f"벡터 저장소 생성 중 오류 발생: {e}")
            return False

    @staticmethod
    def process_uploaded_files(uploaded_files) -> bool:
        if not uploaded_files:
            st.error("업로드된 파일이 없습니다.")
            return False

        all_documents = []

        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = temp_file.name

            documents = PDFProcessor.pdf_to_documents(temp_path)
            if documents:
                all_documents.extend(documents)
                st.success(f"{uploaded_file.name} 파일 처리 완료")
            else:
                st.warning(f"{uploaded_file.name} 파일 처리 실패")

            os.unlink(temp_path)

        if not all_documents:
            st.error("모든 파일 처리에 실패했습니다.")
            return False

        smaller_documents = PDFProcessor.chunk_documents(all_documents)

        if "index_name" not in st.session_state:
            st.session_state.index_name = f"faiss_index_{uuid.uuid4().hex[:8]}"

        success = PDFProcessor.save_to_vector_store(smaller_documents, st.session_state.index_name)

        if success:
            st.success(f"총 {len(all_documents)}개의 문서, {len(smaller_documents)}개의 청크가 처리되었습니다.")

        return success

# RAG 시스템
class RAGSystem:
    def __init__(self, api_key: str, index_name: str = "faiss_index"):
        self.api_key = api_key
        self.index_name = index_name

    def get_rag_chain(self) -> Runnable:
        template = """
        아래 컨텍스트를 바탕으로 질문에 답해주세요:

        **사용자가 학과 관련 질문을 하면, 아래 규칙을 따릅니다.**

        1. 응답은 최대 5문장 이내로 작성합니다.
        2. 명확한 답변이 어려울 경우 **"잘 모르겠습니다."**라고 답변합니다.
        3. 공손하고 이해하기 쉬운 표현을 사용합니다.
        4. 질문에 **'디지털경영전공'이라는 단어가 없더라도**, 관련 정보를 PDF에서 찾아 답변합니다.
        5. 사용자의 질문 의도를 정확히 파악하여, **가장 관련성이 높은 정보**를 제공합니다.
        6. 학생이 추가 질문을 할 수 있도록 부드러운 마무리 문장을 사용합니다.
        7. 내용을 사용자 친화적으로 정리해 줍니다.
        8. 한국어 외의 언어로 질문이 들어오면 해당 언어로 답변합니다.

        컨텍스트: {context}

        질문: {question}

        응답:
        """

        custom_rag_prompt = PromptTemplate.from_template(template)
        model = ChatAnthropic(model="claude-3-opus-20240229", anthropic_api_key=self.api_key)

        return custom_rag_prompt | model | StrOutputParser()

    @st.cache_resource
    def get_vector_db(_self, index_name):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=st.secrets["openai"]["API_KEY"])
            return FAISS.load_local(index_name, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"벡터 DB 로드 중 오류 발생: {e}")
            return None

    def process_question(self, user_question: str) -> Tuple[str, List[Document]]:
        vector_db = self.get_vector_db(self.index_name)
        if not vector_db:
            return "시스템 오류가 발생했습니다. PDF 파일을 다시 업로드해주세요.", []

        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        retrieve_docs = retriever.invoke(user_question)

        chain = self.get_rag_chain()

        try:
            response = chain.invoke({"question": user_question, "context": retrieve_docs})
            return response, retrieve_docs
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")
            return "질문 처리 중 오류가 발생했습니다.", []

# UI 클래스
class ChatbotUI:
    @staticmethod
    def save_feedback(questions: List[Dict], feedbacks: List[Dict]) -> bool:
        if not questions and not feedbacks:
            st.warning("저장할 질문 또는 피드백 데이터가 없습니다.")
            return False

        try:
            formatted_questions = [q["질문"] if isinstance(q, dict) else q for q in questions]
            formatted_feedbacks = [f["피드백"] if isinstance(f, dict) else f for f in feedbacks]

            max_length = max(len(formatted_questions), len(formatted_feedbacks))
            formatted_questions.extend([""] * (max_length - len(formatted_questions)))
            formatted_feedbacks.extend([""] * (max_length - len(formatted_feedbacks)))

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
    st.set_page_config(initial_sidebar_state="expanded", layout="wide", page_icon="🤖", page_title="디지털경영전공 챗봇")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "user_questions" not in st.session_state:
        st.session_state.user_questions = []
    if "user_feedback" not in st.session_state:
        st.session_state.user_feedback = []
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "index_name" not in st.session_state:
        st.session_state.index_name = f"faiss_index_{uuid.uuid4().hex[:8]}"

    st.header("디지털경영전공 챗봇")

    left_column, mid_column, right_column = st.columns([1, 2, 1])

    with left_column:
        st.subheader("PDF 업로드")
        uploaded_files = st.file_uploader("PDF 파일을 업로드해주세요 (여러 파일 가능)", type=["pdf"], accept_multiple_files=True)

        if st.button("업로드한 PDF 처리하기", disabled=not uploaded_files):
            with st.spinner("PDF 파일 처리 중..."):
                success = PDFProcessor.process_uploaded_files(uploaded_files)
                if success:
                    st.session_state.pdf_processed = True
                    st.success("모든 PDF 파일 처리가 완료되었습니다!")
                else:
                    st.session_state.pdf_processed = False
                    st.error("PDF 파일 처리 중 오류가 발생했습니다.")

    with mid_column:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("PDF 내용에 대해 궁금한 점을 질문해 주세요.")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            if not st.session_state.pdf_processed:
                with st.chat_message("assistant"):
                    assistant_response = "먼저 왼쪽에서 PDF 파일을 업로드하고 처리해주세요."
                    st.markdown(assistant_response)
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            else:
                rag_system = RAGSystem(anthropic_api_key, st.session_state.index_name)

                with st.spinner("질문에 대한 답변을 생성 중입니다..."):
                    try:
                        response, context = rag_system.process_question(prompt)
                        with st.chat_message("assistant"):
                            st.markdown(response)
                            if context:
                                with st.expander("관련 문서 보기"):
                                    for idx, document in enumerate(context, 1):
                                        st.subheader(f"관련 문서 {idx}")
                                        st.write(document.page_content)
                                        if document.metadata and 'file_path' in document.metadata:
                                            file_name = os.path.basename(document.metadata['file_path'])
                                            st.caption(f"출처: {file_name}")

                        st.session_state.messages.append({"role": "assistant", "content": response})
                    except Exception as e:
                        st.error(f"질문 처리 중 오류 발생: {str(e)}")

    with right_column:
        st.subheader("추가 질문 및 피드백")
        user_question = st.text_input("추가 질문을 남겨주세요!", placeholder="과목 변경 or 행사 문의")

        if st.button("질문 제출"):
            if user_question:
                st.session_state.user_questions.append({"질문": user_question})
                st.success("질문이 제출되었습니다.")
                st.experimental_rerun()
            else:
                st.warning("질문을 입력해주세요.")

        feedback = st.radio("응답이 만족스러우셨나요?", ("만족", "불만족"))

        if feedback == "불만족":
            reason = st.text_area("불만족 사유를 알려주세요.")
            if st.button("피드백 제출"):
                if reason:
                    st.session_state.user_feedback.append({"피드백": reason})
                    st.success("피드백이 제출되었습니다.")
                    st.experimental_rerun()
                else:
                    st.warning("사유를 입력해주세요.")

        ui = ChatbotUI()
        if st.button("질문 및 피드백 CSV로 저장"):
            if ui.save_feedback(st.session_state.user_questions, st.session_state.user_feedback):
                st.success("저장 완료!")
                st.session_state.user_questions = []
                st.session_state.user_feedback = []
                st.experimental_rerun()

if __name__ == "__main__":
    main()
