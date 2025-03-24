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
import tempfile
import uuid

# OpenAI API 키 로드
time.sleep(1)
api_key = st.secrets["openai"]["API_KEY"]

# PDF 인덱스 생성 스크립트 (한 번만 실행)
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        st.warning("data/ 폴더가 생성되었습니다. PDF 파일을 여기에 넣고 다시 실행하세요.")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        st.error("data/ 폴더에 PDF 파일이 없습니다. PDF를 추가한 후 다시 실행하세요.")
        return

    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    st.success(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성 완료!")

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

# RAG 시스템 (OpenAI 전용)
class RAGSystem:
    def __init__(self, api_key: str, index_name: str = "faiss_index_internal"):
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

        답변:
        """

        custom_rag_prompt = PromptTemplate.from_template(template)
        model = ChatOpenAI(model="gpt-4o", openai_api_key=self.api_key)

        return custom_rag_prompt | model | StrOutputParser()

    @st.cache_resource
    def get_vector_db(_self):
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
            return FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"벡터 DB 로드 중 오류 발생: {e}")
            return None

    def process_question(self, user_question: str) -> Tuple[str, List[Document]]:
        vector_db = self.get_vector_db()
        if not vector_db:
            return "시스템 오류가 발생했습니다. PDF 인덱스를 다시 생성해주세요.", []

        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        retrieve_docs = retriever.invoke(user_question)

        chain = self.get_rag_chain()

        try:
            response = chain.invoke({"question": user_question, "context": retrieve_docs})
            return response, retrieve_docs
        except Exception as e:
            st.error(f"응답 생성 중 오류 발생: {e}")
            return "질문 처리 중 오류가 발생했습니다.", []

def main():
    st.set_page_config(initial_sidebar_state="expanded", layout="wide", page_icon="🤖", page_title="디지털경영전공 챗봇")

    if st.button("📥 (관리자) 인덱스 다시 생성하기"):
        generate_faiss_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("🎓 디지털경영전공 챗봇")

    left_column, mid_column, right_column = st.columns([1, 2, 1])

    with mid_column:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        prompt = st.chat_input("궁금한 점을 입력해 주세요.")

        if prompt:
            with st.chat_message("user"):
                st.markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            rag_system = RAGSystem(api_key)

            with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요 😊"):
                response, context = rag_system.process_question(prompt)
                with st.chat_message("assistant"):
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    with right_column:
        st.subheader("📢 피드백 남기기")
        feedback = st.text_area("개발자에게 전하고 싶은 말을 작성해 주세요!")

        if st.button("피드백 제출"):
            if feedback.strip():
                with open("feedback_log.csv", mode="a", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), feedback])
                st.success("피드백이 제출되었습니다!")
                st.rerun()
            else:
                st.warning("피드백 내용을 입력해 주세요.")

if __name__ == "__main__":
    main()
