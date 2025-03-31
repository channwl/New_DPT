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
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationChain
from typing import List, Tuple
import os
import csv
import time
import uuid

# API 키 로드
time.sleep(1)
api_key = st.secrets["anthropic"]["API_KEY"]

# PDF 처리 클래스 정의
class PDFProcessor:
    #PDF를 문서 list로 변환
    @staticmethod
    def pdf_to_documents(pdf_path: str) -> List[Document]:
        loader = PyMuPDFLoader(pdf_path)
        documents = loader.load()
        for d in documents:
            d.metadata['file_path'] = pdf_path
        return documents
    #chunking!
    @staticmethod
    def chunk_documents(documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        return splitter.split_documents(documents)

# PDF 인덱스 생성 함수
def generate_faiss_index():
    pdf_dir = "data/"
    all_documents = []
    
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        st.warning("data/ 폴더가 생성되었습니다. PDF 파일을 여기에 넣고 다시 실행해주세요.")
        return

    pdf_files = [file for file in os.listdir(pdf_dir) if file.endswith(".pdf")]
    if not pdf_files:
        st.error("data/ 폴더에 PDF 파일이 없습니다. PDF를 추가한 후 다시 실행해주세요.")
        return

    #PDF 파일 문서화
    for file_name in pdf_files:
        docs = PDFProcessor.pdf_to_documents(os.path.join(pdf_dir, file_name))
        all_documents.extend(docs)

    #문서 chunking, vector embedding 생성, 인덱싱
    chunks = PDFProcessor.chunk_documents(all_documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_internal")
    st.success(f"{len(pdf_files)}개의 PDF 파일로 인덱스 생성이 완료되었습니다.")

#RAG
class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Claude 3.7 Sonnet 모델로 변경
        self.llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219", 
            anthropic_api_key=self.api_key,
            temperature=0.1,  # 일관된 응답을 위해 낮은 온도 설정
            max_tokens=1000
        )
        
        # 대화 기억 모듈 초기화
        self.memory = ConversationSummaryMemory(llm=self.llm)
        self.conversation_chain = ConversationChain(
            llm=self.llm, 
            memory=self.memory, 
            verbose=True
        )

    def get_rag_chain(self) -> Runnable:
        template = """
        📚 대화 컨텍스트 기반 맞춤형 응답 가이드라인:

        1. 답변은 최대 4문장 이내로 간결하고 명확하게 작성합니다.
        2. 중요한 내용은 핵심만 요약해서 전달합니다.
        3. 답변이 어려우면 "잘 모르겠습니다."라고 정중히 답변합니다.
        4. 질문에 '디지털경영전공' 단어가 없어도 관련 정보를 PDF에서 찾아서 답변합니다.
        5. 이해하기 쉬운 짧은 문장과 불릿 포인트로 정리합니다.
        6. 마지막에 "추가로 궁금하신 점이 있다면 언제든지 말씀해주세요."라고 안내합니다.
        7. 한국어 외 언어로 질문 시 해당 언어로 번역하여 답변합니다.
        8. 관련된 참고 사항이 있다면 간단히 덧붙입니다.
        9. 챗봇 어투는 항상 친절하고 단정하게 유지합니다.
        10. 핵심 내용은 **굵게** 표시해 강조합니다.
        11. 복잡한 정보는 **불릿 포인트**로 요약 정리합니다.
        12. 전공 과목 안내 시에는 전체 리스트를 구체적으로 나열합니다.
        13. 추가 안내는 "추가로 궁금한 점이 있다면 언제든지 말씀해주세요."로 마무리합니다.
        14. 같은 말을 반복하지 마세요

        대화 이력: {history}
        PDF 컨텍스트: {context}
        현재 질문: {question}

        응답 작성:
        """
        prompt = PromptTemplate.from_template(template)
        
        # Claude 3.7 Sonnet 모델 사용
        model = ChatAnthropic(
            model="claude-3-7-sonnet-20250219", 
            anthropic_api_key=self.api_key,
            temperature=0.1,
            max_tokens=1000
        )
        return prompt | model | StrOutputParser()

    def process_question(self, question: str) -> str:
        # 벡터 데이터베이스에서 관련 문서 검색 (OpenAI 임베딩 그대로 사용)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=os.environ.get("OPENAI_API_KEY"))
        vector_db = FAISS.load_local("faiss_index_internal", embeddings, allow_dangerous_deserialization=True)
        retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(question)
        
        # 대화 기록 요약 가져오기
        conversation_history = self.memory.chat_memory.messages

        # RAG 체인 생성
        chain = self.get_rag_chain()

        # 대화 기록과 문서 컨텍스트를 포함하여 답변 생성
        answer = chain.invoke({
            "question": question, 
            "context": docs, 
            "history": conversation_history
        })

        # 대화 체인에 대화 추가
        self.conversation_chain.predict(input=question)

        return answer

# 메인 함수
def main():
    st.set_page_config(page_title="디지털경영전공 챗봇", layout="wide")

    st.title("🎓 디지털경영전공 챗봇")
    st.caption("여러분의 학과 관련 궁금증을 빠르게 해결해드립니다!")

    if st.button("📥 채팅 시작 !"):
        generate_faiss_index()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    left_col, mid_col, right_col = st.columns([1, 2.5, 1.2])

    with left_col:
        st.subheader("📚 사용 가이드")
        st.markdown("""
        - 채팅 시작! 버튼을 눌러주세요.<br>
        - 궁금한 점에 대해서 물어보세요 !.<br>
        - 추가 문의는 디지털경영전공 홈페이지나 학과 사무실(044-860-1560)로 문의해 주세요.
        """, unsafe_allow_html=True)

    with mid_col:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div style='background-color: #731034; padding: 10px; border-radius: 20px; margin-bottom: 10px; color: white; max-width: 70%; box-shadow: 0px 2px 5px rgba(0,0,0,0.1);'>
                💬 <b>질문:</b> {msg["content"]}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background-color: #f8f8f8; padding: 10px; border-radius: 20px; margin-bottom: 10px; margin-left: auto; box-shadow: 0px 2px 5px rgba(0,0,0,0.1); max-width: 70%;'>
                🤖 <b>답변:</b> {msg["content"]}
                </div>""", unsafe_allow_html=True)

        prompt = st.chat_input("궁금한 점을 입력해 주세요.")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            rag = RAGSystem(api_key)

            previous_qa = None
            if len(st.session_state.messages) >= 2:
                prev_question = st.session_state.messages[-2]["content"]
                prev_answer = st.session_state.messages[-1]["content"]
                previous_qa = (prev_question, prev_answer)

            with st.spinner("질문을 이해하는 중입니다. 잠시만 기다려주세요 😊"):
                answer = rag.process_question(prompt, previous_qa)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

    with right_col:
        st.subheader("📢 개발자에게 의견 보내기")
        feedback_input = st.text_area("챗봇에 대한 개선 의견이나 하고 싶은 말을 남겨주세요!")
        if st.button("피드백 제출"):
            if feedback_input.strip() != "":
                with open("feedback_log.csv", mode="a", encoding="utf-8-sig", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow([time.strftime('%Y-%m-%d %H:%M:%S'), feedback_input])
                st.success("소중한 의견 감사합니다!")
                st.rerun()
            else:
                st.warning("피드백 내용을 입력해 주세요.")

        st.subheader("📝 최근 질문 히스토리")
        for i, q in enumerate([m["content"] for m in st.session_state.messages if m["role"] == "user"][-5:], 1):
            st.markdown(f"{i}. {q}")

if __name__ == "__main__":
    main()
