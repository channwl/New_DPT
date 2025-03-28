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

# Anthropic API 키 로드 (Streamlit secrets 사용)
time.sleep(1)
api_key = st.secrets["anthropic"]["API_KEY"]

# PDF 인덱스 생성 스크립트와 다른 클래스들은 기존 코드 그대로 유지

class RAGSystem:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
        # Claude 3.5 Haiku 모델로 변경
        self.llm = ChatAnthropic(
            model="claude-3-5-haiku-20240307", 
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

        1. **대화 전체 맥락 고려**: 이전 대화 내용을 철저히 분석하고 연결합니다.
        2. **일관성 유지**: 이전 답변과 모순되지 않도록 주의합니다.
        3. 답변은 최대 4문장, 간결하고 명확하게 작성합니다.
        4. 중요 내용은 핵심만 요약해서 전달합니다.
        5. **상황별 대응**:
           - 반복 질문: 새로운 관점 또는 추가 정보 제공
           - 모호한 질문: 구체적 맥락 확인 후 답변
           - 연속 질문: 이전 대화 흐름 자연스럽게 이어가기

        대화 이력: {history}
        PDF 컨텍스트: {context}
        현재 질문: {question}

        응답 작성:
        """
        prompt = PromptTemplate.from_template(template)
        
        # Claude 3.5 Haiku 모델 사용
        model = ChatAnthropic(
            model="claude-3-5-haiku-20240307", 
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
