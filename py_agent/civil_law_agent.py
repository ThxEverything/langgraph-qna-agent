import os
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from typing import List
from pydantic import BaseModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, MessagesState
from datetime import datetime
import logging

# ============================
# 환경 설정
# ============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4.1", temperature=0)

# Pinecone retriever 설정
small_llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
# embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

# ============================
# Pinecone 설정 (민법 조문)
# ============================
vector_store = PineconeVectorStore(
    index_name="civil-law-index",
    embedding=embeddings,
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ============================
# 로그 설정 (파일 + 콘솔)
# ============================
logging.basicConfig(
    filename="agent_history.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def log_info(msg: str):
    logging.info(msg)
    print(msg)  # 콘솔에도 보여주기


# ============================
# 데이터 모델
# ============================
class QuestionCategory(BaseModel):
    category: str  # 정의 / 요건 / 절차 / 비교 / 일반질문 등


class FAQCheck(BaseModel):
    is_in_faq: bool
    context: List[Document]


def get_last_human_message(messages: list) -> HumanMessage:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg
    # raise ValueError("No HumanMessage found in messages.")
    logging.warning("No HumanMessage found, returning default")
    return HumanMessage(content="No HumanMessage found in messages.")


# ============================
# Tool 1: 질문 분류
# ============================
@tool
def classify_question(messages: list) -> dict:
    """Classify the user's question into one of the following types.
    Type Examples: Definition, Requirements (Conditions), Procedure, Comparison, Exception, General Question
    """
    # messages에서 가장 마지막 HumanMessage의 content만 추출
    log_info(f"[DEBUG] messages = {messages}")
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[classify_question] 입력: {question}")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert Civil Act Question Classifier. Classify the question into a single type.",
            ),
            (
                "user",
                "Question: {question}\nRespond only with the type. (Definition/Requirements/Procedure/Comparison/Exception/General)",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})

    log_info(f"[classify_question] 결과: {result.strip()}")
    # return QuestionCategory(category=result.strip())
    return {"category": result.strip()}


# ============================
# Tool 2: 문서 선택 (필요 시 확장)
# ============================
@tool
def select_document(messages: list) -> dict:
    """Select the document most relevant to the question.
    The current document is 'civil_law' only.
    """
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[select_document] '{question}' → civil_law 선택")
    # 나중에 "형법", "근로기준법", "판례" 추가 가능
    # return "civil_law"
    return {"document_name": "civil_law"}


# ============================
# Tool 3: FAQ 여부 체크
# ver 1: Simple condition
# accuracy: low | speed: fast
# ============================
@tool
def check_faq_ver1(messages: list) -> dict:
    """Simple FAQ detection:
    - No LLM matching logic (fast)
    - If context exists, LLM formats the FAQ answer more clearly
    """  # tool description
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[check_faq] FAQ 여부 확인 (simple): {question}")
    # 필요시 Pinecone에 FAQ 인덱스 추가 가능
    faq_vs = PineconeVectorStore(index_name="civil-law-faq", embedding=embeddings)
    faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 3})
    context = faq_retriever.invoke(question)

    is_in = len(context) > 0
    log_info(f"[check_faq] FAQ 여부: {is_in}")
    # return FAQCheck(is_in_faq=is_in, context=context)
    return {"is_in_faq": is_in, "context": context}


# ============================
# Tool 3: FAQ 여부 체크
# ver 2: Prompt-based
# accuracy: high | speed: slow
# ============================
@tool
def check_faq_ver2(messages: list) -> dict:
    """LLM-Based Precision Judgment Method.
    The LLM determines (Yes/No) whether the context retrieved from the FAQ index is actually relevant to an FAQ.
    """
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[check_faq] FAQ 여부 점검 (LLM base): {question}")

    # FAQ retriever
    faq_vs = PineconeVectorStore(index_name="civil-law-faq", embedding=embeddings)
    faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 3})
    context = faq_retriever.invoke(question)

    # FAQ가 없으면 바로 반환
    if len(context) == 0:
        log_info("[check_faq] FAQ 검색 결과 없음 → FAQ 아님")
        # return FAQCheck(is_in_faq=False, context=[])
        return {"is_in_faq": False, "context": []}

    # -------------------------------
    # FAQ가 존재하면 → LLM이 답변을 자연스럽게 재구성
    # (Yes/No 판단이 아니라 답변 포맷팅)
    # -------------------------------
    faq_format_prompt = ChatPromptTemplate.from_template(
        """ You are an assistant who rewrites FAQ answers clearly and concisely.
Rewrite the FAQ answer below so that it directly answers the user's question.

[User Question]
{question}

[FAQ Raw Answer]
{answer}

Rewrite professionally in Korean.
"""
    )

    chain = faq_format_prompt | llm | StrOutputParser()

    formatted_answer = chain.invoke(
        {"question": question, "answer": context[0].page_content}
    )

    # LLM이 재작성한 답변을 Document 형태로 감싸서 반환
    formatted_doc = Document(
        page_content=formatted_answer, metadata=context[0].metadata
    )

    log_info("[check_faq] FAQ 존재 → LLM 포맷팅 후 반환")

    # return FAQCheck(is_in_faq=True, context=[formatted_doc])
    return {"is_in_faq": True, "context": [formatted_doc]}


# ============================
# Tool 3: FAQ 여부 체크
# ver 3: Hybrid
# accuracy: middle ~ high | speed: middle
# ============================
@tool
def check_faq_ver3(messages: list) -> dict:
    """Hybrid Approach:
    - The LLM judgment is executed only when the context is 'sufficiently meaningful' (or 'highly relevant').
    - Otherwise, it is simply determined not to be an FAQ.
    """
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[check_faq] FAQ 여부 점검 (hybrid): {question}")

    # FAQ retriever
    faq_vs = PineconeVectorStore(index_name="civil-law-faq", embedding=embeddings)
    faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 3})
    context = faq_retriever.invoke(question)

    # 0개면 FAQ가 아님
    if len(context) == 0:
        log_info("[check_faq] FAQ 검색 결과 없음 → FAQ 아님")
        # return FAQCheck(is_in_faq=False, context=[])
        return {"is_in_faq": False, "context": []}

    # -------------------------------
    # 하이브리드 기준:
    # context 길이가 너무 짧으면 LLM 판단 건너뜀
    # -------------------------------
    first_content = context[0].page_content.strip()
    if len(first_content) < 50:
        log_info("[check_faq] context가 너무 짧음 → FAQ 아님으로 처리")
        # return FAQCheck(is_in_faq=False, context=[])
        return {"is_in_faq": False, "context": []}

    # context가 충분히 길다면 LLM이 최종 판단
    check_faq_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Determine whether the user's question matches the FAQ context. "
                "Answer only 'Yes' or 'No'.",
            ),
            (
                "user",
                "Question: {question}\nFAQ Context: {context}\n\nIs this a matching FAQ entry?",
            ),
        ]
    )
    chain = check_faq_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question, "context": context})

    is_match = result.strip().lower() == "yes"
    log_info(f"[check_faq] LLM 판단 결과: {result.strip()}")

    # return FAQCheck(is_in_faq=is_match, context=context if is_match else [])
    return {"is_in_faq": is_match, "context": context if is_match else []}


# ============================
# Tool 4: FAQ 직접 답변
# ============================
@tool
def answer_via_faq(messages: list) -> dict:
    """FAQ 검색 후, 답변을 바로 반환."""
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[answer_via_faq] FAQ 답변 사용: {question}")

    # FAQ retriever
    faq_vs = PineconeVectorStore(index_name="civil-law-faq", embedding=embeddings)
    faq_retriever = faq_vs.as_retriever(search_kwargs={"k": 3})
    docs = faq_retriever.invoke(question)
    if not docs:
        return "죄송합니다. 관련 FAQ가 없습니다."
    # 간단히 첫 문서의 답변 출력 (혹은 요약)
    # return docs[0].page_content
    return {"answer": docs[0].page_content}


# ============================
# Tool 5: 민법 조문 검색
# ============================
@tool
def retrieve_law_context(messages: list, document_name: str) -> list[Document]:
    """Search for provisions related to the question
    within the selected document (document_name)."""
    last_message = get_last_human_message(messages)
    question = last_message.content.strip()
    log_info(f"[retrieve_law_context] question={question}, document={document_name}")
    # 확장: document_name 따라 다른 retriever 사용 가능
    context = retriever.invoke(question)
    log_info(f"[retrieve_law_context] retrieved_docs={len(context)}")
    return {"context": context}


# ============================
# 조문 기반 답변 생성 QA 체인
# ============================
qa_prompt = ChatPromptTemplate.from_template(
    """ You are an expert consultation assistant specializing in the Civil Act.
Below are the provisions of the Civil Act. Use them as the basis to accurately explain the answer to the question.
Always include the source in the format: "According to Article X of the Civil Act...".

[provision]
{context}

[question]
{question}
"""
)

qa_chain = qa_prompt | llm | StrOutputParser()

# ============================
# Agent 생성
# ============================
tools = [
    classify_question,
    select_document,
    check_faq_ver3,
    retrieve_law_context,
    answer_via_faq,
]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="Use the tools to produce accurate Civil Act answers.",
)

# ============================
# LangGraph 흐름 구성
# ============================
graph = StateGraph(MessagesState)

# 1) 노드 추가
graph.add_node("classify", classify_question)
graph.add_node("select_doc", select_document)
graph.add_node("faq_check", check_faq_ver3)
graph.add_node("faq_answer", answer_via_faq)
graph.add_node("law_agent", agent)

# graph.add_agent("law_agent", agent)
# graph.set_entry_point("law_agent")

# classify -> law_agent 로 연결
graph.set_entry_point("classify")

graph.add_edge("classify", "select_doc")
graph.add_edge("select_doc", "faq_check")


# 3) 조건 분기
def faq_condition(state):
    """FAQ 여부에 따라 경로 분기"""
    messages = state["messages"]
    last = messages[-1]

    if "is_in_faq" in last.additional_kwargs:
        return "faq" if last.additional_kwargs["is_in_faq"] else "not_faq"

    return "not_faq"


graph.add_conditional_edges(
    "faq_check",
    faq_condition,
    {
        "faq": "faq_answer",
        "not_faq": "law_agent",
    },
)

law_agent_executor = graph.compile()

# ============================
# Test
# ============================
if __name__ == "__main__":
    query = "상속 순위는 어떻게 되나요?"

    for chunk in law_agent_executor.stream(
        {"messages": [HumanMessage(content=query)]}, stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
