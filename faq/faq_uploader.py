import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
FAQ_INDEX_NAME = "civil-law-faq"

pc = Pinecone(api_key=PINECONE_API_KEY)

if FAQ_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=FAQ_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"'{FAQ_INDEX_NAME}' 인덱스가 생성되었습니다.")
else:
    print(f"'{FAQ_INDEX_NAME}' 인덱스가 이미 존재합니다.")


# Markdown 파일 기반 FAQ 업로드
def load_faq_documents(faq_md_path: str) -> list[Document]:
    """
    FAQ 마크다운 파일을 불러와서 Document 리스트로 변환
    Markdown 내부 형식 예시:
    ## Q: 질문 텍스트
    A: 답변 텍스트
    ---
    여러 질문-답변이 있을 경우 반복
    """
    docs = []
    with open(faq_md_path, encoding="utf-8") as f:
        content = f.read()
    entries = content.split("## Q:")[1:]
    for entry in entries:
        lines = entry.strip().splitlines()
        question = lines[0].strip()
        answer = "\n".join(lines[2:]).strip()  # assume "A:" 다음 줄부터 답변
        doc = Document(page_content=answer, metadata={"question": question})
        docs.append(doc)
    return docs


def upload_faq():
    docs = load_faq_documents("civil_law_faq.md")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vector_store = PineconeVectorStore(
        index_name=FAQ_INDEX_NAME,
        embedding=embeddings,
    )
    vector_store.add_documents(docs)
    print(f"FAQ {len(docs)}건 업로드 완료 ({FAQ_INDEX_NAME})")


if __name__ == "__main__":
    upload_faq()
