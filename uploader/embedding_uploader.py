import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

#  환경 변수 불러오기 (.env에서 키 관리 중인 경우)
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX_NAME = "civil-law-index"

pc = Pinecone(api_key=PINECONE_API_KEY)
# 새 인덱스 생성: OpenAI의 text-embedding-3-large는 1536 차원
# * Automatically create when no index exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(name=PINECONE_INDEX_NAME, dimension=1536, metric="cosine")

if PINECONE_INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    print(f"'{PINECONE_INDEX_NAME}' 인덱스가 생성되었습니다.")
else:
    print(f"'{PINECONE_INDEX_NAME}' 인덱스가 이미 존재합니다.")

index = pc.Index(PINECONE_INDEX_NAME)


#  마크다운 파일 로딩 및 조문 분리
def load_markdown_articles(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_articles = content.split("## ")
    articles = []
    for article in raw_articles[1:]:  # 첫 번째는 빈 항목
        lines = article.strip().split("\n")
        title = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        doc = Document(page_content=body, metadata={"title": title})
        articles.append(doc)
    return articles


#  Pinecone VectorStore에 저장
def upload_to_pinecone(docs):
    # embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"총 {len(docs)}개 조문 업로드 완료")


if __name__ == "__main__":
    file_path = "civil_law_articles.md"
    articles = load_markdown_articles(file_path)
    upload_to_pinecone(articles)
