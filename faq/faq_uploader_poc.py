import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# 하드코딩 FAQ 업로드
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

faq_examples = [
    Document(
        page_content="상속은 어떤 절차로 진행되나요?", metadata={"source": "civil_faq"}
    ),
    Document(
        page_content="혼인은 어떻게 성립되나요?", metadata={"source": "civil_faq"}
    ),
    Document(
        page_content="입양의 조건은 무엇인가요?", metadata={"source": "civil_faq"}
    ),
]

# embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = PineconeVectorStore.from_documents(
    documents=faq_examples,
    embedding=embeddings,
    index_name=FAQ_INDEX_NAME,
)
print(f"{len(faq_examples)}개 FAQ 업로드 완료")
