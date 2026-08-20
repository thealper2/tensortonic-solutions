import os
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

SOURCE_PATH = "document.txt"
text = open(SOURCE_PATH, encoding="utf-8").read()
documents = [
    Document(page_content=text, metadata={"source": SOURCE_PATH})
]

llm = ChatOpenAI(
    model="google.gemma-3-4b-it",
    base_url=os.environ["OPENAI_BASE_URL"],
)
embeddings = OpenAIEmbeddings(
    model="amazon.titan-embed-text-v2:0",
    base_url=os.environ["OPENAI_BASE_URL"],
    check_embedding_ctx_length=False,
)

question = "How long is customer data retained after workspace cancellation?"

print(f"Loaded {len(text):,} characters from {SOURCE_PATH}.")
print(f"Question: {question}")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=450,
    chunk_overlap=60,
    add_start_index=True,
)

if splitter is None:
    raise ValueError("Create the RecursiveCharacterTextSplitter before continuing.")

chunks = splitter.split_documents(documents)
for index, chunk in enumerate(chunks, start=1):
    chunk.metadata["chunk_id"] = f"chunk-{index}"

print(f"Created {len(chunks)} chunks.")
for chunk in chunks:
    preview = chunk.page_content[:120].replace("\n", " ")
    print(f"{chunk.metadata['chunk_id']}: {preview}...")
    
from uuid import uuid4
from langchain_chroma import Chroma

collection_name = f"northstar-support-{uuid4().hex[:8]}"

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name=collection_name,
)

if vector_store is None:
    raise ValueError("Create the Chroma vector store before continuing.")

print(f"Indexed {len(chunks)} chunks in {collection_name}.")

def retrieve(query: str, k: int = 3):
    hits = vector_store.similarity_search_with_score(query, k=k)
    if hits is None:
        raise ValueError("Call vector_store.similarity_search_with_score here.")
    return hits


hits = retrieve(question, k=3)

for rank, (document, distance) in enumerate(hits, start=1):
    chunk_id = document.metadata["chunk_id"]
    preview = document.page_content[:160].replace("\n", " ")
    print(f"{rank}. {chunk_id} | distance={distance:.4f}")
    print(f"   {preview}...")
    
def build_context(retrieved_hits) -> str:
    blocks = []
    for document, _distance in retrieved_hits:
        chunk_id = document.metadata["chunk_id"]
        blocks.append(f"[{chunk_id}]\n{document.page_content}")

    context = "\n\n".join(blocks)
    if context is None:
        raise ValueError("Join the context blocks before continuing.")
    return context


context = build_context(hits)
print(context)

SYSTEM_PROMPT = """You answer support questions using only the provided context.
Cite every factual claim with a source such as [chunk-2].
If the context does not contain the answer, say: I do not know based on the provided context.
"""


def answer_with_rag(query: str) -> dict:
    retrieved_hits = retrieve(query, k=3)
    context = build_context(retrieved_hits)
    messages = [
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {query}"),
    ]

    response = llm.invoke(messages)
    if response is None:
        raise ValueError("Invoke the chat model before returning the answer.")

    source_ids = [
        document.metadata["chunk_id"]
        for document, _distance in retrieved_hits
    ]
    return {"answer": response.content, "source_ids": source_ids}


result = answer_with_rag(question)
print(result["answer"])
print(f"Retrieved sources: {', '.join(result['source_ids'])}")

evaluation_cases = [
    {
        "question": "How long does a password reset link remain valid?",
        "expected_evidence": "15 minutes",
    },
    {
        "question": "How long is data retained after cancellation?",
        "expected_evidence": "90 days",
    },
    {
        "question": "Which file formats are available for data export?",
        "expected_evidence": "CSV and JSON",
    },
]

passed = 0
for case in evaluation_cases:
    retrieved_hits = retrieve(case["question"], k=3)
    evidence = " ".join(
        document.page_content for document, _distance in retrieved_hits
    )
    found = case["expected_evidence"].lower() in evidence.lower()
    passed += int(found)
    status = "PASS" if found else "MISS"
    print(f"{status}: {case['question']}")

print(f"Retrieval coverage: {passed}/{len(evaluation_cases)}")

unknown_question = "Does Northstar Cloud support on-premise deployment?"
unknown_result = answer_with_rag(unknown_question)
print(f"Out-of-scope question: {unknown_question}")
print(unknown_result["answer"])