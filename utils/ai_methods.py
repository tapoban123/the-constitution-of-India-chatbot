# __import__("pysqlite3")
# import sys

# sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from utils.secrets import LLM_API_KEY, GEMINI_MODELS
from utils.models import DocumentDataModel
from rich import print as rprint


def get_LLM():
    llm = ChatGoogleGenerativeAI(
        google_api_key=LLM_API_KEY.GEMINI_API_KEY.value,
        model=GEMINI_MODELS.GEMINI_MODEL.value,
    )

    return llm


def get_embedding_model():
    import asyncio

    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    model = GoogleGenerativeAIEmbeddings(
        google_api_key=LLM_API_KEY.GEMINI_API_KEY.value,
        model=GEMINI_MODELS.GEMINI_EMBEDDING_MODEL.value,
    )
    return model


def get_vector_store() -> Chroma:
    embedding_model = get_embedding_model()
    # loader = PyPDFDirectoryLoader("data")
    loader = PyPDFLoader("data/The_Constitution_of_India.pdf")
    # loader = PyPDFLoader("data/01-Page-1-29.pdf")
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
    )

    docs = text_splitter.split_documents(pages)

    print(len(docs))

    # for page in loader.lazy_load():
    #     pages.append(page)

    vector_store = Chroma.from_documents(docs, embedding_model)
    return vector_store


def search_vector_store(query: str, vector_store: Chroma) -> DocumentDataModel:
    docs = vector_store.similarity_search(query)
    docs_no_duplicates = []

    if len(docs) == len(docs_no_duplicates) == 1:
        docs_no_duplicates = docs
    else:
        for doc in docs:
            is_duplicate = False
            for dup in docs_no_duplicates:
                if doc.metadata["page_label"] == dup.metadata["page_label"]:
                    is_duplicate = True
                    break
            if not is_duplicate:
                docs_no_duplicates.append(doc)

    # rprint([doc.metadata["page_label"] for doc in docs])
    # rprint([doc.metadata["page_label"] for doc in docs_no_duplicates])

    data = "\n\n".join(text.page_content for text in docs_no_duplicates)
    page_nos = [page.metadata["page_label"] for page in docs_no_duplicates]
    # print(docs_no_duplicates, flush=True)

    return DocumentDataModel(data=data, page_nos=page_nos)


def get_ai_result(query: str, context: DocumentDataModel):
    llm = get_LLM()
    # print(context.page_nos)
    prompt = """
You are a helpful legal assistant.

Your task is to answer the user's query using only the provided context. If the answer is not available in the context, clearly mention that and still answer the query accurately on your own. In such cases, you must also mention that the answer was not found in the context.

Always return the relevant page numbers provided in {page_nos} as a separate section at the beginning of your response, so the user can refer to those pages in their physical book.

Your answer must be:
- Simple and easy to understand  
- Clearly explained  
- Helpful for remembering key points

Format your response like this:
-------
Relevant Page Numbers: {page_nos}

Your well-explained and easy-to-understand answer here
------

Context:  
{context}

Query:  
{query}
"""

    template = PromptTemplate(
        template=prompt,
        input_variables=["context", "query", "page_nos"],
    )

    # rprint(
    #     template.invoke(
    #         {
    #             "context": context.data,
    #             "query": query,
    #             "page_nos": context.page_nos,
    #         }
    #     )
    # )

    chain = template | llm | StrOutputParser()

    result = chain.invoke(
        {
            "context": context.data,
            "query": query,
            "page_nos": context.page_nos,
        }
    )
    return result
