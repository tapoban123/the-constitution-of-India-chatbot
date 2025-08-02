from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from utils.secrets import LLM_API_KEY, GEMINI_MODELS
from utils.models import DocumentDataModel


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
    loader = PyPDFDirectoryLoader("data")
    pages = loader.load()
    # for page in loader.lazy_load():
    #     pages.append(page)

    vector_store = Chroma.from_documents(pages, embedding_model)
    return vector_store


def search_vector_store(query: str, vector_store: Chroma) -> DocumentDataModel:
    docs = vector_store.similarity_search(query)

    data = "\n\n".join(text.page_content for text in docs)
    for doc in docs:
        print(doc.metadata)
    page_nos = [page.metadata["page"] for page in docs]

    return DocumentDataModel(data=data, page_nos=page_nos)


def get_ai_result(query: str, context: DocumentDataModel):
    llm = get_LLM()
    prompt = """You are a helpful Legal assistant.\nAnswer queries only from the given context.
    If answer not provided in context, mention that the answer does not exist in context and answer correctly by yourself with the page numbers.
    Also, return the page numbers of the context as provided. Ignore in case of duplicate information or page numbers.
    
    Context: 
    {context}
    
    Query:
    {query}
    
    Page numbers:
    {page_nos}
    """

    template = PromptTemplate(
        template=prompt,
        input_variables=["context", "query", "page_nos"],
    )

    chain = template | llm | StrOutputParser()

    result = chain.invoke(
        {
            "context": context.data,
            "query": query,
            "page_nos": context.page_nos,
        }
    )
    return result
