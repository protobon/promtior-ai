from os import path
from operator import itemgetter
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, format_document
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langserve import add_routes
from typing import List, Tuple
from langserve.pydantic_v1 import BaseModel, Field

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = path.dirname(path.abspath(__file__))
persistent_directory = path.join(current_dir, "db", "promtior")

if not path.exists(persistent_directory):
    raise FileNotFoundError("Persistent directory not found, initialize database by running database.py")

# Contextualize question based on chat history
CONDENSE_Q_TEMPLATE = """
Given a chat history and the latest user question 
which might reference context in the chat history, 
formulate a standalone question which can be understood 
without the chat history. Don't answer the question, just 
reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}
Follow Up Question: {question}
Standalone question:"""

CONDENSE_Q_PROMPT = PromptTemplate.from_template(CONDENSE_Q_TEMPLATE)

# Answer question prompt
QA_TEMPLATE = """
You are an assistant for question-answering tasks about a company called Promtior.ai.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say 'I don't know'.
Keep the answer concise and use three sentences maximum.

Context: {context}\n\n
Question: {question}
"""

# Create a prompt template for answering questions
QA_PROMPT = ChatPromptTemplate.from_template(QA_TEMPLATE)

# Create a prompt to convert retrieved documents to text
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")


def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return "\n\n".join(doc_strings)


def _format_chat_history(chat_history: List[Tuple]) -> str:
    """Format chat history into a string."""
    buffer = ""
    for dialogue_turn in chat_history:
        human = "Human: " + dialogue_turn[0]
        ai = "AI: " + dialogue_turn[1]
        buffer += "\n" + "\n".join([human, ai])
    return buffer


# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=OpenAIEmbeddings())

# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

# Create a ChatOpenAI model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

_inputs = RunnableMap(
    standalone_question=RunnablePassthrough.assign(
        chat_history=lambda x: _format_chat_history(x["chat_history"])
    )
    | CONDENSE_Q_PROMPT
    | llm
    | StrOutputParser(),
)

_context = {
    "context": itemgetter("standalone_question") | retriever | _combine_documents,
    "question": lambda x: x["standalone_question"],
}


# User input
class ChatHistory(BaseModel):
    """Chat history with the bot."""

    chat_history: List[Tuple[str, str]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "question"}},
    )
    question: str


conversational_qa_chain = (
    _inputs | _context | QA_PROMPT | llm | StrOutputParser()
).with_types(input_type=ChatHistory)

app = FastAPI(
    title="Promtior AI assistant",
    description="Chatbot that uses RAG to answer questions about the company Promtior.ai"
)


@app.get("/")
async def home():
    """ Redirect to chat playground """
    return RedirectResponse(url="/promtior/playground")

add_routes(
    app,
    conversational_qa_chain,
    path="/promtior",
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
