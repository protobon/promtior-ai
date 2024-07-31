from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from os import path
import requests
from bs4 import BeautifulSoup

load_dotenv()


def fetch_page_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from paragraphs and headers
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return "\n".join([element.get_text(strip=True) for element in text_elements])


# Define the db persistent directory
current_dir = path.dirname(path.abspath(__file__))
db_dir = path.join(current_dir, "db")
persistent_directory = path.join(db_dir, "promtior")

if not path.exists(persistent_directory):
    # Urls to scrape relevant data from
    urls = [
        "https://www.promtior.ai/",
        "https://www.promtior.ai/service",
        "https://www.promtior.ai/use-cases",
        "https://www.promtior.ai/contacto",
    ]

    # Split the scraped content into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for url in urls:
        print(f"\n--- Fetching content from {url} ---")
        content = fetch_page_content(url)
        docs = text_splitter.create_documents([content])
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(docs)}")
        print(f"Sample chunk:\n{docs[0].page_content}\n")
        documents.extend(docs)
    print("\n------\n")
    print(f"\n--- Saving {len(documents)} documents to db ---")
    db = Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory=persistent_directory)
else:
    print("\n--- Persistent Directory Found ---")
    db = Chroma(persist_directory=persistent_directory, embedding_function=OpenAIEmbeddings())


# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)

query = "What is promtio.ai's contact number?"

results = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(results):
    print(f"--- Document {i} ---\n{doc.page_content}\n")
