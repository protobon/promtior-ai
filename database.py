from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from os import path
import requests
from bs4 import BeautifulSoup

load_dotenv()

# Define the db persistent directory
current_dir = path.dirname(path.abspath(__file__))
db_dir = path.join(current_dir, "db")
persistent_directory = path.join(db_dir, "promtior")


def fetch_page_content(url: str) -> str:
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from paragraphs and headers
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
    return "\n".join([element.get_text(strip=True) for element in text_elements])


def load_pdf(fp: str) -> Document:
    loader = PyPDFLoader(fp)
    return loader.load()[2]  # only page 2 has relevant information


def populate_database():
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
        documents.extend(docs)

    # append data from pdf file
    pdf_path = path.join(current_dir, "data", "promtior.pdf")
    if path.exists(pdf_path):
        print(f"\n--- Loading content from {pdf_path} ---")
        pdf_doc = load_pdf(pdf_path)
        documents.append(pdf_doc)

    print("\n------\n")
    print(f"\n--- Saving {len(documents)} documents to db ---")
    Chroma.from_documents(documents, embedding=OpenAIEmbeddings(), persist_directory=persistent_directory)


if __name__ == "__main__":
    populate_database()
