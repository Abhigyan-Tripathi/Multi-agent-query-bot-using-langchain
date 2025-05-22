import streamlit as st
import langchain_community
from dotenv import load_dotenv
import os

load_dotenv()

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph
import requests
from bs4 import BeautifulSoup

# Define Stategraph
class QueryState(BaseModel):
    query: str
    docs: list = Field(default_factory=list)        
    web_results: list = Field(default_factory=list)  
    answer: str = ""                                  

graph = StateGraph(QueryState)

# retriever node 
def retrieve_docs(state: QueryState) -> QueryState:
    results = vectordb.similarity_search(state.query, k=3)
    state.docs = [doc.page_content for doc in results]  
    return state
graph.add_node("retrieve", retrieve_docs)

# web search node
from googlesearch import search
def web_search_agent(state: QueryState) -> QueryState:
    urls = list(search(state.query, num_results=5))
    texts = []
    for url in urls:
        try:
            res = requests.get(url, timeout=5)
            soup = BeautifulSoup(res.text, 'html.parser')
            texts.append(soup.get_text())
        except Exception:
            continue
    state.web_results = texts
    return state
graph.add_node("web_search", web_search_agent)

# setting up the response llm
from huggingface_hub import login
login(token=hf_token)
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
)

chat = ChatHuggingFace(llm=llm, verbose=True)

# answer using rag
def answer_with_docs(state: QueryState) -> QueryState:
    context = "\n\n".join(state.docs)
    prompt = f"Use the following astronomy info to answer the question:\n{context}\n\nQ: {state.query}"
    state.answer = chat.invoke(prompt)
    return state
graph.add_node("answer_docs", answer_with_docs)

# answer using web
def answer_with_web(state: QueryState) -> QueryState:
    context = "\n\n".join(state.web_results)
    prompt = f"Based on this web content, answer:\n{context}\n\nQ: {state.query}"
    state.answer = chat.invoke(prompt)
    return state
graph.add_node("answer_web", answer_with_web)

# natural llm response (in case of no documents and no web result)
def fallback_llm(state: QueryState) -> QueryState:
    state.answer = chat.invoke(state.query)
    return state
graph.add_node("fallback", fallback_llm)

# adding conditional edge (retrieval->web_search)
def route_after_retrieval(state: QueryState) -> str:
    return "answer_docs" if state.docs else "web_search"
graph.add_conditional_edges("retrieve", route_after_retrieval,["answer_docs","web_search"])

# adding conditional edge (web_search->fallback_llm)
def route_after_web(state: QueryState) -> str:
    return "answer_web" if state.web_results else "fallback"
graph.add_conditional_edges("web_search", route_after_web, ["answer_web","fallback"])

# defining the entry and exit
graph.set_entry_point("retrieve")
graph.set_finish_point("answer_docs")
graph.set_finish_point("answer_web")
graph.set_finish_point("fallback")
app = graph.compile()

png = app.get_graph().draw_mermaid_png()
st.image(png)

#pdf parsing
import fitz
def pdf_parsing(uploaded_pdf):
    text=""
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

# html scrapping
def html_scrapping(url):
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, 'html.parser')
    content = soup.get_text()
    return content
# image text extraction
import easyocr
from PIL import Image
def extract_text_from_image(image_path):
    reader = easyocr.Reader(['en']) 
    results = reader.readtext(image_path.read())
    
    extracted_text = " ".join([text for _, text, _ in results])
    return extracted_text

# vector database creation
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectordb = Chroma(embedding_function=embed)

# main loop using streamlit
def main():
    st.title("Astronomy Chatbot")

    uploaded_pdf = st.file_uploader("Upload PDF", type="pdf")
    url = st.text_input("Enter a URL to scrape")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    query = st.text_input("Ask your question here:")

    if st.button("Send"):
        all_text = []

        if uploaded_pdf:
            text=pdf_parsing(uploaded_pdf)
            all_text.append(text)
        
        if url:
            text=html_scrapping(url)
            all_text.append(text)

        if uploaded_image:
            text=extract_text_from_image(uploaded_image)
            all_text.append(text)

        if all_text:
            combined_text=combined_text = "\n".join(all_text)
            docs = [Document(page_content=combined_text)]
            vectordb.add_documents(docs)
        
        state = QueryState(query=query)
        response = app.invoke(state.dict())
        st.write(response["answer"])

if __name__ == "__main__":
    main()