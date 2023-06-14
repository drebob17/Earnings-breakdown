#streamlit for UI/app interface
import streamlit as st

#Import dependancies
from langchain.llms import GPT4All
from langchain import PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.base import Embeddings
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from gpt4all import GPT4All
from langchain.llms import HuggingFaceHub

#Import PDF document loader
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader


#Import chroma as the vector store
from langchain.vectorstores import Chroma
# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

#PATH = 'C:/Users/andre/AppData/Local/nomic.ai/GPT4All/ggml-gpt4all-j-v1.3-groovy.bin'   #path to weights


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text 

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    #llm = GPT4All("ggml-gpt4all-j-v1.3-groovy")
    llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.3, "max_length": 512},
    huggingfacehub_api_token="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"
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)




def main():
    load_dotenv()
    st.set_page_config(page_title="Quick Earnings Breakdown",
                       page_icon=":💰:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Earnings 💰")
    user_question = st.text_input("Ask a question about the report:")
    if user_question:
        handle_userinput(user_question)


    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)

                #get text chunks
                text_chunks = get_text_chunks(raw_text)

                #create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)




if __name__ == '__main__':
    main()


