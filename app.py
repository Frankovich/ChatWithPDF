import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import os

os.environ["OPENAI_API_KEY"] = "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"


def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write("Generating Profile:")
            #st.write(
            #    user_template.replace("{{MSG}}", message.content),
            #    unsafe_allow_html=True,
            #)
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )

def main():
    st.set_page_config(page_title="Email to Profile (Alpha) :apple:", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Email to Profile (Alpha) :apple:")
    if st.button("Predict Profile"):
        user_question = "Using the emails, create a profile for the person who is receiving the emails and keep the" \
                        " format neat (add a new line after each category) and only output that. " \
                        "If there is not enough information to fill out a field, try to give a possible guess and" \
                        " make sure to write \"Possible\" before it" \
                        " and format it this way:\n"\
                        "Name:\n Hobbies:\n Possible Gender:\n Connections:\n Education:\n Possible Criminal Involement:\n" \
                        "Where is Virginia Tech"
        #user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        txt_docs = st.file_uploader(
            "Upload your TXT files here and click on 'Add Data'", accept_multiple_files=True, type=["txt"]
        )
        if st.button("Add Data"):
            with st.spinner("Adding Data..."):
                # get txt text
                raw_text = get_txt_text(txt_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()            
