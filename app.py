import streamlit as st 
from llama_index import SimpleDirectoryReader,ServiceContext
from llama_index.llms import OpenAI
from llama_index import VectorStoreIndex
from dotenv import load_dotenv
import os
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with st.sidebar:
    st.title("Chat with your Data")
    st.markdown("""
    ### About
                This is a demo of the [Streamlit](https://streamlit.io/) framework for building ML apps.
                   Using LLama Index ,Streamlit and OpenAI
                """)
        


def main():
    st.header("Chat with your Data")
    reader=SimpleDirectoryReader(input_dir='./data')
    docs=reader.load_data()
    service_context=ServiceContext.from_defaults(llm=OpenAI(model ='gpt-3.5-turbo-1106'),temperature=0.5,system_prompt="You are a Machine Learning Engineer and your job is to answer technical questions")
    index=VectorStoreIndex.from_documents(docs,service_context=service_context)
    query=st.text_input("Ask a question related to the data")
    if query:
        chat_engine=index.as_chat_engine(chat_mode='condense_question',verbose=True)
        response=chat_engine.chat(query)
        st.write(response.response)

if __name__ == "__main__":
    main()