import os
import openai
import streamlit as st
from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI


# Load API key from environment variables
load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")


# Streamlit sidebar configuration
with st.sidebar:
    st.title("Chat with your data")
    st.markdown('''
    ## ABOUT
    This app is an LLM-powered chatbot built using:
                
    - [Streamlit](http://streamlit.io) 🚐
    - [Llama Index](https://gpt-index.readthedocs.io/) 🐫  
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model 🌎
    ''')


def create_index(file_path: str, chunk_size: int, chunk_overlap: int, gpt_model: str = "gpt-4o", 
                 temperature: float = 0, system_prompt: str = "You are a machine learning expert. Let answer!"):
    """
    Create an VectorStoreIndex
    """
    reader = SimpleDirectoryReader(input_dir=file_path, recursive=True)
    docs = reader.load_data()
    llm = OpenAI(model=gpt_model, temperature=temperature, system_prompt=system_prompt)
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    Settings.num_output = 512
    Settings.context_window = 3900

    index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

    return index


def main():
    st.header("🌼🌼🌼 Chatbot for GPT4ALL 🌼🌼🌼")

    index = create_index(file_path='./data', 
        chunk_size=512, chunk_overlap=20, model="gpt-4o-mini", temperature=0, 
        system_prompt="""
    You are an teacher, working in LLM like GPT sector. Please answer concisely and briefly.
    Start your answer with 'Great! An intelligent question!'.
    End it with an '\n' (line break), then 'Any question else?'
    """)
    
    # Get user query and generate response
    query = st.text_input("Ask questions related to your data 👌")
    if query:
        chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
        response = chat_engine.chat(query)
        st.write(f"User 🤵: {query}\n")
        st.write(f"Assistant 😒: {response.response}")


if __name__ == "__main__":
    main()


# how much cost to gpu?
# how much spend on OpenAI?
# what model is gpt4all based on?
# import os
# import streamlit as st
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM
# from transformers import AutoModelForCausalLM, AutoTokenizer

# # Streamlit sidebar configuration
# with st.sidebar:
#     st.title("Chat with your data")
#     st.markdown(''' 
#     ## ABOUT 
#     This app is an LLM-powered chatbot built using:     
#     - [Streamlit](http://streamlit.io) 
#     - [Llama Index](https://gpt-index.readthedocs.io/)        
#     - [Hugging Face](https://huggingface.co/models) LLM Model 
#     ''')

# def main():
#     st.header("Chatbot for Your Data")

#     # Load documents and initialize LLM-based index
#     reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
#     docs = reader.load_data()

#     # Load lightweight models
#     model_name = "gpt2"  
#     model = AutoModelForCausalLM.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name)

#     # Initialize LLM and embedding model
#     llm = HuggingFaceLLM(model=model, tokenizer=tokenizer)
#     embed_model = HuggingFaceEmbedding(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"  # Lightweight embedding model
#     )
    
#     # Set up settings
#     Settings.llm = llm
#     Settings.embed_model = embed_model
#     Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
#     Settings.num_output = 512
#     Settings.context_window = 3900
    
#     # Create vector store index
#     index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

#     # Get user query and generate response
#     query = st.text_input("Ask questions related to your data")
#     if query:
#         chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
#         response = chat_engine.chat(query)
#         st.write(f"User: {query}\n")
#         st.write(f"Assistant: {response.response}")

# if __name__ == "__main__":
#     main()

# import os
# import requests
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from llama_index.llms.huggingface import HuggingFaceLLM

# # Hàm lấy phản hồi từ API Hugging Face
# def query_huggingface_api(model_name, query):
#     api_url = f"https://api-inference.huggingface.co/models/{model_name}"
#     headers = {"Authorization": f"Bearer ???"}
#     payload = {"inputs": query}
#     response = requests.post(api_url, headers=headers, json=payload)
#     return response.json()

# def main():
#     # Load documents and initialize LLM-based index
#     reader = SimpleDirectoryReader(input_dir='./data', recursive=True)
#     docs = reader.load_data()

#     # Model names
#     llm_model_name = "EleutherAI/gpt-neo-2.7B"
#     embed_model_name = "BAAI/bge-small-en"  # Lightweight embedding model

#     # Initialize LLM and embedding model
#     llm = HuggingFaceLLM(
#     context_window=4096,
#     max_new_tokens=512,
#     generate_kwargs={"temperature": 0.1, "do_sample": False},tokenizer_name=llm_model_name, model_name=llm_model_name)
#     embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

#     # Set up settings
#     # Settings.llm = llm
#     Settings.embed_model = embed_model
#     Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
#     Settings.num_output = 512
#     Settings.context_window = 2000

#     # Create vector store index
#     index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

#     # Get user query and generate response
#     query = "How much cost to train gpu"
#     if query:
#         ## Tạo QueryEngine từ index
#         query_engine = index.as_query_engine(llm=llm)

#         # Tìm kiếm tài liệu liên quan
#         relevant_docs = query_engine.query(query)  # Lấy các tài liệu liên quan

#         # Kết hợp tài liệu tìm thấy thành một chuỗi
#         context = "\n".join([doc['text'] for doc in relevant_docs])

#         # Gọi API để sinh câu trả lời từ nội dung đã tìm thấy
#         answer_query = f"{context}\n\nAnswer the question: {query}"
#         print(answer_query)
#         response = query_huggingface_api(llm_model_name, answer_query)

#         if response and isinstance(response, list):
#             assistant_response = response[0]['generated_text']  # Lấy phản hồi từ mô hình
#             print(f"Assistant: {assistant_response}")
#         else:
#             print("Assistant: Sorry, I don't have an answer for that.")

# if __name__ == "__main__":
#     main()
