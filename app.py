import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationChain
from langchain.vectorstores import Pinecone
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain import LLMChain
from langchain.chains import RetrievalQA
import textwrap
import sys
import torch
import timeit
import datetime
import requests
import nltk
import pinecone

import os
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Website AI Chat Assistant - Open Source Version", layout="wide")
st.subheader("Welcome to Open Website AI Chat Assistant Life Enhancing with AI!")
st.write("Important notice: This Open Website AI Chat Assistant is offered for information and study purpose only and by no means for any other use. Any user should never interact with the AI Assistant in any way that is against any related promulgated regulations. The user is the only entity responsible for interactions taken between the user and the AI Chat Assistant.")
css_file = "main.css"
with open(css_file) as f:
    st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)  

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')
model_name = os.getenv('model_name')
repo_id_Starchat = os.getenv('repo_id_Starchat')
repo_id_Llama2 = os.getenv('repo_id_Llama2')

current_datetime_0= datetime.datetime.now()
print(f"Anything happens, this ST app will execute from top down. Waiting for actions... @ {current_datetime_0}")
print()
print("Beginning of App")
print(HUGGINGFACEHUB_API_TOKEN)
print(model_name)
print(repo_id_Starchat)
print(repo_id_Llama2)
print()

start_1 = timeit.default_timer()
print(f'向量模型加载开始 @ {start_1}')
embeddings=HuggingFaceEmbeddings(model_name=model_name)
end_1 = timeit.default_timer()
print(f'向量模型加载结束 @ {end_1}')
print(f'向量模型加载耗时： @ {end_1 - start_1}') 

pinecone_api_key= os.getenv('pinecone_api_key')
pinecone_environment= os.getenv('pinecone_environment')
index_name = os.getenv('index_name')   #设置secrets时，value填入的是myindex-allminilm-l6-v2-384
namespace_wt_metadata= os.getenv('namespace_wt_metadata')

print("pinecone_api_key: "+pinecone_api_key)
print("pinecone_environment: "+pinecone_environment)
#https://docs.pinecone.io/docs/python-client#list_indexes
print("index_name: "+index_name)
print(f"{index_name}: {index_name}")
print("namespace_wt_metadata: "+namespace_wt_metadata)
print()

print("llm初始化开始")
llm = HuggingFaceHub(repo_id=repo_id_Llama2,
                     model_kwargs={"min_length":512,
                                   "max_new_tokens":3072, "do_sample":True,
                                   "temperature":0.1,
                                   "top_k":50,
                                   "top_p":0.95, "eos_token_id":49155})

print("llm初始化结束")

prompt_template = """
You are a very helpful AI assistant who is an expert in intellectual property industry. Please ONLY use {context} to answer user's question with as many details as possible. If you don't know the answer, just say that you don't know. DON'T try to make up an answer.
Context: {context}
Question: {question}
Helpful AI Repsonse:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

print("chain初始化开始")
chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=PROMPT)
print("chain初始化结束")

#query = "What is the main business areas of HBSR?"
query = st.text_input("Enter your query here to AI Chat with HBSR website:")
print("输入有变化")
current_datetime_1= datetime.datetime.now()
print(f"输入有变化或程序重新启动或按钮被点击. @ {current_datetime_1}")
print()

if st.button('Process to AI Chat'):
    current_datetime= datetime.datetime.now()
    print(f"按钮被点击... @ {current_datetime}")
    print()
    if query != "" and not query.strip().isspace() and not query == "" and not query.strip() == "" and not query.isspace():
        print("Your query: "+query)
        print()
        with st.spinner('Fetching AI response...'):            
            start_2 = timeit.default_timer()
            print(f'Pinecone初始化开始 @ {start_2}')
            pinecone.init(
                api_key=pinecone_api_key,
                environment=pinecone_environment
            )
            index = pinecone.Index('{index_name}')
            end_2 = timeit.default_timer()
            print(f'Pinecone初始化结束 @ {end_2}')
            print(f'Pinecone初始化耗时： @ {end_2 - start_2}') 
            
            print(f"Pinecone index: {index}")
            #Pinecone index: <pinecone.index.Index object at 0x7f02394ebbb0>
            index_description = pinecone.describe_index(index_name)
            active_indexes = pinecone.list_indexes()
            print(f"index_description: {index_description}")
            print(f"active_indexes: {active_indexes}")
            print("Pinecone向量数据库参数信息打印完毕")
            print()
            
            start_3 = timeit.default_timer()
            print(f'Pinecone向量数据库装载开始 @ {start_3}')
            loaded_v_db_500_wt_metadata = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings, namespace=namespace_wt_metadata)
            end_3 = timeit.default_timer()
            print(f'Pinecone向量数据库装载结束 @ {end_3}')
            print(f'Pinecone向量数据库装载耗时： @ {end_3 - start_3}')         
            
            start_4 = timeit.default_timer()
            print(f'Pinecone向量数据库语义检索开始 @ {start_4}')
            #index = pinecone.Index('{index_name}')
            #需要在执行语义检索之前再次指定index
            docs_wt_metadata_retrieved = loaded_v_db_500_wt_metadata.similarity_search(query, namespace=namespace_wt_metadata, k=10)
            end_4 = timeit.default_timer()
            print(f'Pinecone向量数据库语义检索结束 @ {end_4}')
            print(f'Pinecone向量数据库语义检索耗时： @ {end_4 - start_4}') 
            
            unique_docs_wt_metadata_retrieved = []
            seen_page_content = set()
            for doc in docs_wt_metadata_retrieved:
                if doc.page_content not in seen_page_content:
                    unique_docs_wt_metadata_retrieved.append(doc)
                    seen_page_content.add(doc.page_content)
                    
            start_5 = timeit.default_timer()
            print(f'LLM QA开始 @ {start_5}')
            #可以提取Source的LLM QA形式
            ai_response_wt_metadata_1=chain({"input_documents": unique_docs_wt_metadata_retrieved, "question": query}, return_only_outputs=False)
            #也是字典形式，包括三个键：'input_documents'、'question'和'output_text'（默认就是False）
            #由于设置了return_only_outputs=False，因此这个ai_response中，会显示输入的input_documents、question和最终的输出结果output_text，可以从input_documents获取信息来源
            end_5 = timeit.default_timer()
            print(f'LLM QA结束 @ {end_5}')
            print(f'LLM QA耗时： @ {end_5 - start_5}') 
            
            output_text_wt_metadata_1=ai_response_wt_metadata_1['output_text'].replace('\n\n', '\n')
            print("AI Response:\n"+output_text_wt_metadata_1)
            st.write("AI Response:")
            st.write(output_text_wt_metadata_1)
            
            # 获取来源列表
            sources_wt_metadata_1 = [doc.metadata['source'] for doc in ai_response_wt_metadata_1['input_documents']]
            # 去重并打印来源
            unique_sources_wt_metadata_1 = list(set(sources_wt_metadata_1))
            print("\nSource:")  #打印Source这个字符之前先打印一个空行
            st.write("Sources referenced:")
            for source_wt_metadata_1 in unique_sources_wt_metadata_1:
                print(source_wt_metadata_1)
                st.write(source_wt_metadata_1)  
