import os
import streamlit as st
from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI
from tools import Agents, Tasks
from langchain_community.llms import Ollama

st.set_page_config(
    page_title="나도이제만19세주식을할수있는",
    page_icon="⭐",
)

st.title("나이가되었기에`대주주`가되기위하여하나씩")

company = st.text_input("Enter the company name:", "")

model_options = ["gpt-4o", "gpt-3.5-turbo", "llama3:8b", "llama3:70b"]
selected_model = st.selectbox("Select the model:", model_options)

if st.button("Run Analysis"):
    os.environ["OPENAI_MODEL_NAME"] = selected_model

    agents = Agents()
    tasks = Tasks()

    researcher = agents.researcher()
    technical_analyst = agents.technical_analyst()
    financial_analyst = agents.financial_analyst()
    hedge_fund_manager = agents.hedge_fund_manager()

    research_task = tasks.research(researcher)
    technical_task = tasks.technical_analysis(technical_analyst)
    financial_task = tasks.financial_analysis(financial_analyst)
    recommend_task = tasks.investment_recommendation(
        hedge_fund_manager,
        [
            research_task,
            technical_task,
            financial_task,
        ],
    )

    if selected_model in ["llama3:8b", "llama3:70b"]:
        manager_llm = Ollama(model=selected_model)
    else:
        manager_llm = ChatOpenAI(model=selected_model)

    crew = Crew(
        agents=[
            researcher,
            technical_analyst,
            financial_analyst,
            hedge_fund_manager,
        ],
        tasks=[
            research_task,
            technical_task,
            financial_task,
            recommend_task,
        ],
        verbose=2,
        process=Process.hierarchical,
        manager_llm=manager_llm,
        memory=True,
    )

    result = crew.kickoff(
        inputs=dict(
            company=company,
        ),
    )
    
    st.write("Result:", result)

# Markdown 파일 읽기 섹션
st.sidebar.title("Markdown 파일 보기")

def list_files_in_directory(directory):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.endswith(".md"):
                files.append(os.path.join(root, filename))
    return files

def read_markdown_file(markdown_file):
    with open(markdown_file, 'r', encoding='utf-8') as file:
        return file.read()

directory_path = './output'
markdown_files = list_files_in_directory(directory_path)

option = st.sidebar.selectbox('Select a markdown file', markdown_files)

if option:
    markdown_content = read_markdown_file(option)
    st.markdown(markdown_content)
