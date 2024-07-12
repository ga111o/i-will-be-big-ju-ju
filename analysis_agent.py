import os
import streamlit as st
from crewai import Crew
from crewai.process import Process
from langchain_openai import ChatOpenAI
from tools import Agents, Tasks
from langchain_community.llms import Ollama

st.title("당신에게주식을")

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
