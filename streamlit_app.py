
import re
import streamlit as st
import sys


# Must precede any llm module imports

# from langtrace_python_sdk import langtrace

# langtrace.init(api_key = 'a8f171dce8f1c150104082f5adbe1a67710f292d255ba94c5082347f372e04af')

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from crewai_tools import BaseTool

# llm = LLM(
#     model="ollama/mistral",
#     base_url="http://localhost:11434"
# )

llm = LLM(
    model="groq/llama3-8b-8192",
    temperature=0.7
)


class FeatureEngineeringTool(BaseTool):
    name: str = "Feature Scaling Tool"
    description: str = "Scales numerical features to a standard range using techniques like normalization or standardization also encode categorical values"

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            df_engineered = df.copy()
            
            # Encode categorical variables
            le = LabelEncoder()
            categorical_cols = df_engineered.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                df_engineered[col] = le.fit_transform(df_engineered[col].astype(str))
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
            df_engineered[numerical_cols] = scaler.fit_transform(df_engineered[numerical_cols])

            output_path = 'engineered_features.csv'
            df_engineered.to_csv(output_path, index=False)
            
            return f"file saved to {output_path}"
            
        except Exception as e:
            print(f"Error engineering features: {str(e)}")
            return None


class CsvRAGtool(BaseTool):
    name: str = "CSV Query Tool"
    description: str = "A tool that analyzes CSV data and answers questions about its content using natural language queries."

    def _run(self, query: str,file_path: str) -> str:
        try:
            llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")

            # Add allow_dangerous_code=True to acknowledge the security implications
            agent = create_csv_agent(
                llm,
                file_path,
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            return agent.run(query)
            
        except Exception as e:
            print(f"Error processing CSV query: {str(e)}")
            return None



from crewai_tools import DirectoryReadTool

docs_tool_a = DirectoryReadTool(directory='data.csv')
csv_rag = CsvRAGtool()
feture_eng = FeatureEngineeringTool()
feature_engineering_agent = Agent(
    role="Feature Engineering Specialist",
    goal="Your task is to analyze the different features of data and tell that, is feature engineering really required for data or not. If required then do feature engineering by scaling numeric features and encoding categorical features",
    backstory="Expert in feature scaling, encoding features",
    tools=[docs_tool_a, csv_rag, feture_eng],
    llm=llm,
    verbose=True
)

# Create Feature Engineering task
feature_engineering_task = Task(
    description="""
    1. Load the preprocessed dataset
    2. Check if feature scaling really reqd or not
    3. If feature engineering required then do feature engineering
    4. provide a short description description of decision taken
    """,
    agent=feature_engineering_agent,
    expected_output="Final answer ",
    # human_input= True
)

# Create and run the crew
crew = Crew(
    agents=[feature_engineering_agent],
    tasks=[feature_engineering_task],
    process=Process.sequential
)

# result = crew.kickoff()
# print(result)
















###########################################################################################
# Print agent process to Streamlit app container                                          #
# This portion of the code is adapted from @AbubakrChan; thank you!                       #
# https://github.com/AbubakrChan/crewai-UI-business-product-launch/blob/main/main.py#L210 #
###########################################################################################
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # # Check if the data contains 'task' information
        # task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        # task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        # task_value = None
        # if task_match_object:
        #     task_value = task_match_object.group(1)
        # elif task_match_input:
        #     task_value = task_match_input.group(1).strip()

        # if task_value:
        #     st.toast(":robot_face: " + task_value)

        # if "City Selection Exper
        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []







with st.status("🤖 **Agents at work...**", state="running", expanded=True) as status:
        with st.container(height=500, border=False):
            sys.stdout = StreamToExpander(st)
            result = crew.kickoff()
            print(result)

st.markdown(result)