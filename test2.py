
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import os
from crewai_tools import BaseTool, DirectoryReadTool
from crewai_tools import DirectoryReadTool
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
import pickle

# llm = LLM(
#     model="ollama/mistral",
#     base_url="http://localhost:11434"
# )

llm = LLM(
    model="groq/llama-3.1-8b-instant",
    temperature=0.7,
    timeout=800 
)


class CsvRAGtool(BaseTool):
    name: str = "CSV Query Tool"
    description: str = "Analyzes CSV data and answers questions using natural language queries."

    def _run(self, query: str) -> str:
        try:
            llm = ChatGroq(temperature=0, model_name="gemma2-9b-it")

            agent = create_csv_agent(
                llm,
                "data.csv",
                verbose=True,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                handle_parsing_errors=True,
                allow_dangerous_code=True
            )

            return agent.run(query)
            
        except Exception as e:
            print(f"Error processing CSV query: {str(e)}")
            return None


docs_tool_a = DirectoryReadTool(directory='data.csv')
csv_rag = CsvRAGtool()
data_analysis_agent = Agent(
    role="Data Analysis Specialist",
    goal="Analyze the dataset to determine if data cleaning and feature engineering are required.",
    backstory="Expert in data cleaning, feature scaling, encoding",
    tools=[docs_tool_a, csv_rag],
    llm=llm,
    verbose=True
)

# Create Feature Engineering task
data_analysis_task = Task(
    description="""
    1. Load the dataset from 'data.csv' to begin the analysis process.
    
    2. Perform data cleaning checks to ensure the dataset's integrity:
       - Identify and report any null (missing) values in the dataset. Specify which columns contain null values and the number of such occurrences.
       - Detect and report any duplicate entries in the dataset. Provide the number of duplicate rows identified and in which columns they occur.

    3. Conduct a thorough feature engineering assessment to determine if modifications are required for improved model performance:
       - Examine the data types of each column to ensure they are appropriate for their intended use. Note any inconsistencies or mismatches in data types.
       - Analyze the range (minimum and maximum values) of numerical columns to assess the need for feature scaling. Provide insights into whether standardization or normalization is required based on the observed ranges.
    """,
    agent=data_analysis_agent,
    expected_output="""
    data cleaning: required or not
    feature engineering: required or not
    description: short description with numerical content
    """
)


class DataCleaner(BaseTool):
    name: str = "Data Preprocessor"
    description: str = "Preprocesses data by handling missing values, removing duplicates"

    def _run(self, file_path: str) -> str:
        try:
            # Load the data
            df = pd.read_csv(file_path)
            
            # Get initial info
            initial_shape = df.shape
            initial_missing = df.isnull().sum().sum()
            
            # Calculate the percentage of missing values
            missing_percentage = (initial_missing / (df.size)) * 100
            
            # Handle missing values
            if missing_percentage < 5:
                df = df.dropna()
            else:
                # Use SimpleImputer for numerical columns
                num_cols = df.select_dtypes(include=['number']).columns
                if not num_cols.empty:
                    num_imputer = SimpleImputer(strategy='mean')
                    df[num_cols] = num_imputer.fit_transform(df[num_cols])
                
                # Use SimpleImputer for categorical columns
                cat_cols = df.select_dtypes(include=['object']).columns
                if not cat_cols.empty:
                    cat_imputer = SimpleImputer(strategy='most_frequent')
                    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
            
            # Remove duplicate entries
            df = df.drop_duplicates()
            
            # Identify categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
            
            # Get final info
            final_shape = df.shape
            final_missing = df.isnull().sum().sum()
            
            # Save the processed data
            processed_file_path = os.path.join('data.csv')
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            df.to_csv(processed_file_path, index=False)
            
            return f"Cleaned data saved to {processed_file_path}"
        except Exception as e:
            return f"Error in preprocessing: {str(e)}"

class FeatureEngineering(BaseTool):
    name: str = "Feature Scaling and encoding Tool"
    description: str = "Scales numerical features and encodes categorical values"

    def _run(self, file_path: str, target: str, model:str) -> str:
        try:
            df = pd.read_csv(file_path)
            df_engineered = df.copy()
            
            # Encode categorical variables
            label_encoders = {}
            categorical_cols = df_engineered.select_dtypes(include=['object']).columns
            categorical_cols = [col for col in categorical_cols if col != target]  # Filter out the target column
            for col in categorical_cols:
                le = LabelEncoder()
                df_engineered[col] = le.fit_transform(df_engineered[col].astype(str))
                label_encoders[col] = le

            # Create artifacts directory if it doesn't exist
            os.makedirs('artifacts', exist_ok=True)
            
            # Save the label encoder
            encoder_filename = os.path.join('artifacts', 'label_encoder.pkl')
            with open(encoder_filename, 'wb') as file:
                pickle.dump(label_encoders, file)

            ## Check whether label encoding is necessory or not is model is classification
            dtype_target = df_engineered[target].dtype
            print(dtype_target)
            if dtype_target == "object" and model == "classification":
                print("Label encoding necessory")
                le_target = LabelEncoder()
                df_engineered[target] = le_target.fit_transform(df_engineered[target].astype(str))
                target_encoder_filename = os.path.join('artifacts', 'target_label_encoder.pkl')
                with open(target_encoder_filename, 'wb') as file:
                    pickle.dump(le_target, file)
            else:
                print("Not necessory")
            
            # Scale numerical features
            numerical_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
            numerical_cols = [col for col in numerical_cols if col != target] 
            if not numerical_cols.empty:
                scaler = StandardScaler()
                df_engineered[numerical_cols] = scaler.fit_transform(df_engineered[numerical_cols])
                
                # Save the scaler
                scaler_filename = os.path.join('artifacts', 'scaler.pkl')
                with open(scaler_filename, 'wb') as file:
                    pickle.dump(scaler, file)

            output_path = os.path.join('artifacts', 'engineered_features.csv')
            df_engineered.to_csv(output_path, index=False)
            
            return f"Feature engineering completed. File saved to {output_path}"
            
        except Exception as e:
            return f"Error in feature engineering: {str(e)}"


from crewai_tools import DirectoryReadTool

docs_tool_b= DirectoryReadTool(directory='data.csv')
csv_rag = CsvRAGtool()
data_cleaner_tool = DataCleaner()
feature_engineer_tool = FeatureEngineering()

data_preprocessing_agent = Agent(
    role="Data preprocessing Specialist",
    goal="Efficiently clean and prepare data for training ML model",
    backstory="Expert in data cleaning, feature scaling, encoding",
    tools=[docs_tool_b,data_cleaner_tool,feature_engineer_tool],
    llm=llm,
    verbose=True
)

# Create Feature Engineering task
data_preprocessing_task = Task(
    description="""
    Preprocess the dataset using the following steps:
    1. Use DataCleaner tool to clean the data if required
       - Handle missing values
       - Remove duplicates
    2. Use FeatureEngineering tool to transform the data if required
       Parameters to use:
       - file_path: 'data.csv'
       - target: {target}
       - model_type: {model}
    """,
    agent=data_preprocessing_agent,
    expected_output="""Data preprocessing completed"""
)

# Create and run the crew
crew = Crew(
    agents=[data_analysis_agent, data_preprocessing_agent],
    tasks=[data_analysis_task, data_preprocessing_task],
    process=Process.sequential
)

target = 'Price'
model = 'regression'
# Execute the pipeline
try:
    result = crew.kickoff(inputs={'target': target, 'model': model})
    print(result)
except Exception as e:
    print(f"Error executing pipeline: {str(e)}")
