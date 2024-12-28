# Must precede any llm module imports

# from langtrace_python_sdk import langtrace

# langtrace.init(api_key = 'a8f171dce8f1c150104082f5adbe1a67710f292d255ba94c5082347f372e04af')

from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from crewai_tools import BaseTool
import os
import pickle
# from crewai_tools import BaseTool

# llm = LLM(
#     model="ollama/mistral",
#     base_url="http://localhost:11434"
# )

llm = LLM(
    model="groq/llama-3.3-70b-specdec",
    temperature=0.7
)



class DataPreprocessor(BaseTool):
    name: str = "Data Preprocessor"
    description: str = "Preprocesses data by handling missing values, removing duplicates, and encoding categorical variables."

    def _run(self, file_path: str) -> str:
        # Load the data
        df = pd.read_csv(file_path)
        
        # Get initial info
        initial_shape = df.shape
        initial_missing = df.isnull().sum().sum()
        
        # Calculate the percentage of missing values
        missing_percentage = (initial_missing / df.size) * 100
        
        # Handle missing values
        if missing_percentage < 5:
            df = df.dropna()
        else:
            # Use SimpleImputer for numerical columns
            num_cols = df.select_dtypes(include=['number']).columns
            num_imputer = SimpleImputer(strategy='mean')
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            
            # Use SimpleImputer for categorical columns
            cat_cols = df.select_dtypes(include=['object']).columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
        
        # Remove duplicate entries
        df = df.drop_duplicates()
        
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # # Convert categorical variables to numerical
        # label_encoder = LabelEncoder()
        # for col in categorical_columns:
        #     df[col] = label_encoder.fit_transform(df[col])
        
        # Get final info
        final_shape = df.shape
        final_missing = df.isnull().sum().sum()
        
        # Save the processed data
        # processed_file_path = os.path.join('processed_data.csv')
        processed_file_path = os.path.join('processed_data', 'processed_data.csv')
        os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
        df.to_csv(processed_file_path, index=False)
        
        return f"""
        Data preprocessing completed:
        - Initial shape: {initial_shape}
        - Initial missing values: {initial_missing}
        - Final shape: {final_shape}
        - Final missing values: {final_missing}
        - Categorical variables encoded: {categorical_columns}
        - Duplicates removed

        """



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

            # Save the label encoder as a pickle file in the 'artifacts' folder
            encoder_filename = os.path.join('artifacts', 'label_encoder.pkl')
            with open(encoder_filename, 'wb') as file:
                pickle.dump(le, file)
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
            df_engineered[numerical_cols] = scaler.fit_transform(df_engineered[numerical_cols])

            # Save the scaler as a pickle file in the 'artifacts' folder
            scaler_filename = os.path.join('artifacts', 'scaler.pkl')
            with open(scaler_filename, 'wb') as file:
                pickle.dump(scaler, file)

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


###########################################################################################
# DATA PREPROCESSING AGENT AND TASK
###########################################################################################

from crewai_tools import DirectoryReadTool

docs_tool_a = DirectoryReadTool(directory='data.csv')
data_processing_tool = DataPreprocessor()
data_preprocessing_agent = Agent(
    role="Data Preprocessing Specialist",
    goal="Load, clean, and perform initial transformations on datasets",
    backstory="Expert in data cleaning and preprocessing using pandas, numpy, and sklearn libraries",
    llm=llm,
    tools=[docs_tool_a, data_processing_tool],
    verbose=True
    # allow_code_execution=True
)

data_preprocessing_task = Task(
  description="""
  Load the file, handle missing values (remove missing values if number of missing values is less than 5 percent of dataset else use imputer), remove duplicates, and convert categorical variables to numerical values to make the dataset model-ready.
  """,
  expected_output='Processed dataset saved successfully',
  agent=data_preprocessing_agent,
  )


###########################################################################################
# FEATURE ENGINEERING AGENT AND TASK
###########################################################################################



from crewai_tools import DirectoryReadTool
processed_file_path = os.path.join('processed_data', 'processed_data.csv')
docs_tool = DirectoryReadTool(directory=processed_file_path)
csv_rag = CsvRAGtool()
feture_eng = FeatureEngineeringTool()
feature_engineering_agent = Agent(
    role="Feature Engineering Specialist",
    goal="Your task is to analyze the different features of data and tell that, is feature engineering really required for data or not. If required then do feature engineering by scaling numeric features and encoding categorical features",
    backstory="Expert in feature scaling, encoding features",
    tools=[docs_tool, csv_rag, feture_eng],
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
    expected_output="short description about why need to do feature engineering",
    # human_input= True
)

# Create and run the crew
crew = Crew(
    agents=[data_preprocessing_agent,feature_engineering_agent],
    tasks=[data_preprocessing_task,feature_engineering_task],
    process=Process.sequential
)

result = crew.kickoff()
print(result)