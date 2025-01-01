# Must precede any llm module imports
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import tool
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_groq import ChatGroq
from crewai_tools import BaseTool, DirectoryReadTool
import os
import pickle

# Initialize LLM
llm = LLM(
    model="groq/gemma2-9b-it",
    temperature=0.7
)

# llm = LLM(
#     model="ollama/llama3.2",
#     base_url="http://localhost:11434"
# )

class DataPreprocessor(BaseTool):
    name: str = "Data Preprocessor"
    description: str = "Preprocesses data by handling missing values, removing duplicates, and encoding categorical variables."

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
            processed_file_path = os.path.join('artifacts', 'processed_data.csv')
            os.makedirs(os.path.dirname(processed_file_path), exist_ok=True)
            df.to_csv(processed_file_path, index=False)
            
            # return f"""
            # Data preprocessing completed:
            # - Initial shape: {initial_shape}
            # - Initial missing values: {initial_missing}
            # - Final shape: {final_shape}
            # - Final missing values: {final_missing}
            # - Categorical variables found: {categorical_columns}
            # - Duplicates removed
            # """
            return f"Cleaned data saved to {processed_file_path}"
        except Exception as e:
            return f"Error in preprocessing: {str(e)}"

class FeatureEngineeringTool(BaseTool):
    name: str = "Feature Scaling Tool"
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

class CsvRAGtool(BaseTool):
    name: str = "CSV Query Tool"
    description: str = "Analyzes CSV data and answers questions using natural language queries."

    def _run(self, query: str, file_path: str) -> str:
        try:
            llm = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
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
            return f"Error in CSV query: {str(e)}"

# Initialize paths and tools
input_path = os.path.join('artifacts', 'after_drop_col.csv')
processed_file_path = os.path.join('artifacts', 'processed_data.csv')

# Initialize tools
docs_tool_preprocessing = DirectoryReadTool(directory=input_path)
data_processing_tool = DataPreprocessor()
docs_tool_engineering = DirectoryReadTool(directory=processed_file_path)
csv_rag = CsvRAGtool()
feature_eng = FeatureEngineeringTool()

# Create agents
data_preprocessing_agent = Agent(
    role="Data Preprocessing Specialist",
    goal="Load, clean, and perform initial transformations on datasets",
    backstory="Expert in data cleaning and preprocessing using pandas, numpy, and sklearn libraries",
    llm=llm,
    tools=[docs_tool_preprocessing, data_processing_tool],
    verbose=True
)

feature_engineering_agent = Agent(
    role="Feature Engineering Specialist",
    goal="Analyze features and perform feature engineering if required",
    backstory="Expert in feature scaling and encoding",
    tools=[docs_tool_engineering, csv_rag, feature_eng],
    llm=llm,
    verbose=True
)

# Create tasks
data_preprocessing_task = Task(
    description="""
    1. Load the file 'artifacts/after_drop_col.csv'
    2. Handle missing values (remove if <5% missing, else use imputer)
    3. Remove duplicates
    Save the processed dataset.
    """,
    expected_output='Processed dataset saved successfully',
    agent=data_preprocessing_agent
)

feature_engineering_task = Task(
    description="""
    1. Load the cleaned data
    2. Analyze if feature engineering is required
    3. input variable from user target - {target}, model - {model}
    4. If required, perform feature scaling and encoding using tool
    5. Provide justification for decisions made and it should be in short
    """,
    agent=feature_engineering_agent,
    expected_output="Analysis and feature engineering report"
)

# Create and run the crew
crew = Crew(
    agents=[data_preprocessing_agent, feature_engineering_agent],
    tasks=[data_preprocessing_task, feature_engineering_task],
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