# from crewai import Agent, Task, Crew, Process, LLM
# import pandas as pd
# import numpy as np
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from dotenv import load_dotenv
# import os

# load_dotenv()


# # Initialize LLM
# llm = LLM(
#     model="groq/llama-3.3-70b-versatile",
#     temperature=0.7
# )







# def load_csv(file_path):
#     """Load CSV file and perform initial validation"""
#     try:
#         df = pd.read_csv(file_path)
        
#         # Basic validation
#         print(f"Dataset Shape: {df.shape}")
#         print("\nData Types:\n", df.dtypes)
#         print("\nMissing Values:\n", df.isnull().sum())
        
#         return df
        
#     except Exception as e:
#         print(f"Error loading file: {str(e)}")
#         return None

# def clean_data(df):
#     """Clean data and handle missing values"""
#     try:
#         # Create a copy to avoid modifying original data
#         df_clean = df.copy()
        
#         # Handle missing numerical values
#         numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
#         if len(numerical_cols) > 0:
#             num_imputer = SimpleImputer(strategy='mean')
#             df_clean[numerical_cols] = num_imputer.fit_transform(df_clean[numerical_cols])
        
#         # Handle missing categorical values
#         categorical_cols = df_clean.select_dtypes(include=['object']).columns
#         if len(categorical_cols) > 0:
#             cat_imputer = SimpleImputer(strategy='most_frequent')
#             df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
        
#         # Remove duplicates
#         df_clean = df_clean.drop_duplicates()
        
#         # Remove columns with high percentage of missing values
#         threshold = 0.7
#         df_clean = df_clean.dropna(axis=1, thresh=int(threshold * len(df_clean)))
        
#         return df_clean
        
#     except Exception as e:
#         print(f"Error cleaning data: {str(e)}")
#         return None

# def engineer_features(df):
#     """Perform feature engineering and prepare for ML"""
#     try:
#         df_engineered = df.copy()
        
#         # Encode categorical variables
#         le = LabelEncoder()
#         categorical_cols = df_engineered.select_dtypes(include=['object']).columns
#         for col in categorical_cols:
#             df_engineered[col] = le.fit_transform(df_engineered[col].astype(str))
        
#         # Scale numerical features
#         scaler = StandardScaler()
#         numerical_cols = df_engineered.select_dtypes(include=['int64', 'float64']).columns
#         df_engineered[numerical_cols] = scaler.fit_transform(df_engineered[numerical_cols])
        
#         return df_engineered
        
#     except Exception as e:
#         print(f"Error engineering features: {str(e)}")
#         return None

# # def create_agents():
#     """Create and return the agents for data processing"""
# data_loader = Agent(
#     role='Data Loader',
#     goal='Load and validate CSV data',
#     backstory='Expert in data ingestion and initial validation',
#     tools=[load_csv],
#     llm = llm,
#     verbose=True
# )

# data_cleaner = Agent(
#     role='Data Cleaner',
#     goal='Clean and preprocess data',
#     backstory='Specialist in data cleaning and handling missing values',
#     tools=[clean_data],
#     llm = llm,
#     verbose=True
# )

# feature_engineer = Agent(
#     role='Feature Engineer',
#     goal='Transform and prepare features for ML',
#     backstory='Expert in feature engineering and data transformation',
#     tools=[engineer_features],
#     llm = llm,
#     verbose=True
# )
    
 

# """Create and return the tasks for data processing"""
# load_task = Task(
#     description='Load and validate the CSV file',
#     agent=data_loader,
#     expected_output='Loaded DataFrame',
#     tools=[load_csv],
#     context={'file_path': file_path}
# )

# clean_task = Task(
#     description='Clean and preprocess the data',
#     agent=data_cleaner,
#     expected_output='Cleaned DataFrame',
#     tools=[clean_data]
# )

# engineer_task = Task(
#     description='Engineer features for ML',
#     agent=feature_engineer,
#     expected_output='ML-ready DataFrame',
#     tools=[engineer_features]
# )





# # Example usage
# if __name__ == "__main__":
#         # Create and run the crew
#     crew = Crew(
#         agents=[data_loader, data_cleaner, feature_engineer],
#         tasks=[load_task, clean_task, engineer_task],
#         process=Process.sequential
#     )

#     result = crew.kickoff(input="house.csv")




from crewai import Agent, Task, Crew, Process, LLM
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from dotenv import load_dotenv
import os
from crewai.tools import tool

load_dotenv()


# Initialize LLM
llm = LLM(
    model="groq/llama-3.3-70b-versatile",
    temperature=0.7
)




@tool("Tool Name")
def my_simple_tool(question: str) -> str:
    """Tool description for clarity."""
    # Tool logic here
    return "Tool output"




@tool("load_csv")
def load_csv(file_path):
    """Load CSV file and perform initial validation"""
    try:
        df = pd.read_csv(file_path)
        
        # Basic validation
        print(f"Dataset Shape: {df.shape}")
        print("\nData Types:\n", df.dtypes)
        print("\nMissing Values:\n", df.isnull().sum())
        
        return df
        
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None

def clean_data(df):
    """Clean data and handle missing values"""
    try:
        # Create a copy to avoid modifying original data
        df_clean = df.copy()
        
        # Handle missing numerical values
        numerical_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            num_imputer = SimpleImputer(strategy='mean')
            df_clean[numerical_cols] = num_imputer.fit_transform(df_clean[numerical_cols])
        
        # Handle missing categorical values
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols] = cat_imputer.fit_transform(df_clean[categorical_cols])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Remove columns with high percentage of missing values
        threshold = 0.7
        df_clean = df_clean.dropna(axis=1, thresh=int(threshold * len(df_clean)))
        
        return df_clean
        
    except Exception as e:
        print(f"Error cleaning data: {str(e)}")
        return None

def engineer_features(df):
    """Perform feature engineering and prepare for ML"""
    try:
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
        
        return df_engineered
        
    except Exception as e:
        print(f"Error engineering features: {str(e)}")
        return None

# def create_agents():
    """Create and return the agents for data processing"""
data_loader = Agent(
    role='Data Loader',
    goal='Load and validate CSV data',
    backstory='Expert in data ingestion and initial validation',
    tools=[load_csv],
    llm = llm,
    verbose=True
)

data_cleaner = Agent(
    role='Data Cleaner',
    goal='Clean and preprocess data',
    backstory='Specialist in data cleaning and handling missing values',
    tools=[clean_data],
    llm = llm,
    verbose=True
)

feature_engineer = Agent(
    role='Feature Engineer',
    goal='Transform and prepare features for ML',
    backstory='Expert in feature engineering and data transformation',
    tools=[engineer_features],
    llm = llm,
    verbose=True
)
    
 

"""Create and return the tasks for data processing"""
load_task = Task(
    description='Load and validate the CSV file',
    agent=data_loader,
    expected_output='Loaded DataFrame',
    tools=[load_csv],
    context={'file_path': file_path};
)

clean_task = Task(
    description='Clean and preprocess the data',
    agent=data_cleaner,
    expected_output='Cleaned DataFrame',
    tools=[clean_data]
)

engineer_task = Task(
    description='Engineer features for ML',
    agent=feature_engineer,
    expected_output='ML-ready DataFrame',
    tools=[engineer_features]
)





# Example usage
if __name__ == "__main__":
        # Create and run the crew
    crew = Crew(
        agents=[data_loader, data_cleaner, feature_engineer],
        tasks=[load_task, clean_task, engineer_task],
        process=Process.sequential
    )

    result = crew.kickoff(input="house.csv")

