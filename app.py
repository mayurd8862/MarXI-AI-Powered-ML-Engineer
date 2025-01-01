import streamlit as st
import os
import subprocess
import shutil
import pandas as pd
from src.data_ingestion import data_ingestion

st.title("🤖MarXI: AI powered ML engineer")

# Function to save the uploaded file
def save_uploaded_file(uploaded_file):
    # Create a directory to save the file if it doesn't exist
    save_dir = "artifacts"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the file to the directory with the name 'raw.csv'
    file_path = os.path.join(save_dir, "raw.csv")
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path


project_name = st.text_input("🎯 Enter the Project name here:", placeholder = 'Ex. House price prediction')
st.session_state.project_name = project_name


# project_desc = st.text_area("📇 Enter project description: ", placeholder = "Ex. Develop a machine learning model to predict house prices, facilitating informed decisions for buyers, sellers, and real estate professionals.")
# st.session_state.project_desc = project_desc

# Create a file uploader widget
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Save the file
    file_path = save_uploaded_file(uploaded_file)
    st.success(f"File saved to {file_path}")

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Display the dataframe
    st.write("## DataFrame")
    st.dataframe(df[:5])

    option = st.selectbox(
        "How would you like to be contacted?",
        ("Analysis", "Predictive Model", "Classification Model"))
    st.session_state.option = option

    target = st.selectbox(
        "Select 'TARGET' variable",
        df.columns)
    st.session_state.target = target

    col_drops = st.multiselect(
        "Select columns from the dataset you want to drop",
        df.columns,
    )
    st.session_state.colums_to_drop = col_drops

    # st.write(col_drops)
    # col_drops.append(target)
    if col_drops:
        df = df.drop(columns=col_drops)
        st.write("### Updated DataFrame after dropping columns")
        st.dataframe(df[:5])
        drop_file_path = os.path.join("artifacts", "after_drop_col.csv")
        df.to_csv(drop_file_path,index=False)

    

btn = st.button("🛠️ Build")

if btn:

    st.write("\n🌱 Started Building Your Project:")

    with st.spinner("Data Ingestion started..."):
        raw_file_path = os.path.join("artifacts", "raw.csv")
        # data_ingestion(raw_file_path)
        

    st.success("✔️1. Data ingestion complete. Training and testing datasets have been saved to the artifacts folder.")
    st.write("**Target Variable:**",target)
    st.write("**Columns to drop before training:**",col_drops)


    with st.spinner("project setup..."):

        if option == 'Classification Model':
            # shutil.rmtree()
            pass

        elif option == 'Predictive Model':
            pass

    st.success("✔️1. Project folder created and done with the files setup.")




    
    # proj = os.path.join(os.getcwd(),'project')
    with st.spinner("Creating environment..."):
        pass

    st.success("✔️3. Created virtual environment for the project")



    with st.spinner("Creating github repository ..."):
        pass

    st.success("✔️4. GitHub repo created for the project ") 




st.session_state


# repository_url = 'https://github.com/mayurd8862/MarXI-Archive.git'
# copy_MarXI_archive(repository_url,project_path)