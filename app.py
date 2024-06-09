# st.write("Flow of project")
# st.write(
# """
# 1. Create separate folder for the project
# 2. Create Virtual environment of the project 
# 3. connect project with github repo
# 4. take Project name from the user
# 4. Take input data and description of data from user 
# 5. Save raw file of the data in data folder
# 6. provide target and feature variables from the data
# 7. data ingestion, preprocessor, trainer
# 8. Start Building 
# """
# )


import streamlit as st
import os
from utils.github_setup import copy_MarXI_archive, create_repo, commit_push
from utils.EDA import data_analysis
from utils.save_csv import save_uploaded_file
from utils.create_env import *
import subprocess
import shutil

st.title("🤖MarXI: AI powered ML engineer")


parent_path = os.path.dirname(os.getcwd())
project_path = os.path.join(parent_path,'project')
project_path = os.path.join(parent_path,'project')

data = st.file_uploader("📤 Upload a CSV file", type="csv")
if data:
    st.success('CSV file uploaded Successfully!', icon='✔️')



data_desc = st.text_area("📈 Enter dataset description: ", placeholder= "Provide information about columns in dataset, dataset source(kaggle, dataocean etc.)")
st.session_state.data_desc = data_desc


option = st.selectbox(
    "How would you like to be contacted?",
    ("Analysis", "Predictive Model", "Classification Model"))
st.session_state.option = option


if option == 'Analysis':
    st.markdown("**📉 Exploratory Data analysis**")
    if st.button("EDA"):
        data_analysis(data)
    # pass


else:

    # st.write("You selected:", option)
    project_name = st.text_input("🎯 Enter the Project name here:", placeholder = 'Ex. House price prediction')
    st.session_state.project_name = project_name


    project_desc = st.text_area("📇 Enter project description: ", placeholder = "Ex. Develop a machine learning model to predict house prices, facilitating informed decisions for buyers, sellers, and real estate professionals.")
    st.session_state.project_desc = project_desc


    if st.button("🛠️ Build"):

        st.write("\n🌱 Started Building Your Project:")

        with st.spinner("project setup..."):
            repository_url = 'https://github.com/mayurd8862/MarXI-Archive.git'
            copy_MarXI_archive(repository_url,project_path)

            # copy_MarXI_archive(repository_url)
            file = os.path.join(project_path, 'template.py')
            subprocess.run(['python', file], cwd=project_path)

            if option == 'Classification Model':
                # shutil.rmtree()
                pass

            elif option == 'Predictive Model':
                pass

        st.write("✔️1. Project folder created and done with the files setup.")


        with st.spinner("Saving project info..."):
            data_path = os.path.join(project_path,'data')
            if data is not None:
            # Save the uploaded file
                save_uploaded_file(data, data_path)

        st.write("✔️2. Saved uploaded CSV file.")


        
        # proj = os.path.join(os.getcwd(),'project')
        with st.spinner("Creating environment..."):
            create_env(project_path)

        st.write("✔️3. Created virtual environment for the project")

    

        with st.spinner("Creating github repository ..."):
            access_token = "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"
            repo_name = project_name
            create_repo(project_path,access_token, repo_name)

        st.write("✔️4. GitHub repo created for the project ") 






st.session_state


# repository_url = 'https://github.com/mayurd8862/MarXI-Archive.git'
# copy_MarXI_archive(repository_url,project_path)