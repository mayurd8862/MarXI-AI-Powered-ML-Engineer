import os

# Define the folder path where you want to save the uploaded file
folder_path = os.getcwd()

# Function to save the uploaded file
def save_uploaded_file(uploaded_file, folder):
    # Create the necessary directories if they don't exist
    os.makedirs(os.path.join(folder), exist_ok=True)
    with open(os.path.join(folder, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())