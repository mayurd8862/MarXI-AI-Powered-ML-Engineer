import subprocess
import os

project_path = os.path.join(os.getcwd(),'project')

def create_env(project_path):
    # Path to the virtual environment
    venv_path = os.path.join(project_path, 'venv')

    # Create a virtual environment named "venv"
    subprocess.run(["python", "-m", "venv", venv_path])

    # Check for pip updates
    subprocess.run([os.path.join(venv_path, 'Scripts', 'python.exe'), "-m", "pip", "install", "--upgrade", "pip"])

def download_req(project_path):
    venv_path = os.path.join(project_path, 'venv')
    # Check for pip updates
    subprocess.run([os.path.join(venv_path, 'Scripts', 'python.exe'), "-m", "pip", "install", "--upgrade", "pip"])

    req = os.path.join(project_path, 'requirements.txt')
    subprocess.run([os.path.join(venv_path, 'Scripts', 'pip'), "install", "-r", req])
    print("...................................................................\n")


# NOTE: This is to run files in project folder inside our environment 
def run_file(file_name):
    venv_path = os.path.join(project_path, 'venv')
    file_path = os.path.join(project_path, file_name)
    subprocess.run([os.path.join(venv_path, 'Scripts', 'python'), file_path])
