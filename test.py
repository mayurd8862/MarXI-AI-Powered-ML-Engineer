import shutil
import os

parent_path = os.path.dirname(os.getcwd())
project_path = os.path.join(parent_path,'project')

def copy_files(source_folder, destination_folder):
    # List all files in the source folder
    files = os.listdir(source_folder)
    
    # Iterate through each file and copy it to the destination folder
    for file in files:
        source_file = os.path.join(source_folder, file)
        destination_file = os.path.join(destination_folder, file)
        shutil.copy(source_file, destination_file)

# # # Example usage
source_folder = os.path.join(parent_path,r'MarXI\prediction_archive')
destination_folder = project_path

# Ensure destination folder exists
os.makedirs(destination_folder, exist_ok=True)

# # Call the function to copy files
copy_files(source_folder, destination_folder)

