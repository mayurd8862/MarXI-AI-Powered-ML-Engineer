
import tempfile
import subprocess
import os
import shutil

def copy_MarXI_archive(repo_url, path):
    try:
        # Create a temporary directory for cloning the repository
        temp_path = tempfile.mkdtemp(prefix='temp_clone_')

        # Clone the repository to the temp folder directory
        subprocess.run(['git', 'clone', '--depth', '1', repo_url, temp_path], check=True)

        # Create the project directory if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Move the files from the temp folder to the project directory
        for item in os.listdir(temp_path):
            source = os.path.join(temp_path, item)
            destination = os.path.join(path, item)
            if os.path.isdir(source) and item != ".git":
                shutil.move(source, destination)
            elif os.path.isfile(source):
                shutil.move(source, path)

    except subprocess.CalledProcessError as e:
        print("Error:", e)

    except PermissionError as e:
        print("Permission error during file operations:", e)

    except Exception as e:
        print("Error:", e)

    finally:
        # Clean up: Attempt to remove the temp folder, ignoring any errors
        try:
            shutil.rmtree(temp_path, ignore_errors=True)
        except Exception as e:
            print("Error during cleanup:", e)
