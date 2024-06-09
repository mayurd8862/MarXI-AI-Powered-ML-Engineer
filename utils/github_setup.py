
import tempfile
import subprocess
import os
import shutil
from github import Github


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


def create_repo(proj_path, access_token, repo_name):

    # Create the 'mlproject' directory if it doesn't exist
    if not os.path.exists(proj_path):
        os.makedirs(proj_path)

    # Create a Github instance with the personal access token
    g = Github(access_token)
    
    # Create a new repository
    user = g.get_user()
    repo = user.create_repo(repo_name)

    # Initialize local git repository
    os.chdir(proj_path)
    os.system('git init')

    # Add remote origin
    os.system(f'git remote add origin {repo.clone_url}')

    # Add and commit files
    os.system('git add .')
    os.system('git commit -m "Initial commit"')

    # Push to remote repository
    os.system('git push -u origin master')

    print("Repository created successfully!")
    # return "Repository created successfully!"



def commit_push(proj_path,msg):
    # Add and commit files
    subprocess.run(["git", "add", "."], cwd=proj_path)
    subprocess.run(["git", "commit", "-m", msg], cwd=proj_path)

    # Push to remote repository
    subprocess.run(["git", "push", "-u", "origin", "master"], cwd=proj_path)
