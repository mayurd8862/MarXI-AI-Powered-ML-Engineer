# MarXI - AI Powered Machine Learning Engineer

MarXI is a comprehensive machine learning engineer that automates the process of setting up, managing, and deploying machine learning projects. It facilitates data ingestion, preprocessing, model training, and provides seamless integration with GitHub for version control.

## Features

1. **Create a separate folder for the project**
2. **Create a virtual environment for the project**
3. **Connect the project with a GitHub repository**
4. **Take project name from the user**
5. **Take input data and description of data from the user**
6. **Save the raw file of the data in a data folder**
7. **Identify target and feature variables from the data**
8. **Data ingestion, preprocessing, and training**
9. **Start building the machine learning model**

## Getting Started

Follow the steps below to set up and use MarXI for your machine learning projects.

### Prerequisites

Ensure you have the following installed:
- Python 3
- Git

### Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/MarXI.git
    cd MarXI
    ```

2. Run the setup script to create a virtual environment and install dependencies:
    ```sh
    bash setup.sh
    ```

### Usage

1. **Create a New Project**
    ```sh
    python marxi.py --create-project
    ```
    You will be prompted to enter the project name.

2. **Connect to GitHub**
    ```sh
    python marxi.py --connect-github
    ```
    Follow the instructions to link your project with a GitHub repository.

3. **Add Data**
    ```sh
    python marxi.py --add-data
    ```
    You will be prompted to enter the path to your data file and a description of the data.

4. **Data Ingestion and Preprocessing**
    ```sh
    python marxi.py --ingest-data
    ```
    MarXI will automatically identify target and feature variables and preprocess the data.

5. **Train the Model**
    ```sh
    python marxi.py --train-model
    ```
    This command will start the model training process.

### Project Structure

