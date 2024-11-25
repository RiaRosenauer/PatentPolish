# PatentPolish

PatentPolish is a tool designed to improve the quality of patent filings by identifying formal mistakes, particularly focusing on reference sign consistency between patent text and figures.

## Self-deployed Version: 
http://shnei.de:8501/ 

## Features
- Analyzes patent documents for reference sign consistency
- Compares reference signs in text (abstract, claims, descriptions) with those in figures
- Identifies discrepancies and potential errors in reference sign usage
- Provides a user-friendly interface for reviewing and addressing issues

## What you need
- Python 3.11 or higher
- pixi installed
  `curl -fsSL https://pixi.sh/install.sh | bash`
- git installed
  `brew install git`

## Getting started
1. Clone the repository `git clone "https://github.com/epo/CF24-PatentPolish.git"`
2. Navigate to the cloned project directory: `cd "CF24-PatentPolish/Source Code/patentpolish"`
3. If you want to run the code, duplicate the .env_example, rename it to .env and fill in the relevant API keys. We unfortunately cannot and are not allowed to push our own. If you want to test the app without installing it, checkout our deployed version: http://shnei.de:8501/ 
4. `pixi install`
5. `pixi run main` for testing an example or `pixi run frontend` to run our frontend

## Troubleshooting
1. Try using pixi 0.29.0
2. Delete .pixi folder and pixi.lock and try again

## Project Structure
The main components of the project are located in the src/patentpolish directory:
- sign_examiner.py: 
  This file defines an Examiner class and related functionality for analyzing patent documents, particularly focusing on reference signs in patent texts and images. Here's a summary of its main components and functionality:
  - **Data Models**: Defines various data models using Pydantic for structured representation of reference signs, text types, and analysis results.
  - **Examiner Class**: Initializes with a patent object or reference number and contains methods for extracting and analyzing reference signs from both text and images.
  - **Text Analysis**: Extracts reference signs from abstract, claims, and description sections using OpenAI's GPT model for accurate extraction and context understanding.
  - **Image Analysis**: Extracts reference signs from patent figures using OpenAI's vision model.
  - **Comparison and Analysis**: Compares reference signs found in text and images, identifying discrepancies and analyzing consistency of concept descriptions for each sign.
  - **Utility Functions**: Includes helper functions for text processing, image encoding, and concept similarity checking.
  - **Execution**: Provides a run_comparison function to perform the entire analysis process and includes a _main_ section for standalone execution and testing.
  - **Logging**: Implements logging to track the analysis process and highlight important findings.
  
- api_connector.py: Handles connection to the EPO service
  - **OAuth Authentication**: Utilizes OAuth2 for secure access to the EPO API using client credentials stored in environment variables.
    
  - **Patent Data Retrieval**: 
    - Checks for locally stored patent data (in JSON format) before making API requests.
    - Fetches key patent details, including the title, abstract, claims, descriptions, images, and PDF documents.
    
  - **PDF Management**: Merges multiple PDF pages into a single document and handles the conversion of patent images from PDF to PNG format.

- frontend.py: Implements the Streamlit-based user interface
  - **Patent Creation/Loading**
    - ⁠Users can enter a patent name or publication number
    - ⁠Patents can be loaded from local files or downloaded from the EPO (European Patent Office)
  - **Patent Editing**
    - ⁠Edit patent title, abstract, claims, and descriptions
    - ⁠Add, edit, or delete claims and descriptions
    - ⁠Upload and manage patent images
  - **Formality Check**
    - ⁠Perform a formality check on the patent
    - ⁠Analyze reference signs in text and images
    - ⁠Compare signs and identify potential issues
  - **Results Display**
    - Show a list of reference signs with potential errors
    - ⁠Display relevant patent images
    - ⁠Highlight occurrences of selected reference signs in the patent text
    - ⁠Present a sortable table of all reference signs and their occurrences

## Configuration
The project uses configuration files located in the config directory. For production environments, use the config_prod.yaml file.

## Docker Support
The project includes Docker support for containerized deployment. Refer to the Dockerfile and docker-compose.yaml for container configurations.

## Acknowledgments
European Patent Office (EPO) for providing access to patent data
