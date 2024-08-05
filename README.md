# Streamlit CSV Analyzer
This is a Streamlit application that allows users to upload CSV files, perform statistical analysis, generate various plots, and interact with the data using a conversational interface. The application leverages machine learning models for enhanced data interaction.

## Features 
- CSV File Upload: Easily upload CSV files for analysis.
- Statistical Analysis: Calculate and display mean, median, mode, standard deviation, and correlation of the data.
- Plot Generation: Generate histograms, scatter plots, and line plots.
- Graph Summarization: Provide summaries for the generated plots.
- Conversational Interface: Interact with your CSV data using a chat-like interface.

## Installation
 - Clone the Repository
```
git clone https://github.com/yourusername/ChatwithCSV_LLM.git
cd ChatwithCSV_LLM
```
 - Create and activate a virtual environment (optional but recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
 - Install the dependencies:
```
pip install -r requirements.txt
```

## Usage
1. Run the Streamlit Application
```
streamlit run app.py
```
2. Upload your CSV file through the interface.
3. Explore the Data:
   - View statistical summaries.
   - Generate and visualize plots.
   - Interact with the data using the conversational interface.


## File Structure
```
streamlit-csv-analyzer/
│
├── app.py               # Main application script
├── requirements.txt     # List of dependencies
└── README.md            # Project documentation (this file)
```

## Dependencies
- streamlit
- pandas
- matplotlib
- seaborn
- numpy
- langchain-community
- langchain
- faiss-cpu

For a complete list of dependencies, refer to the requirements.txt file.
