import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import tempfile
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to load the Llama model
def load_llm():
    llm = CTransformers(
        model = "llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5
    )
    return llm

# Function to read and parse CSV
def read_csv(file):
    return pd.read_csv(file)

# Function to calculate statistics
def calculate_statistics(data):
    return {
        'mean': data.mean().to_dict(),
        'median': data.median().to_dict(),
        'mode': data.mode().iloc[0].to_dict(),
        'std_dev': data.std().to_dict(),
        'correlation': data.corr().to_dict()
    }

# Function to plot histogram
def plot_histogram(data, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[column], kde=True, ax=ax)
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    return fig

# Function to plot scatter plot
def plot_scatter(data, column1, column2):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=column1, y=column2, ax=ax)
    ax.set_title(f'Scatter Plot: {column1} vs {column2}')
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    return fig

# Function to plot line plot
def plot_line(data, column):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data.index, data[column])
    ax.set_title(f'Line Plot of {column}')
    ax.set_xlabel('Index')
    ax.set_ylabel(column)
    return fig

# Function to generate graph summary
def generate_graph_summary(data, plot_type, column1, column2=None):
    summary = f"Summary of {plot_type}:\n\n"
    
    if plot_type == "Histogram":
        mean = data[column1].mean()
        median = data[column1].median()
        std_dev = data[column1].std()
        skewness = data[column1].skew()
        
        summary += f"Column: {column1}\n"
        summary += f"Mean: {mean:.2f}\n"
        summary += f"Median: {median:.2f}\n"
        summary += f"Standard Deviation: {std_dev:.2f}\n"
        summary += f"Skewness: {skewness:.2f}\n\n"
        
        if skewness > 0.5:
            summary += "The distribution is positively skewed (right-tailed).\n"
        elif skewness < -0.5:
            summary += "The distribution is negatively skewed (left-tailed).\n"
        else:
            summary += "The distribution is approximately symmetric.\n"
        
    elif plot_type == "Scatter Plot":
        correlation = data[column1].corr(data[column2])
        summary += f"Correlation between {column1} and {column2}: {correlation:.2f}\n\n"
        
        if correlation > 0.7:
            summary += "There is a strong positive correlation between the variables.\n"
        elif correlation < -0.7:
            summary += "There is a strong negative correlation between the variables.\n"
        elif -0.3 <= correlation <= 0.3:
            summary += "There is little to no correlation between the variables.\n"
        else:
            summary += "There is a moderate correlation between the variables.\n"
        
    elif plot_type == "Line Plot":
        trend = np.polyfit(data.index, data[column1], 1)
        summary += f"Column: {column1}\n"
        summary += f"Overall trend: {'Increasing' if trend[0] > 0 else 'Decreasing'}\n"
        summary += f"Slope: {trend[0]:.4f}\n\n"
        
        if abs(trend[0]) > 1:
            summary += "The trend shows a steep change over time.\n"
        elif 0.1 < abs(trend[0]) <= 1:
            summary += "The trend shows a moderate change over time.\n"
        else:
            summary += "The trend shows a slight change over time.\n"
    
    return summary

def run_streamlit():
    st.title("CSV Analysis and Chat with Llama2 🦙📊")

    # Load the LLM at the beginning
    llm = load_llm()

    uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")

    if uploaded_file:
        # Read the CSV file
        data = read_csv(uploaded_file)
        
        # Use tempfile because CSVLoader only accepts a file_path
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
        documents = loader.load()

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',
                                           model_kwargs={'device': 'cpu'})

        db = FAISS.from_documents(documents, embeddings)
        db.save_local(DB_FAISS_PATH)
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,  # Now llm is defined and accessible here
            retriever=db.as_retriever(),
            memory=memory
        )

        # Data Analysis and Visualization
        st.sidebar.subheader("Data Analysis")
        analysis_option = st.sidebar.selectbox("Choose Analysis", ["Statistics", "Visualization"])

        if analysis_option == "Statistics":
            st.subheader("Statistical Analysis")
            stats = calculate_statistics(data)
            st.write("Mean:", stats['mean'])
            st.write("Median:", stats['median'])
            st.write("Mode:", stats['mode'])
            st.write("Standard Deviation:", stats['std_dev'])
            st.write("Correlation Matrix:")
            st.write(pd.DataFrame(stats['correlation']))

        elif analysis_option == "Visualization":
            st.subheader("Data Visualization")
            plot_type = st.selectbox("Select Plot Type", ["Histogram", "Scatter Plot", "Line Plot"])
            
            if plot_type == "Histogram":
                column = st.selectbox("Select Column for Histogram", data.columns)
                fig = plot_histogram(data, column)
                st.pyplot(fig)
                summary = generate_graph_summary(data, plot_type, column)
            elif plot_type == "Scatter Plot":
                column1 = st.selectbox("Select X-axis Column", data.columns)
                column2 = st.selectbox("Select Y-axis Column", data.columns)
                fig = plot_scatter(data, column1, column2)
                st.pyplot(fig)
                summary = generate_graph_summary(data, plot_type, column1, column2)
            elif plot_type == "Line Plot":
                column = st.selectbox("Select Column for Line Plot", data.columns)
                fig = plot_line(data, column)
                st.pyplot(fig)
                summary = generate_graph_summary(data, plot_type, column)

            st.subheader("Graph Summary")
            st.write(summary)

        # Chat with CSV
        st.sidebar.subheader("Chat with CSV")
        
        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]
        
        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey! 👋"]
            
        response_container = st.container()
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and user_input:
                output = conversational_chat(user_input)
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    st.write("User: " + st.session_state["past"][i])
                    st.write("Bot: " + st.session_state["generated"][i])
                    st.write("---")

if __name__ == '__main__':
    run_streamlit()