# Langchain dependencies
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI  # OpenAI ChatGPT integration
from langchain.memory import ConversationBufferMemory  # For conversational memory

import streamlit as st  # Streamlit for user interface development
import os

# Set up OpenAI API credentials
os.environ["OPENAI_API_KEY"] = ''

# Load OpenAI model using LangChain's OpenAI wrapper
llm = OpenAI(model="gpt-3.5-turbo", temperature=0.5, max_tokens=200) # Temperature determines "creativity"

# Function to load and split the PDF into chunks, and create vectorstore index
@st.cache_resource
def load_pdf():
    pdf_name = '' # e.g. TheLittlePrince.pdf
    loaders = [PyPDFLoader(pdf_name)]
    
    # Creates an index / vector database
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    ).from_loaders(loaders)

    return index

# Function to extract character-specific dialogue from the PDF
def extract_character_dialogues(text, character_name=""): # e.g. The Little Prince
    dialogues = []
    for line in text.split("\n"):
        if character_name in line:  # Modify this logic for better character detection if necessary
            dialogues.append(line)
    return " ".join(dialogues)

# Loads the PDF and extract character-specific data
index = load_pdf()
pdf_data = index.vectorstore.document_store  # Get the raw document text
character_data = extract_character_dialogues(pdf_data)

# Initializes conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question")

# Character's name
character_name = "" # e.g. The Little Prince

# Sets up the RetrievalQA chain with memory
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    memory=memory
)

# App title
st.title('') # e.g. Character Chat with The Little Prince

# Session state message variable to retain old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Prompt for user
prompt = st.chat_input('') # e.g. Start your discussion with The Little Prince here: 

if prompt:
    st.chat_message('user').markdown(prompt)  # Display the user's prompt
    st.session_state.messages.append({'role': 'user', 'content':prompt})  # Store the user prompt in session state

    # Generate a response with the character's perspective
    custom_prompt = f"As {character_name}, respond to the following question based on your personality and experiences: {prompt}"
    response = chain.run(custom_prompt)  # Send the prompt to the RetrievalQA chain

    st.chat_message('assistant').markdown(response)  # Display the assistant's response
    st.session_state.messages.append({'role': 'assistant', 'content': response})  # Store the response in session state