import os
import pandas as pd
import taipy.gui.builder as tgb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from taipy.gui import Gui, notify

# Load Environment Variables
load_dotenv()
GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
if not os.getenv("GOOGLE_AI_API_KEY"):
    raise ValueError("Environment variables not loaded correctly. Check .env file.")

# Define Data Paths
DATA_DIR = './data'

# Initialize Data Loaders
try:
    loader = DirectoryLoader(DATA_DIR, glob='*.txt')
    docs = loader.load()
    if not docs:
        raise ValueError("No documents found in the data directory.")
except Exception as e:
    raise RuntimeError(f"Failed to load documents: {e}")

# Create Document Index
index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=4000, chunk_overlap=200),
).from_documents(docs)

# Configure LLM and Retrieval Chain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.4,
    max_tokens=512,
    timeout=None,
    max_retries=2,
)
chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    input_key="question",
)

# Define the function to query the LLM
def query_llm(query_message):
    return chain.run(query_message)

# Initialize chatbot state variables
query_message = ""
messages = []
messages_dict = {}

def on_init(state):
    state.messages = [
        {
            "style": "assistant_message",
            "content": "Hi, I am StackUp assistant! How can I help you today?",
        },
    ]
    new_conv = create_conv(state)
    state.conv.update_content(state, new_conv)

def create_conv(state):
    messages_dict = {}
    with tgb.Page() as conversation:
        for i, message in enumerate(state.messages):
            text = message["content"].replace("<br>", "\n").replace('"', "'")
            messages_dict[f"message_{i}"] = text
            tgb.text(
                f"{{messages_dict.get('message_{i}') or ''}}",
                class_name=f"message_base {message['style']}",
                mode="md",
            )
    state.messages_dict = messages_dict
    return conversation

def send_message(state):
    state.messages.append(
        {
            "style": "user_message",
            "content": state.query_message,
        }
    )
    state.conv.update_content(state, create_conv(state))
    notify(state, "info", "Sending message...")
    assistant_response = query_llm(state.query_message)
    state.messages.append(
        {
            "style": "assistant_message",
            "content": assistant_response,
        }
    )
    state.conv.update_content(state, create_conv(state))
    state.query_message = ""

def reset_chat(state):
    state.query_message = ""
    on_init(state)

# Design the GUI layout
with tgb.Page() as page:
    with tgb.layout(columns="350px 1"):
        with tgb.part(class_name="sidebar"):
            tgb.text("## StackUp Assistant", mode="md")
            tgb.button(
                "New Conversation",
                class_name="fullwidth plain",
                on_action=reset_chat,
            )

        with tgb.part(class_name="p1"):
            tgb.part(partial="{conv}", height="600px", class_name="card card_chat")
            with tgb.part("card mt1"):
                tgb.input(
                    "{query_message}",
                    on_action=send_message,
                    change_delay=-1,
                    label="Write your message:",
                    class_name="fullwidth",
                )

# Start the GUI with unsafe werkzeug settings
gui = Gui(page)
conv = gui.add_partial("")
gui.run(title="StackUp Assistant", dark_mode=False, margin="0px", debug=False, 
        port=5000, allow_unsafe_werkzeug=True, use_reloader=True, conv=conv,)
