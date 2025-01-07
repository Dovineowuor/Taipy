import os
import pandas as pd
import taipy.gui.builder as tgb
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.indexes import VectorstoreIndexCreator
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from taipy.gui import Gui, notify
from google.colab import userdata
from pyngrok import ngrok
from langchain.vectorstores import Chroma

# Get your authtoken from https://dashboard.ngrok.com/auth
# and set it as an environment variable or directly in the code
NGROK_AUTH_TOKEN = userdata.get('NGROK_AUTH_TOKEN') or 'YOUR_AUTH_TOKEN'  # Replace with your token if you want to hardcode
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Load environment variables
load_dotenv()  # Load .env file (if available)
HF_TOKEN = userdata.get("HF_TOKEN") or os.getenv("HF_TOKEN")
GOOGLE_AI_API_KEY = userdata.get("GOOGLE_AI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")

# Check if the tokens are loaded correctly
if not HF_TOKEN or not GOOGLE_AI_API_KEY:
    print("Warning: HF_TOKEN or GOOGLE_AI_API_KEY not found. Please check your environment variables.")

# Define the folder path for the FAQ text file
data_dir = "./Taipy/data"

# Initialize the data loader
loader = DirectoryLoader(data_dir, glob='*.txt')
docs = loader.load()

# Create the document index (Using Chroma for persistence)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory='./chroma_db')  # Persist data
index = VectorstoreIndexCreator(
    embedding=embeddings,
    vectorstore_kwargs={"persist_directory": "./chroma_db"}  # Persist data
).from_documents(docs)

# Configure the LLM and retrieval chain
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

# Run the app using Flask and ngrok for public access
public_url = ngrok.connect(60675)  # Expose the Flask server
print(f" * Running on {public_url}")

# Start the GUI with unsafe werkzeug settings
gui = Gui(page)
conv = gui.add_partial("")
# gui.run(title="StackUp Assistant", dark_mode=False, margin="0px", debug=True, port=60675, allow_unsafe_werkzeug=True)
gui.run(title="StackUp Assistant", dark_mode=False, margin="0px", debug=False, 
        port=60675, allow_unsafe_werkzeug=True, use_reloader=False)
