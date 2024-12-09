import random
import json
import torch
import streamlit as st
from model import NeuralNet
from chatbot import bag_of_words, tokenize

# Load device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load metadata from a separate JSON file
with open('data.json', 'r') as f:
    data = json.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]

# Load the model
model = NeuralNet(input_size, hidden_size, output_size).to(device)
model_state = torch.load("model_weights.pth", map_location=device)
model.load_state_dict(model_state)
model.eval()  # Set the model to evaluation mode

# Chatbot function
def get_response(sentence):
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).reshape(1, X.shape[0]).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                return random.choice(intent['responses'])

    return "I do not understand..."

# Streamlit interface
def main():

    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://media.istockphoto.com/id/1488335095/vector/3d-vector-robot-chatbot-ai-in-science-and-business-technology-and-engineering-concept.jpg?s=612x612&w=0&k=20&c=MSxiR6V1gROmrUBe1GpylDXs0D5CHT-mn0Up8D50mr8=");
            background-size: cover;
            background-position: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("AI Chatbot")
    st.write("Welcome! Ask me anything. Type 'quit' to end the chat.")
    if "history" not in st.session_state:
        st.session_state.history = []

    # Display chat history
    for chat in st.session_state.history:
        st.write(f"You: {chat['user']}")
        st.write(f"bot: {chat['bot']}")



    # Get user input and display response
    user_input = st.text_input("You:")

    if st.button("Send") and user_input:
        if user_input.lower() == 'quit':
            st.write("Chat ended.")
        else:
            # Get response from chatbot
            response = get_response(user_input)
            st.session_state.history.append({"user": user_input, "bot": response})

            

if __name__ == "__main__":
    main()
