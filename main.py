
import random
import json
import torch
import streamlit as st
from model import NeuralNet
from chatbot import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents from JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)
with open('data.json', 'r') as f:
    data = json.load(f)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
all_tags = data["tags"]  # Renamed from `tags`

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model_state = torch.load("model_weights.pth", weights_only=True) 
model.load_state_dict(model_state)
model.eval() 

bot_name = "bot"
print("Let's chat! Type 'quit' to exit.")

while True:
    sentence = input('You: ')
    if sentence == "quit":
        break
    
    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    
    predicted_index = predicted.item()
    if 0 <= predicted_index < len(all_tags):
        tag = all_tags[predicted_index]
    else:
        print(f"Invalid index: {predicted_index} for all_tags of length {len(all_tags)}")
        continue  # Skip the rest of the loop if index is invalid

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted_index]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand......")

def main():
    st.title("bot_name")
    st.write("Welcome! Ask me anything. Type 'quit' to end the chat.")
    
    if "history" not in st.session_state:
        st.session_state.history = []
    
    user_input = st.text_input("You:", key="input")

    if user_input:
        if user_input.lower() == 'quit':
            st.write("Chat ended.")
        else:
            # Get response from chatbot
            response = get_response(user_input)
            st.session_state.history.append({"user": user_input, "bot": response})

    # Display chat history
if st.session_state.history:
        for chat in st.session_state.history:
            st.write(f"You: {chat['user']}")
            st.write(f"Bot: {chat['bot']}")

if __name__ == "__main__":
    main()
