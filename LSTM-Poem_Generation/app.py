import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Import pre-trained model and tokenizer
model = load_model('model1.keras')
with open('tokenizer1.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the function for generating text
def generate_text(input_text, no_lines, model, tokenizer, seq_length, text_length=15):
    general_text = []
    for i in range(no_lines):
        text = []
        for _ in range(text_length):
            encoded = tokenizer.texts_to_sequences([input_text])
            encoded = pad_sequences(encoded, maxlen=seq_length, padding="pre")
            y_pred = np.argmax(model.predict(encoded), axis=-1)  # Predict the next word index
            predicted_word = ""
            for word, index in tokenizer.word_index.items():
                if index == y_pred:
                    predicted_word = word
                    break
            input_text = input_text + ' ' + predicted_word
            text.append(predicted_word)
        input_text = text[-1]  # Set input_text to the last word of the created line
        text = " ".join(text)
        general_text.append(text)

    return general_text

# Streamlit app interface
st.title("Text Generation with LSTM")

# Input fields for text generation
st.header("Poem Text Generation ")
seq_length = 10
#st.sidebar.number_input("Sequence Length", min_value=1, value=50, step=1)
input_text = st.text_input("Enter Starting Text", value="me")
no_lines = st.number_input("Number of Lines", min_value=1, value=6, step=1)

if st.button("Generate Text"):
    # Generate text
    try:
        output = generate_text(input_text, no_lines, model, tokenizer, seq_length)
        st.subheader("Generated Text")
        for i, line in enumerate(output):
            st.write(f"Line {i + 1}: {line}")
    except Exception as e:
        st.error(f"Error generating text: {e}")
