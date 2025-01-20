# Import necessary libraries
import streamlit as st
from transformers import pipeline
from pprint import pprint

# Create the text generation pipeline
gen = pipeline('text-generation')

# Streamlit app
def main():
    st.title('Text Generation with Transformers')
    # Input prompt from the user
    prompt = st.text_input("Enter a prompt", "The cats and dogs are fighting over free food")
    # Generate text on button click
    if st.button('Generate Text'):
        # Generate the text using the provided prompt
        generated_text = gen(prompt, num_return_sequences=3, max_length=50)
        
        # Display the generated text
        st.subheader('Generated Text:')
        for i, text in enumerate(generated_text):
            st.write(f"Option {i + 1}: {text['generated_text']}")

# Run the app
if __name__ == '__main__':
    main()
