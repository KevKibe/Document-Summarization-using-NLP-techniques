import streamlit as st
import pandas as pd
import torch

from transformers import BartTokenizer, BartForConditionalGeneration

tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")

def generate_summaries(content_list):
    """
    Generates summaries for a list of news articles.

    Args:
    - content_list (List[str]): A list of news article content.

    Returns:
    - List[str]: A list of summaries generated for each news article.
    """
    print('Length of content list:', len(content_list))
    # Initialize the tokenizer and model

    summaries = []
    for content in content_list:
        inputs = tokenizer(content, padding=True, truncation=True, max_length=512, return_tensors='pt')
        outputs = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_length=150, num_beams=4, length_penalty=2.0)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    print('Length of the summarized article:', len(summaries[0]))
    print(summaries)

    return summaries
PAGE_STYLE = """
<style>
body {
    font-family: Arial, sans-serif;
    background-color: #f8f9fa;
    margin: 0;
    padding: 0;
}

.container {
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
}

.title {
    font-size: 24px;
    font-weight: bold;
    margin-bottom: 20px;
}

.button {
    background-color: #007bff;
    color: #fff;
    padding: 12px 20px;
    border: none;
    cursor: pointer;
    font-size: 16px;
    margin-top: 16px;

</style>
"""
# Set the app title
st.title("Large Text Summarizer")

# Create a text input for entering the news article content
content_list = st.text_area("Enter the large text:", height=200)

# Create a button for generating the summary
if st.button("Generate Summary"):
    if content_list:
        # Call the generate_summaries function to generate the summary
        summaries = generate_summaries([content_list])

        # Display the summary
        st.subheader("Summary:")
        st.write(summaries[0])
        st.success("Summary generated successfully!")
    else:
        st.warning("Please enter some content for summarization.")
