import streamlit as st
import pandas as pd
import torch

from src.bart import t5_summary


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
text = st.text_area("Enter the large text:", height=200)
# Check if the "Generate Summary" button is clicked
if st.button("Generate Summary"):
    # If so, generate a summary using the t5_summary() function
    summary = t5_summary(text)
    
    # Display the summary as a subheader and highlight it in green
    st.subheader("Summary:")
    st.success(summary)
else:
    # If the button is not clicked, show a warning message
    st.warning("Please enter some content for summarization.")

