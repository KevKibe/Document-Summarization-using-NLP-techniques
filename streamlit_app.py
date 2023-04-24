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

# Create a button for generating the summary
if st.button("Generate Summary"):
   
            summary = t5_summary(text)
        # Display the summary
        st.subheader("Summary:")
        
        st.success(summary)
    else:
        st.warning("Please enter some content for summarization.")
