import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarize(content):
    # Load the abstractive summarization model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

    # Preprocess the news article by encoding it as input_ids and attention_mask
    inputs = tokenizer(content, padding=True, truncation=True, max_length=512, return_tensors='pt')

    # Generate summary
    outputs = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=150, num_beams=4, early_stopping=True)

    # Decode the summary from the output_ids
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary
    # Set the app title
    st.title("Large text Summarizer")

    # Create a text input for entering the news article content
    content = st.text_area("Enter the large text:", height=200)

    # Create a button for generating the summary
    if st.button("Generate Summary"):
        if content:
            # Call the summarize function to generate the summary
            summary = summarize(content)

            # Display the summary
            st.subheader("Summary:")
            st.write(summary)
            st.success("Summary generated successfully!")
        else:
            st.warning("Please enter some content for summarization.")
