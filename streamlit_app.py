import streamlit as st
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
