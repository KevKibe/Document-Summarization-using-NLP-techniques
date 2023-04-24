
import transformers
from transformers import BartTokenizer, BartForConditionalGeneration

def bart_summary(INCONTEXT: str) -> str:
    """
    Summary of BART model.
    
    """
    # Load BART model
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    ARTICLE = INCONTEXT
    inputs = tokenizer("summarize: " + ARTICLE,
                       return_tensors="pt",
                       max_length=1024,
                       truncation=True,
                       padding="max_length")
    outputs = model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=150,
        min_length=50,
        no_repeat_ngram_size=3,
        early_stopping=True,
        num_return_sequences=1,
    )
    SUMMARY = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return SUMMARY
