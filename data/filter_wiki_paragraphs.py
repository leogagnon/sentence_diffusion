from datasets.load import load_dataset
from transformers.models.t5.tokenization_t5 import T5Tokenizer

enc_tokenizer = T5Tokenizer.from_pretrained("thesephist/contra-bottleneck-t5-xl-wikipedia")

ds = load_dataset("singletongue/wikipedia-paragraphs", 'enwiki-20250901')

def extract_and_filter_paragraphs_batched(examples):
    """
    Extract individual paragraphs from documents and filter them.
    Returns a flattened structure where each item is a single paragraph.
    """
    filtered_paragraphs = []
    flat_paragraphs = [item for sublist in examples['paragraph_texts'] for item in sublist]
    
    # Handle empty batch
    if flat_paragraphs == []:
        return {"input_ids": filtered_paragraphs}
    
    enc_tokens = enc_tokenizer.batch_encode_plus(flat_paragraphs)['input_ids']
        
    for i, text in enumerate(flat_paragraphs):
        if not text[0].isupper():
            continue
        
        # Check encoder tokenization (T5)
        if len(enc_tokens[i]) < 64 or len(enc_tokens[i]) > 128:
            continue

        filtered_paragraphs.append(text)

    return {"input_ids": filtered_paragraphs}

# Apply the transformation to create a flattened paragraph dataset
print("Extracting and filtering individual paragraphs...")
paragraph_ds = ds.map(
    extract_and_filter_paragraphs_batched,
    batched=True,
    batch_size=10,
    num_proc=4,
    remove_columns=ds['train'].column_names,  # Remove original columns
    keep_in_memory=True,
    desc="Extracting individual paragraphs"
)

# The result will have many more items since we're flattening
print(f"Total filtered paragraphs: {len(paragraph_ds['train'])}")

# Save the paragraph dataset
paragraph_ds.save_to_disk("/network/scratch/l/leo.gagnon/sentence_diffusion/data/wikipedia-paragraphs-filtered")