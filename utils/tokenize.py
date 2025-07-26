from transformers import AutoTokenizer, AutoModel
import torch
import os

os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

def tokenize_dataframe_fields(df, model_name="bert-base-uncased", max_length=512):
    """
    Tokenizes multiple columns from a DataFrame along with a combined version,
    and returns both tokenized outputs and original cleaned fields.

    Args:
        df (pd.DataFrame): DataFrame containing required columns.
        model_name (str): Hugging Face model name (default: 'bert-base-uncased').
        max_length (int): Max sequence length for padding/truncation.

    Returns:
        dict: Dictionary with tokenized tensors and the original cleaned text fields.
    """
    print(f"\n📦 Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Fill missing values and define combined field
    print("🧹 Cleaning and combining fields...")
    def combine_fields(row):
        return (
            f"Article Title: {row['Article Title']} "
            f"Abstract: {row['Abstract']} "
            f"Keywords: {row['Author Keywords']} {row['Keywords Plus']}, "
            f"Research Area: {row['Research Areas']}"
        )

    df_filled = df.fillna('')
    df_filled['combined_text'] = df_filled.apply(combine_fields, axis=1)

    print("🔠 Tokenizing individual fields...")
    tokenized_title = tokenizer(df_filled['Article Title'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_abstract = tokenizer(df_filled['Abstract'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_keywords = tokenizer(df_filled['Author Keywords'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_keywords_plus = tokenizer(df_filled['Keywords Plus'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    tokenized_research_area = tokenizer(df_filled['Research Areas'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    print("🔗 Tokenizing combined field...")
    tokenized_combined = tokenizer(df_filled['combined_text'].tolist(), padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    print("\n📏 Token shapes:")
    print("Title:", tokenized_title['input_ids'].shape)
    print("Abstract:", tokenized_abstract['input_ids'].shape)
    print("Author Keywords:", tokenized_keywords['input_ids'].shape)
    print("Keywords Plus:", tokenized_keywords_plus['input_ids'].shape)
    print("Research Areas:", tokenized_research_area['input_ids'].shape)
    print("Combined:", tokenized_combined['input_ids'].shape)

    print('\n🔍 Sample tokenized title input_ids:')
    print(tokenized_title['input_ids'][0])

    return {
        'tokenized': {
            'title': tokenized_title,
            'abstract': tokenized_abstract,
            'author_keywords': tokenized_keywords,
            'keywords_plus': tokenized_keywords_plus,
            'research_areas': tokenized_research_area,
            'combined': tokenized_combined,
        },
        'original_text': df_filled[['Article Title', 'Abstract', 'Author Keywords', 'Keywords Plus', 'Research Areas', 'combined_text']]
    }

def encode_tokenized_inputs_with_text(
    tokenized_output_dict, field="combined", model_name="bert-base-uncased",
    use_cls=True, batch_size=8
):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"\n📤 Encoding field: '{field}' using model: {model_name}")
    print(f"⚙️  Using device: {device}")

    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    tokenized_inputs = tokenized_output_dict['tokenized'][field]
    original_text = tokenized_output_dict['original_text']
    input_length = tokenized_inputs['input_ids'].shape[0]

    relevant_text_column = field if field in original_text.columns else 'combined_text'
    text_df = original_text[[relevant_text_column]].copy().reset_index(drop=True)

    all_embeddings = []

    for i in range(0, input_length, batch_size):
        batch = {k: v[i:i+batch_size].to(device) for k, v in tokenized_inputs.items()}

        with torch.no_grad():
            outputs = model(**batch)

        if use_cls:
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            batch_embeddings = outputs.pooler_output

        all_embeddings.append(batch_embeddings.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    return embeddings, text_df