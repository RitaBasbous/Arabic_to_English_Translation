# Arabic-English Neural Machine Translation

A neural machine translation project that fine-tunes the mBART model for Arabic to English translation using parallel sentence pairs.

## Overview

This project implements a sequence-to-sequence translation model using Facebook's mBART (Multilingual BART) pre-trained model. The model is fine-tuned on Arabic-English parallel corpus to perform high-quality translation from Arabic to English.

## Model

- **Base Model**: `facebook/mbart-large-50-many-to-many-mmt`
- **Architecture**: mBART (Multilingual Bidirectional and Auto-Regressive Transformers)
- **Task**: Arabic (ar_AR) to English (en_XX) translation
- **Training Approach**: Fine-tuning pre-trained multilingual model on domain-specific data

## Requirements

```bash
pip install transformers
pip install datasets
pip install pandas
pip install torch  # or tensorflow
```

### Key Dependencies

- `transformers`: Hugging Face Transformers library for the mBART model
- `datasets`: Hugging Face Datasets library for data handling
- `pandas`: Data manipulation and processing
- `torch` or `tensorflow`: Deep learning framework

## Data Format

The training data should be in a tab-separated format in a file named `ara.txt`:

```
English sentence 1\tArabic sentence 1
English sentence 2\tArabic sentence 2
...
```

**Structure:**
- Column 1: English text (target language)
- Column 2: Arabic text (source language)
- Separator: Tab character (`\t`)
- Encoding: UTF-8

## Project Structure

```
├── main.ipynb          # Main training notebook
├── ara.txt             # Parallel Arabic-English corpus (not included)
└── README.md           # This file
```

## Usage

### 1. Data Preparation

Ensure your parallel corpus file `ara.txt` is in the correct format:
- Tab-separated values
- UTF-8 encoding
- One sentence pair per line

### 2. Running the Notebook

Open `main.ipynb` in Jupyter Notebook or Google Colab and run the cells sequentially:

1. **Import Libraries**: Load required packages
2. **Load Data**: Read and preprocess the parallel corpus
3. **Load Model**: Initialize mBART model and tokenizer
4. **Tokenize Data**: Prepare data for training
5. **Train Model**: Fine-tune the model on your data
6. **Evaluate**: Test the model performance

### 3. Key Steps in the Pipeline

**Data Loading:**
```python
# Loads parallel sentences from ara.txt
# Splits into training (90%) and test (10%) sets
# Creates a Hugging Face Dataset object
```

**Tokenization:**
```python
# Tokenizes both source (Arabic) and target (English) texts
# Max sequence length: 128 tokens
# Padding: max_length
# Truncation: enabled
```

**Model Configuration:**
```python
# Source language: ar_AR (Arabic)
# Target language: en_XX (English)
# Pre-trained model handles 50 languages
```

## Training Configuration

The model uses:
- `Seq2SeqTrainer` from Hugging Face Transformers
- `DataCollatorForSeq2Seq` for dynamic batching
- Train/test split: 90/10
- Maximum sequence length: 128 tokens

## Model Output

After training, the model can translate Arabic text to English. The fine-tuned model weights and tokenizer configuration will be saved for inference.

## Notes

- The model is pre-trained on 50 languages, making it suitable for multilingual tasks
- Fine-tuning on domain-specific data improves translation quality
- Adjust hyperparameters based on your dataset size and computational resources
- Consider using GPU acceleration for faster training

## Example Translation

```python
# After training, use the model for translation:
model.eval()
tokenizer.src_lang = "ar_AR"
inputs = tokenizer("نص عربي", return_tensors="pt")
translated = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"])
translation = tokenizer.decode(translated[0], skip_special_tokens=True)
```

## License

Please refer to the licenses of the respective libraries and models used:
- [Hugging Face Transformers](https://github.com/huggingface/transformers/blob/main/LICENSE)
- [mBART Model License](https://github.com/facebookresearch/fairseq/tree/main/examples/mbart)

## Acknowledgments

- Facebook AI Research for the mBART model
- Hugging Face for the Transformers library
- The creators of the Arabic-English parallel corpus
