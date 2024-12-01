import nltk
import os

# Define the path to your custom 'data' folder
nltk_data_dir = os.path.join(os.getcwd(), 'data', 'nltk_data')

# Make sure the folder exists, create it if necessary
os.makedirs(nltk_data_dir, exist_ok=True)

# Download the 'punkt' tokenizer data to the 'data/nltk_data' folder
nltk.download('punkt', download_dir=nltk_data_dir)

# Verify if 'punkt' is available in the specified directory
print(nltk.data.find('tokenizers/punkt'))
from nltk.tokenize import word_tokenize
text = "Hello, world! How are you today?"
tokens = word_tokenize(text)
print(tokens)
