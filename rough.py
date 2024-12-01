import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

text = "Hello, world! How are you today?"
tokens = word_tokenize(text)
print(tokens)
