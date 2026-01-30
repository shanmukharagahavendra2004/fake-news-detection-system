import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



STOPWORDS=set(stopwords.words('english'))
lemmatizer=WordNetLemmatizer()

#cleaning function
def clean_text(text: str) -> str:
    """
    Perform NLP preprocessing on input text:
    - Lowercasing
    - Remove punctuation
    - Remove numbers
    - Remove stopwords
    - Lemmatization
    """

    if not isinstance(text,str):
        return ""
    
    # 1. Lowercase
    text=text.lower()

    # 2. Remove URLs
    text=re.sub(r"http\S+|www\S+|https\S+","",text)

    # 3. Remove punctuation
    text=text.translate(str.maketrans("","",string.punctuation))

    # 4. Remove numbers
    text=re.sub(r"\d+","",text)

    # 5. Tokenization
    tokens=text.split()

    # 6. Remove stopwords and lemmatization
    cleaned_tokens=[
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in STOPWORDS and len(word)>2
    ]

    cleaned_text=" ".join(cleaned_tokens)

    return cleaned_text

if __name__=="__main__":
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    sample_text="""
    Breaking News!!! The government announced new policies today.
    Read more at https://news.example.com
    """

    print("Original Text:\n",sample_text)
    print("\nCleaned Text:\n",clean_text(sample_text))