import re
import string
import pandas as pd
import nltk

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

STOP_WORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str) or len(text) == 0:
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def remove_stopwords(text: str) -> str:
    """Remove stopwords from text."""
    if not text:
        return ""
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return ' '.join(tokens)

def lemmatize_text(text: str) -> str:
    """Lemmatize text."""
    if not text:
        return ""
    tokens = text.split()
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

def full_preprocess(text: str) -> str:
    """Full preprocessing pipeline."""
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text

def preprocess_dataframe(df: pd.DataFrame, text_cols: list) -> pd.DataFrame:
    """Preprocess multiple text columns in a dataframe."""
    print(f"📝 Preprocessing text columns: {text_cols}")

    for col in text_cols:
        if col in df.columns:
            clean_col = f"{col}_clean"
            df[clean_col] = df[col].fillna('').apply(full_preprocess)
            print(f"  ✅ Processed column: {col} → {clean_col}")
        else:
            print(f"  ⚠️ Column not found: {col}")

    print("✅ Text preprocessing done!")
    return df

if __name__ == "__main__":
    # Quick test
    sample = "This is an Amazing movie! It's full of action & adventure. Visit http://movie.com"
    print("Original:", sample)
    print("Cleaned:", clean_text(sample))
    print("No stopwords:", remove_stopwords(clean_text(sample)))
    print("Lemmatized:", full_preprocess(sample))