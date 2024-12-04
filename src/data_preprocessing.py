import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from .helpers import load_data

def remove_non_ascii(text):
    return ''.join([char for char in text if ord(char) < 128])

def clean_data(df):
    stop_words = set(stopwords.words('english'))
    # Drops columns if any of its valus are NaN
    df = df.dropna(axis=1, how='any')
    df = df.rename(columns={'v1': 'label', 'v2': 'sms'})
    # Convert to lowercase and remove puncuation
    df['cleaned_sms'] = df['sms'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
    # Remove non-ASCII characters
    df['cleaned_sms'] = df['cleaned_sms'].apply(remove_non_ascii)
    # Tokenize and remove stopwords
    df['cleaned_sms'] = df['cleaned_sms'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))
    # print(df['label'].value_counts())
    return df

def pre_process_data(filepath):
    df = load_data(filepath)
    clean_df = clean_data(df)
    return clean_df


