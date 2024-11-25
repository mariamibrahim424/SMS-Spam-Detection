import pandas as pd 
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    return df

def clean_data(df):
    stop_words = set(stopwords.words('english'))
    # Drops columns if any of its valus are NaN
    df = df.dropna(axis=1, how='any')
    df = df.rename(columns={'v1': 'label', 'v2': 'sms'})
    # Convert to lowercase and remove puncuation
    df['sms'] = df['sms'].str.lower().str.translate(str.maketrans('', '', string.punctuation))
    # Tokenize and remove stopwords
    df['sms_tokens'] = df['sms'].apply(lambda x: [word for word in word_tokenize(x) if word not in stop_words])
    df['cleaned_sms'] = df['sms_tokens'].apply(lambda x: ' '.join(x))

    return df

def write_clean_data(clean_df):
    output_df = clean_df[['label', 'cleaned_sms']]
    output_df.to_csv('../data/preprocessed_sms.csv', index=False, mode='w')

def main():
    df = load_data('../data/raw_sms.csv')
    clean_df = clean_data(df)
    write_clean_data(clean_df)

if __name__ == "__main__":
    main()


