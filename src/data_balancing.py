from sklearn.utils import resample
import pandas as pd 
from .helpers import write_data
from nltk.tokenize import word_tokenize

def balance_data_oversample(df):
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']

    # Oversample the minority class (spam)
    spam_oversampled = resample(spam, replace=True, n_samples=len(ham), random_state=42)  
    
    # Combine the two classes
    balanced_df = pd.concat([ham, spam_oversampled])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create a new column 'balanced_sms' from 'cleaned_sms'
    balanced_df['balanced_sms'] = balanced_df['cleaned_sms']
    
    # Tokenize the 'balanced_sms' column
    balanced_df['sms_tokens'] = balanced_df['balanced_sms'].apply(lambda x: [word for word in word_tokenize(x)])
    
    write_data(balanced_df, './data/balanced_sms.csv', 'balanced_sms')
    return balanced_df


def balance_data_undersample(df):
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']
    
    # Undersample the majority class (ham)
    ham_undersampled = resample(ham, replace=False, n_samples=len(spam), random_state=42)  
    
    # Combine the two classes
    balanced_df = pd.concat([ham_undersampled, spam])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Create a new column 'balanced_sms' from 'cleaned_sms'
    balanced_df['balanced_sms'] = balanced_df['cleaned_sms']
    write_data(balanced_df, './data/balanced_sms.csv', 'balanced_sms')
    return balanced_df
