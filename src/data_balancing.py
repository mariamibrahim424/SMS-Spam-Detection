from sklearn.utils import resample
import pandas as pd 
from .helpers import write_data

def balance_data_oversample(df):
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']

    spam_oversampled = resample(spam,replace=True, n_samples=len(ham),random_state=42)  
    
    # Combine the two classes
    balanced_df = pd.concat([ham, spam_oversampled])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    write_data(balanced_df,'./data/balanced_sms.csv','cleaned_sms')
    return balanced_df


def balance_data_undersample(df):
    ham = df[df['label'] == 'ham']
    spam = df[df['label'] == 'spam']
    
    ham_undersampled = resample(ham, replace=False, n_samples=len(spam), random_state=42)  
    
    # Combine the two classes
    balanced_df = pd.concat([ham_undersampled, spam])
    
    # Shuffle the dataset
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    write_data(balanced_df,'./data/balanced_sms.csv','cleaned_sms')
    return balanced_df
