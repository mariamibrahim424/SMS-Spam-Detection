import pandas as pd 

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1')
    return df

def write_data(df,filepath, sms_type):
    output_df = df[['label', sms_type]]
    output_df.to_csv(filepath, index=False, mode='w')
    print(f"Data written to {filepath}")
