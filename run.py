from src.data_preprocessing import pre_process_data
from src.data_balancing import balance_data_oversample, balance_data_undersample

data = pre_process_data('./data/raw_sms.csv')
balanced_data = balance_data_undersample(data)