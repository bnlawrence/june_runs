import pandas as pd

run_numbers = [58, 112, 135, 185]

bert = pd.read_csv('../../parameters/bert_parameters.csv')

bert.iloc[run_numbers].to_csv('parameters.csv', index=False)
