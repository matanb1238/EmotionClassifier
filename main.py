import pandas as pd

df = pd.read_csv('data/text.csv')  # or whatever the file name is
print(df.shape)               # outputs (number_of_rows, number_of_columns)
print(df['label'].value_counts())  # optional: see distribution across emotion labels