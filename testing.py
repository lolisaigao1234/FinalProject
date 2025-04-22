import pandas as pd

data = pd.read_parquet("F:\PythonProject\IS567FP\cache\parquet\SNLI_train_features_lexical_syntactic_sample80.parquet")

print(data.values.tolist())