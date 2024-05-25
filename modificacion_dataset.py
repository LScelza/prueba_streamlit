import pandas as pd

df = pd.read_parquet('Dataset/recomendacion3.parquet')

df = df.sample(20)

df.to_parquet('Dataset/recomendacion3_v1.parquet')