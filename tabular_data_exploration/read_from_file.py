import arff
import pandas as pd

with open("datasets/adult-census.arff", "r") as f:
    dataset = arff.load(f)

# Converta para DataFrame
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# Salve como CSV
df.to_csv("datasets/adult-census.csv", index=False)