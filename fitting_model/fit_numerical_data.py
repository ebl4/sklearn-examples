import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

adult_census = pd.read_csv("datasets/adult-census.csv")

numerical_columns = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

target_name = "class"
adult_census = adult_census[numerical_columns + [target_name]]

target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])

model = KNeighborsClassifier()
_ = model.fit(data, target)

target_pred = model.predict(data)

# print(target_pred[:5])
# print(target[:5])

# print(target[:5] == target_pred[:5])

print((target == target_pred).mean())
