import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

adult_census = pd.read_csv("datasets/adult-census.csv")

numerical_columns = [
    "age",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name])

data_numeric = adult_census[numerical_columns]

# Divide the data into a training and testing set
data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42
)

#print(data_train.describe())

scaler = StandardScaler()

scaler.fit(data_train)

print(scaler.mean_)
print(scaler.scale_)

# Apply the scaling to the training data
data_train_scaled = scaler.transform(data_train)
print(data_train_scaled)

data_train_scaled = scaler.fit_transform(data_train)
print(data_train_scaled)

