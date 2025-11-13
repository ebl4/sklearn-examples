import pandas as pd
import matplotlib.pyplot as plt
import tabular_data_exploration.inspect_data as ins

adult_census = pd.read_csv("datasets/adult-census.csv")
#print(adult_census)

target_column = "class"
# print(adult_census[target_column].value_counts())

# print(
#     f"The dataset contains {adult_census.shape[0]} samples and "
#     f"{adult_census.shape[1]} columns"
# )

# ins.plot_histogram(adult_census)

# print(adult_census["sex"].value_counts())

# print(adult_census["education"].value_counts())

# crostab = pd.crosstab(
#     index=adult_census["education"], columns=adult_census["education-num"]
# )

# ins.plot_pairplot(adult_census, n_samples_to_plot=5000, target_column=target_column)

ins.plot_decision_boundaries(adult_census, n_samples_to_plot=500, target_column=target_column)
