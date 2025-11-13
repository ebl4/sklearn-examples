import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

adult_census = pd.read_csv("datasets/adult-census.csv")
#print(adult_census)

target_column = "class"
# print(adult_census[target_column].value_counts())

# print(
#     f"The dataset contains {adult_census.shape[0]} samples and "
#     f"{adult_census.shape[1]} columns"
# )

# _ = adult_census.hist(figsize=(20, 14))
# plt.show()

# print(adult_census["sex"].value_counts())

# print(adult_census["education"].value_counts())

# crostab = pd.crosstab(
#     index=adult_census["education"], columns=adult_census["education-num"]
# )



# We plot a subset of the data to keep the plot readable and make the plotting
# faster
n_samples_to_plot = 5000
columns = ["age", "education-num", "hours-per-week"]
_ = sns.pairplot(
    data=adult_census[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)
plt.show()

