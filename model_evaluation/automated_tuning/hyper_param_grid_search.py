
from sklearn.compose import make_column_selector as selector, make_column_transformer
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
import seaborn as sns
import matplotlib.pyplot as plt

adult_census = pd.read_csv("../../datasets/adult-census.csv")

target_name = "class"
target = adult_census[target_name] # y

# Drop target and redundant columns
data = adult_census.drop(columns=[target_name, "education-num"]) # X

data_train, data_test, target_train, target_test = \
    train_test_split(data, target, random_state=42)
    
# Identify categorical columns
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)

# Build OrdinalEncoder by passing it the known categories.
categorical_preprocessor = OrdinalEncoder(
    handle_unknown="use_encoded_value", unknown_value=-1
)

# Use make_column_transformer to select the categorical 
# columns and apply the OrdinalEncoder to them.
preprocessor = make_column_transformer(
    (categorical_preprocessor, categorical_columns),
    remainder="passthrough",
    # Silence a deprecation warning in scikit-learn v1.6 related to how the
    # ColumnTransformer stores an attribute that we do not use in this notebook
    force_int_remainder_cols=False,
)

# Use a tree-based classifier (i.e. histogram gradient-boosting) 
# to predict whether or not a person earns more than 50 k$ a year.
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        (
            "classifier",
            HistGradientBoostingClassifier(random_state=42, max_leaf_nodes=4),
        ),
    ]
)
model

# Define the hyperparameter grid to search over
param_grid = {
    "classifier__learning_rate": (0.01, 0.1, 1, 10),  # 4 possible values
    "classifier__max_leaf_nodes": (3, 10, 30),  # 3 possible values
}  # 12 unique combinations
model_grid_search = GridSearchCV(model, param_grid=param_grid, n_jobs=2, cv=2)
model_grid_search.fit(data_train, target_train)

print(f"The best set of parameters is: {model_grid_search.best_params_}")

# model_grid_search.predict(data_test.iloc[0:5])

# Evaluate the model on the test set
accuracy = model_grid_search.score(data_test, target_test)
print(
    f"The test accuracy score of the grid-search pipeline is: {accuracy:.2f}"
)

# Extract and sort the cross-validation results
cv_results = pd.DataFrame(model_grid_search.cv_results_).sort_values(
    "mean_test_score", ascending=False
)

# get the parameter names
column_results = [f"param_{name}" for name in param_grid.keys()]
column_results += ["mean_test_score", "std_test_score", "rank_test_score"]
cv_results = cv_results[column_results]

# shorten the parameter names for better readability
def shorten_param(param_name):
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name

# rename the parameter columns
cv_results = cv_results.rename(shorten_param, axis=1)

# pivot the cv_results DataFrame for better visualization
pivoted_cv_results = cv_results.pivot_table(
    values="mean_test_score",
    index=["learning_rate"],
    columns=["max_leaf_nodes"],
)

# print(pivoted_cv_results)

# Visualize the results using a heatmap
ax = sns.heatmap(
    pivoted_cv_results,
    annot=True,
    cmap="YlGnBu",
    vmin=0.7,
    vmax=0.9,
    cbar_kws={"label": "mean test accuracy"},
)
ax.invert_yaxis()
plt.show()