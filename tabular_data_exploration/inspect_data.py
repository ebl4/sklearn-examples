import matplotlib.pyplot as plt
import seaborn as sns

def plot_decision_boundaries(adult_census, n_samples_to_plot, target_column):
    ax = sns.scatterplot(
        x="age",
        y="hours-per-week",
        data=adult_census[:n_samples_to_plot],
        hue=target_column,
        alpha=0.5,
    )
    
    age_limit = 27
    plt.axvline(x=age_limit, ymin=0, ymax=1, color="black", linestyle="--")

    hours_per_week_limit = 40
    plt.axhline(
        y=hours_per_week_limit, xmin=0.18, xmax=1, color="black", linestyle="--"
    )

    plt.annotate("<=50K", (17, 25), rotation=90, fontsize=35)
    plt.annotate("<=50K", (35, 20), fontsize=35)
    _ = plt.annotate("???", (45, 60), fontsize=35)
    plt.show()
    
# We plot a subset of the data to keep the plot readable and make the plotting
# faster
def plot_pairplot(adult_census, n_samples_to_plot, target_column):
    sns.pairplot(
        adult_census[:n_samples_to_plot],
        vars=["age", "hours-per-week", "education-num"],
        hue=target_column,
        plot_kws={"alpha": 0.5},
        diag_kind="hist",
        diag_kws={"bins": 30},
    )
    plt.show()
    
def plot_histogram(adult_census):
    _ = adult_census.hist(figsize=(20, 14))
    plt.show()