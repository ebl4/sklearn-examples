from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

# create a pipeline object
pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
)

# load the iris dataset and split it into train and test sets
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# fit the hole pipeline
pipe.fit(X_train, y_train)

# evaluate the model
print(accuracy_score(pipe.predict(X_test), y_test))