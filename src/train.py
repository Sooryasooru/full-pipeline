from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import scale_data

from prometheus_client import start_http_server, Gauge
import joblib
import time

accuracy_metric = Gauge('model_accuracy', 'Model accuracy')

def train_model():
    # load dataset from sklearn
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    X_train, X_test = scale_data(X_train, X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    joblib.dump(model, "model.pkl")

    return acc


if __name__ == "__main__":
    start_http_server(8000)

    acc = train_model()
    accuracy_metric.set(acc)

    print("Accuracy:", acc)

    while True:
        time.sleep(5)