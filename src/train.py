from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import scale_data

from prometheus_client import start_http_server, Gauge
import joblib
import time
import os

accuracy_metric = Gauge('model_accuracy', 'Model accuracy')

def train_model():
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

    # ✅ run infinite loop only in local, not in CI
    if os.getenv("CI") != "true":
        while True:
            time.sleep(5)














from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from src.preprocessing import scale_data
import joblib


def train_model():
    X,y = load_iris(return_X_y=True)
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
    X_test,X_train = scale_data(X_test,X_train)

    model = LogisticRegression()
    model.fit(X_train,y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test,pred)

    joblib.dump(model,"model.pkl")

    return acc


if __name__ == "__main__":
    acc = train_model()
    print("accurayc",acc)