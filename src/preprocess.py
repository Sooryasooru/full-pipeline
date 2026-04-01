from sklearn.preprocessing import StandardScaler

def scale_data(X_train,X_test):
    scaler = StandardScaler()
    X_train_scaler = scaler.fit_transform(X_train)
    X_test_scaler = scaler.fit_transform(X_test)
    return X_train,X_test

