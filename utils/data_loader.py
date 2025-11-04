import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path):
    df = pd.read_csv(path)
    
    # Encoding etichette
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['label'])

    X = df.drop('label', axis=1)
    y = df['label']

    # Standardizzazione
    X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, random_state=111)

    # One-hot encoding
    y_train = pd.get_dummies(y_train)
    y_val = pd.get_dummies(y_val)
    y_test = pd.get_dummies(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test
