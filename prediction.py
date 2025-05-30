import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
from datetime import datetime

def load_dataset():
    df = pd.read_parquet('results.pq')
    df['temps'] = pd.to_timedelta(df['temps'])
    df['temps'] = (df['temps'].dt.total_seconds() / 3600).round(2)
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%y')
    return df

def alpha_from_half_life(days):
    return np.log(2) / days

def get_prediction(name,utmb_index,distance,denivele,races,test_size=0,weight_runner=1e4):
    alpha = alpha_from_half_life(365)
    df = load_dataset()
    if races is not None:
        races['live'] = True
        df = pd.concat([df,races]).drop_duplicates(subset=['course','nom','date'],keep='last')
        #df.to_parquet('results.pq')
    df = df.dropna(how='any',axis=0)
    df["date"] = pd.to_datetime(df["date"])
    days_since = (datetime.now() - df["date"]).dt.days
    temporal_weights = np.array(np.exp(-alpha * days_since))
    weights_runner = np.where(df["nom"] == name, weight_runner, 1.0)
    weights = weights_runner * temporal_weights
    X = df[["distance", "denivele","utmb_index"]]
    y = df["temps"]
    if test_size>0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, y_train = X, y

    model = LinearRegression()
    model.fit(X_train, y_train,sample_weight=weights[X_train.reset_index(drop=True).index])


    if test_size>0:
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ RMSE : {rmse:.2f} h")
        print(f"✅ R² : {r2:.2f}")
    
    X_oracle = pd.DataFrame([[distance, denivele,utmb_index]], columns=["distance", "denivele", "utmb_index"])
    predicted_time = model.predict(X_oracle)[0]
    return predicted_time

def get_prediction_utmb_index(races,test_size=0):
    alpha = alpha_from_half_life(365)
    df = load_dataset()
    df = df.dropna(how='any',axis=0)
    df["date"] = pd.to_datetime(df["date"])
    days_since = (datetime.now() - df["date"]).dt.days
    weights = np.array(np.exp(-alpha * days_since))
    X = df[["distance", "denivele","temps"]]
    y = df["utmb_index"]

    if test_size>0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    else:
        X_train, y_train = X, y

    model = LinearRegression()
    model.fit(X_train, y_train,sample_weight=weights[X_train.reset_index(drop=True).index])

    if test_size>0:
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"✅ RMSE : {rmse:.2f}")
        print(f"✅ R² : {r2:.2f}")
    
    predicted_score = model.predict(races[["distance", "denivele","temps"]])
    return int(np.mean(predicted_score))
