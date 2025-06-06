import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score
import numpy as np
from datetime import datetime
import re
from unidecode import unidecode
from difflib import SequenceMatcher


def load_dataset():
    df = pd.read_parquet('results.pq')
    df['temps'] = pd.to_timedelta(df['temps'],errors='coerce')
    df['date'] = pd.to_datetime(df['date'],format='%d/%m/%y',errors='coerce')
    df = df.dropna(subset=['temps', 'date'])
    df['temps'] = (df['temps'].dt.total_seconds() / 3600).round(2)
    return df


def normalize(text):
    text = unidecode(text).lower()
    text = re.sub(r'[^a-z0-9 ]', '', text)
    return text


def similar(a, b, threshold=0.8):
    """Retourne True si a et b sont suffisamment similaires."""
    return SequenceMatcher(None, a, b).ratio() >= threshold

def count_matching_words(query_words, target_words):
    """Compte combien de mots de la requête sont similaires à ceux de la cible."""
    return sum(any(similar(q, t) for t in target_words) for q in query_words)

def find_best_matching_course(df, course_query):
    """Trouve le nom de course le plus similaire en fonction des mots présents."""
    course_names = df['course'].dropna().unique()
    norm_query_words = normalize(course_query).split()

    best_score = -1
    best_match = None

    for course_name in course_names:
        norm_course_words = normalize(course_name).split()
        score = count_matching_words(norm_query_words, norm_course_words)

        if score > best_score:
            best_score = score
            best_match = course_name

    return best_match


def get_prediction_v2(name,distance,denivele, utmb_index):
    df = load_dataset()
    df.utmb_index = df.utmb_index.astype(int)
    df = df[(df.distance.astype(float)>=distance*0.95) & (df.distance.astype(float)<=distance*1.05)]
    df = df[(df.denivele.astype(float)>=denivele*0.95) & (df.denivele.astype(float)<=denivele*1.05)]
    best_course = find_best_matching_course(df, name)
    if best_course is None:
        return None,None,None, None

    df_course = df[df['course'] == best_course]
    distance_found = int(df_course.distance.astype(float).mean())
    denivele_found = int(df_course.denivele.astype(float).mean())

    min_index = utmb_index * 0.9
    max_index = utmb_index * 1.1
    df_filtered = df_course[(df_course['utmb_index'].notna()) & (df_course['utmb_index'] >= min_index) & (df_course['utmb_index'] <= max_index)]

    if df_filtered.empty:
        return None,None,None, None

    df_filtered['poids'] = 1 / (1 + abs(df_filtered['utmb_index'] - utmb_index))
    predicted_time = (df_filtered['temps'] * df_filtered['poids']).sum() / df_filtered['poids'].sum()
    return best_course,distance_found,denivele_found, round(predicted_time, 2)


def alpha_from_half_life(days):
    return np.log(2) / days

def get_prediction(name,utmb_index,distance,denivele,races,test_size=0,weight_runner=1e1):
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
