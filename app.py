import streamlit as st
import pandas as pd
import io
from optimizer_script import generator,charger_trace,formater_duree,to_dict
from prediction import get_prediction_v2,get_prediction_utmb_index
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import json
from calendar_script import generate_ics

st.set_page_config(page_title="Nutrition Trail", layout="wide")
st.title("Optimisation Nutrition Trail")

# === 0. Estimation de la durée de course ===

#st.subheader("Données personnelles et estimation du temps")

col1, col2 = st.columns(2)

with col1:
    predict_time = st.checkbox("🔮 Estimer automatiquement le temps cible", value=True)
with col2:
    predict_utmb_index = st.checkbox("🔮 Estimer automatiquement l'utmb index", value=True)


col3, col4, col5, col6,col7 = st.columns(5)
with col3:
    name = st.text_input("Nom de la course", value="Andorra")
with col4:
    predicted_distance = st.number_input("Distance de la course (km)", min_value=10.0, max_value=1000.0, value=80.0)
with col5:
    predicted_dplus = st.number_input("D+ de la course (m)", min_value=0, max_value=20000, value=3900)
with col6:
    race_date = st.date_input("Date de la course")
with col7:
    race_time = st.time_input("Heure de départ",value='08:00')

st.session_state["race_datetime"] = datetime.combine(race_date, race_time)


st.markdown("**Courses passées (facultatif)**")
default_past_races = pd.DataFrame([
    {"course": 'Laudon',"nom": 'Arthur S.', "distance": 42, "denivele": 2100, "temps (hh:mm:ss)": "5:17:00", "date (yyyy-mm-dd)": "2025-05-01"},
    {"course": 'SainteLyon',"nom": 'Arthur S.', "distance": 83, "denivele": 2100, "temps (hh:mm:ss)": "12:00:00", "date (yyyy-mm-dd)": "2024-12-01"},
])
past_races_df = st.data_editor(default_past_races, num_rows="dynamic", use_container_width=True)



# Préparation du DataFrame de courses passées
try:
    user_races = past_races_df.copy()
    user_races = user_races.rename(columns={'temps (hh:mm:ss)':'temps','date (yyyy-mm-dd)':'date'})
    user_races['temps'] = pd.to_timedelta(user_races['temps'])
    user_races['temps'] = (user_races['temps'].dt.total_seconds() / 3600).round(2)
    user_races['date'] = pd.to_datetime(user_races['date'])
    user_races = user_races.dropna(how='any',axis=0)
except:
    st.error("Erreur de format dans les courses passées.")
    st.stop()

#estimation utmb_index
if "utmb_index" not in st.session_state:
    st.session_state["utmb_index"] = 0

if predict_utmb_index:
    if st.button("📈 Estimer l'utmb_index"):
        st.session_state["utmb_index"] = get_prediction_utmb_index(user_races)

    utmb_index = st.number_input(
        "UTMB Index prédit",
        min_value=0,
        max_value=1000,
        value=st.session_state["utmb_index"]
    )
else:
    utmb_index = st.number_input("UTMB Index", min_value=0, max_value=1000, value=547)

user_races['utmb_index'] = utmb_index



# Initialisation dans session_state si pas encore défini
if 'estimated_time' not in st.session_state:
    st.session_state['estimated_time'] = 0.0

if 'course_found' not in st.session_state:
    st.session_state['course_found'] = ''

if 'distance_found' not in st.session_state:
    st.session_state['distance_found'] = 0.0

if 'denivele_found' not in st.session_state:
    st.session_state['denivele_found'] = 0.0

if predict_time:
    if st.button("📈 Estimer le temps"):
        try:
            course_found,distance_found,denivele_found, estimation = get_prediction_v2(
                name=name,
                distance=predicted_distance,
                denivele=predicted_dplus,
                utmb_index=utmb_index,
            )
            st.session_state['estimated_time'] = estimation
            st.session_state['course_found'] = course_found
            st.session_state['distance_found'] = distance_found
            st.session_state['denivele_found'] = denivele_found

        except Exception as e:
            st.error(f"Erreur dans la prédiction : {e}")

# Affichage de la valeur persistante même sans re-estimation
st.write(f"{st.session_state['course_found']} - {st.session_state['distance_found']}km - {st.session_state['denivele_found']}D+ Temps estimé actuel : **{formater_duree(st.session_state['estimated_time'])} ({st.session_state['estimated_time']:.2f}h)**")



# === 1. Chargement du fichier GPX ===
gpx_file = st.file_uploader("📁 Télécharger le fichier GPX")#, type=["gpx"])
if gpx_file:
    # Charger le fichier en buffer temporaire (car charger_trace attend un chemin)
    # On crée un buffer temporaire en mémoire pour simuler un fichier sur disque
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gpx") as tmp:
        tmp.write(gpx_file.getvalue())
        tmp_path = tmp.name

    # Utilisation de charger_trace
    gpx = charger_trace(tmp_path)

else:
    #gpx = charger_trace('andorra_80.gpx')
    st.warning("Veuillez charger un fichier GPX pour continuer.")
    st.stop()



# === 2. Paramètres de course ===
st.subheader("Paramètres de la course")

col1, col2, col3, col4 = st.columns(4)
with col1:
    TEMPS_CIBLE_TOTAL_HEURES = st.number_input("Temps cible total (heures)", min_value=0.0, value=float(round(st.session_state['estimated_time'], 1)), step=0.5)
with col2:
    NOMBRE_FLASQUES = st.number_input("Nombre de flasques", min_value=1, value=2)
with col3:
    GLUCIDES_CIBLES_G_H = st.number_input("Glucides cibles (g/h)", min_value=20, max_value=150, value=90)
with col4:
    CAFEINE_MAX = st.number_input("Caféine totale max (mg)", min_value=0, value=400)

MONTEE = st.slider("Score montée (1 = excellent ; 10 = estimation moyenne ; 20 = faible)", 1, 20, 10)
DESCENTE = st.slider("Score descente (1 = excellent ; 2 = estimation moyenne ; 3 = faible)", 1, 3, 2)
FATIGUE_MAX = st.slider("Fatigue en fin de course (0 = pas de fatigue ; 1 = extrême)", 0.0, 1.0, 0.3)

# === 3. Points de ravitaillement ===
default_ravitos = "14.7, 20.9, 30.8, 41, 48.8, 58.7, 69.7"
ravitos_input = st.text_input("Kilomètres des ravitaillements (ex: 14.7, 20.9, 30.8, 41, 48.8, 58.7,69.7)" ,value=default_ravitos)
try:
    RAVITOS_KM = [float(km.strip()) for km in ravitos_input.split(",") if km.strip()]
    RAVITOS_KM = [0] + RAVITOS_KM + [gpx.distance.max().round(2)]
except:
    st.error("Format des ravitos incorrect (utiliser des nombres séparés par des virgules.")
    st.stop()

# === 4. Ingrédients à saisir manuellement ===
st.subheader("Ingrédients disponibles")

default_data = [
    {"nom": "puree", "marque" : "naak","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":True},
    {"nom": "gel", "marque" : "maurten","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":True},
    {"nom": "gel_caf", "marque" : "maurten","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":100,"preference":1.0,"presence":True},
    {"nom": "boisson", "marque" : "naak", "glucides": 55, "volume_ml": 500, "categorie": "liquide", "cafeine":0,"preference":1.0,"presence":True},
    {"nom": "pate_de_fruits", "marque" : "decathlon","glucides": 20, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False},
    {"nom": "barre_amande", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False},
    {"nom": "barre_nougat", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False}
]
st.info('Remarque : preference entre 0 et 1')

ingredients_df = st.data_editor(
    pd.DataFrame(default_data),
    num_rows="dynamic",
    use_container_width=True
)

# Convertir le dataframe en liste de dictionnaires
try:
    INGREDIENTS = ingredients_df.to_dict(orient="records")
    INGREDIENTS = [ingredient for ingredient in INGREDIENTS if ingredient.get("presence", False)]

except:
    st.error("Erreur lors de la conversion des ingrédients.")
    st.stop()

st.info('Remarque : la seed permet de contrôler le caractère aléatoire du générateur de stratégie nutritionnelle, libre à toi de mettre un nombre entier positif différent pour générer un rapport différent')
SEED = st.number_input("Seed", value=0)


# === 5. Bouton pour lancer l'analyse ===
if 'results' not in st.session_state:
    st.session_state['results'] = pd.DataFrame(dtype=float)

if st.button("🚀 Générer le plan nutritionnel"):
    with st.spinner("Calcul en cours..."):
        results, plan = generator(
            gpx,
            RAVITOS_KM,
            TEMPS_CIBLE_TOTAL_HEURES,
            MONTEE,
            DESCENTE,
            FATIGUE_MAX,
            INGREDIENTS,
            NOMBRE_FLASQUES,
            GLUCIDES_CIBLES_G_H,
            CAFEINE_MAX,
            SEED
        )
        st.session_state["results"] = results

    st.success(f"✅ Plan nutritionnel généré avec succès ! (seed : {SEED})")

    # Résumé
    resume_data = {
        "Distance (km)": [results.km_fin.iloc[-1]],
        "D+ (m)": [results["D+"].sum()],
        "D- (m)": [results["D-"].sum()],
        "Durée (h)": [TEMPS_CIBLE_TOTAL_HEURES],
        "Glucides (g/h)": [np.round(results.glucides_total.sum() / TEMPS_CIBLE_TOTAL_HEURES, 1)],
        "Boisson (mL/h)": [int(results.flasques.sum() * 500 / TEMPS_CIBLE_TOTAL_HEURES)],
        "Caféine (mg)": [int(plan.cafeine.sum())],
    }
    df_resume = pd.DataFrame(resume_data)
    st.session_state["df_resume"] = df_resume
    # Total ingrédients solides
    total_ingredients = Counter()
    for d in results["ingrédients_solides"]:
        total_ingredients.update(d)
    df_ingredients = pd.DataFrame(list(total_ingredients.items()), columns=["Ingrédient", "Quantité"])
    df_ingredients = pd.concat([df_ingredients.T, pd.Series(['Boisson', results.flasques.sum()], index=['Ingrédient', 'Quantité'])], axis=1).T.reset_index(drop=True)
    st.session_state["df_ingredients"] = df_ingredients

    # Affichage résumé et ingrédients
    st.subheader("Résumé de la course")
    st.dataframe(df_resume)

    st.subheader("Ingrédients totaux")
    st.dataframe(df_ingredients)

    # Profil altitude + lignes ravitos
    st.subheader("Profil d'altitude et ravitaillements")
    fig, ax = plt.subplots(figsize=(7, 2))
    gpx.set_index('distance').altitude.plot(ax=ax)
    for rvt in RAVITOS_KM:
        ax.axvline(x=rvt, color='red', alpha=0.25)
    ax.grid(True)
    ax.set_title('Profil')
    st.pyplot(fig)
    st.session_state["fig"] = fig

    # Tableau plan nutrition par segment
    st.subheader("Plan nutritionnel par segment")
    results['timing'] = results['timing'].str.replace('<br>', '\n')
    st.dataframe(results[['km_debut', 'km_fin', 'D+', 'D-', 'durée', 'durée_cumulée', 'flasques', 'timing']], use_container_width=True)


st.session_state["params"] = {
    "TEMPS_CIBLE_TOTAL_HEURES": TEMPS_CIBLE_TOTAL_HEURES,
    "NOMBRE_FLASQUES": NOMBRE_FLASQUES,
    "GLUCIDES_CIBLES_G_H": GLUCIDES_CIBLES_G_H,
    "CAFEINE_MAX": CAFEINE_MAX,
    "MONTEE": MONTEE,
    "DESCENTE": DESCENTE,
    "FATIGUE_MAX": FATIGUE_MAX,
    "SEED": SEED,
    "RAVITOS_KM": RAVITOS_KM,
}

if len(st.session_state["results"])>0:
    dict_data = to_dict(st.session_state["results"])
    ics_content = generate_ics(dict_data, pd.Timestamp(st.session_state["race_datetime"]))
    if st.download_button(label="Exporter le calendrier",data=ics_content,file_name="nutrition_plan.ics"):
        st.success(f"✅ Calendrier téléchargé avec succès !")

st.info("💡 Astuce : Pense à imprimer cette page!")

