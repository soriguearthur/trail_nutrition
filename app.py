import streamlit as st
import gpxpy
import pandas as pd
import io
from optimizer_script import generator,charger_trace
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Nutrition Trail", layout="wide")
st.title("Optimisation Nutrition Trail")

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
    TEMPS_CIBLE_TOTAL_HEURES = st.number_input("Temps cible total (heures)", min_value=1.0, value=12.0, step=0.5)
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
default_ravitos = "14.7, 20.9, 30.8, 41, 48.8, 58.7, 69.7, 77.9"
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
    {"nom": "gel caf", "marque" : "maurten","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":100,"preference":1.0,"presence":True},
    {"nom": "boisson", "marque" : "naak", "glucides": 55, "volume_ml": 500, "categorie": "liquide", "cafeine":0,"preference":1.0,"presence":True},
    {"nom": "pate de fruits", "marque" : "decathlon","glucides": 20, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False},
    {"nom": "barre amande", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False},
    {"nom": "barre nougat", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1.0,"presence":False}
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

st.info("💡 Astuce : Pense à imprimer cette page!")
