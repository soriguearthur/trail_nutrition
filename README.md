# Nutrition Trail - Générateur de stratégie nutritionnelle pour ultratrail


**Nutrition Trail** est une application open source permettant de générer une stratégie nutritionnelle personnalisée pour les coureurs d’ultratrail et trail longue distance.  
À partir d’une trace GPS au format GPX et de paramètres utilisateur (durée estimée, besoins en glucides par heure, ravitaillements, etc.), l’application calcule un plan précis de nourriture et boisson à consommer, réparti par heure et entre les points de ravitaillement.  

Idéal pour optimiser vos apports énergétiques et hydriques, améliorer la gestion de votre course, et éviter les coups de pompe ou troubles digestifs.

---

## 🎯 Fonctionnalités principales

- Import de trace GPX pour analyser le parcours  
- Paramétrage facile : durée, besoin glucidique, nombre et localisation des ravitaillements  
- Calcul automatique des apports nutritionnels (glucides, caféine, hydratation) par heure et segment entre deux ravitaillements. Cela est effectué via un algorithme d'optimisation sous contrainte réalisée pour chaque heure.
- Visualisation claire du plan nutritionnel sous forme de tableau et graphiques  

---

## 💡 Pourquoi utiliser Nutrition Trail ?

- Planification précise et adaptée à votre parcours et rythme prévu  
- Gain de temps par rapport à une planification manuelle  
- Optimisation de la performance et prévention des désagréments liés à la nutrition  
- Solution libre, transparente et évolutive
- 
- 

---

## Aller plus loin?

- Bientôt un estimateur de durée de course (en fonction de chrono sur le plat mais idéalement en fonction d'un utmb/itra index)
- Prise en compte de la météo
- Optimisation plus poussée en fonction du dénivelé
- 
- Si vous voulez que l'on aille plus loin dans ce projet pour pousser l'optimisation à son extrême, je suis très ouvert à échanger sur un projet à GRANDE ECHELLE (mon mail : arthur.sorigue@gmail.com)


---
## Exemple d'utilisation sur le web

- aller sur : https://nutrition-trail.streamlit.app/
  
---

## Exemple d'utilisation en Python  

from IPython.display import display, HTML  


GLUCIDES_CIBLES_G_H = 90  
NOMBRE_FLASQUES = 2  
TEMPS_CIBLE_TOTAL_HEURES = 12  
CAFEINE_MAX = 400  
MONTEE = 10   
DESCENTE = 2  
FATIGUE_MAX = 0.3  
SEED = 42   

INGREDIENTS = [  
    {"nom": "puree", "marque" : "naak","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":.3,"presence":True},  
    {"nom": "gel", "marque" : "maurten","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1,"presence":True},  
    {"nom": "gel caf", "marque" : "maurten","glucides": 25, "volume_ml": None, "categorie": "solide", "cafeine":100,"preference":1,"presence":True},  
    {"nom": "boisson", "marque" : "naak", "glucides": 55, "volume_ml": 500, "categorie": "liquide", "cafeine":0,"preference":1,"presence":True},  
    {"nom": "pate de fruits", "marque" : "decathlon","glucides": 20, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1,"presence":False},  
    {"nom": "barre amande", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1,"presence":False},  
    {"nom": "barre nougat", "marque" : "decathlon","glucides": 14, "volume_ml": None, "categorie": "solide", "cafeine":0,"preference":1,"presence":False}  
]  
INGREDIENTS = [ingredient for ingredient in INGREDIENTS if ingredient.get("presence", False)]  

RAVITOS_KM = [14.7, 20.9, 30.8, 41, 48.8, 58.7,69.7]  

TRAIL = 'MY_TRAIL'  
GPX_PATH = 'your_gpx_file.gpx'  
gpx = charger_trace(GPX_PATH)  
RAVITOS_KM = [0] + RAVITOS_KM + [gpx.distance.max().round(2)]  


results,plan = generator(gpx,RAVITOS_KM,TEMPS_CIBLE_TOTAL_HEURES,MONTEE,DESCENTE,FATIGUE_MAX,INGREDIENTS,NOMBRE_FLASQUES,GLUCIDES_CIBLES_G_H,CAFEINE_MAX,SEED)  

resume_data = {  
    "Trail": [TRAIL],  
    "Distance": [results.km_fin.iloc[-1]],  
    "D+": [results["D+"].sum()],  
    "D-": [results["D-"].sum()],  
    "Durée (h)": [TEMPS_CIBLE_TOTAL_HEURES],  
    "Glucides (g/h)": [np.round(results.glucides_total.sum() / TEMPS_CIBLE_TOTAL_HEURES, 1)],  
    "Boisson (mL/h)": [int(results.flasques.sum() * 500 / TEMPS_CIBLE_TOTAL_HEURES)],  
    "Caféine (mg)": [int(plan.cafeine.sum())],  
}  
df_resume = pd.DataFrame(resume_data)  

total_ingredients = Counter()  
for d in results["ingrédients_solides"]:  
    total_ingredients.update(d)  
df_ingredients = pd.DataFrame(list(total_ingredients.items()),columns=["Ingrédient", "Quantité"])  
df_ingredients = pd.concat([df_ingredients.T,pd.Series(['boisson',results.flasques.sum()],index=['Ingrédient','Quantité'])],axis=1).T.reset_index(drop=True)  

plt.figure(figsize=(7,2))  
gpx.set_index('distance').altitude.plot()  
for rvt in RAVITOS_KM:  
    plt.axvline(x=rvt,color='red',alpha=.25)  
plt.grid()  
plt.title('Profil')  
plt.show()  

display(HTML(results[['km_debut', 'km_fin', 'D+', 'D-', 'durée', 'durée_cumulée', 'flasques', 'timing']].to_html(escape=False)))  
display(df_resume)  
display(df_ingredients)  

