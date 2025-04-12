import pandas as pd
from pandas.io.formats.style import Styler
import matplotlib.pyplot as plt
import streamlit as st
import cvxpy as cp
import numpy as np
import plotly.express as px
import pycountry
import plotly.graph_objects as go
import datetime
from math import pi
import io
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns

# Chemin du fichier Excel

file_path = "/Users/desma/OneDrive/Documents/Université/Université Paris Dauphine-PSL/M1/S2/Économie de l'Énergie et de l'Environnement/Data_Base.xlsx"

# Mise en cache des données pour éviter les rechargements inutiles
@st.cache_data
def charger_donnees():
    """Charge les DataFrames initiaux et les met en cache pour accélérer l'exécution Streamlit."""
    try:
        # Cours des indices principaux (S&P500, MSCI World, CAC40, STOXX600)
        dfIndices = pd.read_excel(file_path, sheet_name=1, index_col=0).loc["03/01/2000":"04/03/2025"]

        # Cours des composants des indices
        dfSP500 = pd.read_excel(file_path, sheet_name=2, index_col=0).loc["03/01/2000":"04/03/2025"]
        dfSTOXX600 = pd.read_excel(file_path, sheet_name=3, index_col=0).loc["03/01/2000":"04/03/2025"]
        dfCAC40 = pd.read_excel(file_path, sheet_name=4, index_col=0).loc["03/01/2000":"04/03/2025"]

        # Ratios financiers et capitalisation boursière
        dfSP500_ratios = pd.read_excel(file_path, sheet_name=5, index_col=0)
        dfSTOXX600_ratios = pd.read_excel(file_path, sheet_name=6, index_col=0)
        dfCAC40_ratios = pd.read_excel(file_path, sheet_name=7, index_col=0)

        # Classification sectorielle (BICS Levels)
        dfSP500_BICS = pd.read_excel(file_path, sheet_name=8, index_col=0)
        dfSTOXX600_BICS = pd.read_excel(file_path, sheet_name=9, index_col=0)
        dfCAC40_BICS = pd.read_excel(file_path, sheet_name=10, index_col=0)

        # Scores ESG (Environnement, Social, Gouvernance)
        dfSP500_ESG = pd.read_excel(file_path, sheet_name=11, index_col=0)
        dfSTOXX600_ESG = pd.read_excel(file_path, sheet_name=12, index_col=0)
        dfCAC40_ESG = pd.read_excel(file_path, sheet_name=13, index_col=0)

        # Cours des indices ESG (CAC40 ESG et STOXX600 ESG)
        dfIndices_ESG = pd.read_excel(file_path, sheet_name=14, index_col=0).loc["03/01/2000":"04/03/2025"]

        return {
            "dfIndices": dfIndices,
            "dfSP500": dfSP500,
            "dfSTOXX600": dfSTOXX600,
            "dfCAC40": dfCAC40,
            "dfSP500_ratios": dfSP500_ratios,
            "dfSTOXX600_ratios": dfSTOXX600_ratios,
            "dfCAC40_ratios": dfCAC40_ratios,
            "dfSP500_BICS": dfSP500_BICS,
            "dfSTOXX600_BICS": dfSTOXX600_BICS,
            "dfCAC40_BICS": dfCAC40_BICS,
            "dfSP500_ESG": dfSP500_ESG,
            "dfSTOXX600_ESG": dfSTOXX600_ESG,
            "dfCAC40_ESG": dfCAC40_ESG,
            "dfIndices_ESG": dfIndices_ESG
        }
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return None

# Charger les données une seule fois et les conserver en cache
data = charger_donnees()

# Vérifier que les données sont bien chargées
if data:
    dfIndices = data["dfIndices"]
    dfSP500 = data["dfSP500"]
    dfSTOXX600 = data["dfSTOXX600"]
    dfCAC40 = data["dfCAC40"]
    dfSP500_ratios = data["dfSP500_ratios"]
    dfSTOXX600_ratios = data["dfSTOXX600_ratios"]
    dfCAC40_ratios = data["dfCAC40_ratios"]
    dfSP500_BICS = data["dfSP500_BICS"]
    dfSTOXX600_BICS = data["dfSTOXX600_BICS"]
    dfCAC40_BICS = data["dfCAC40_BICS"]
    dfSP500_ESG = data["dfSP500_ESG"]
    dfSTOXX600_ESG = data["dfSTOXX600_ESG"]
    dfCAC40_ESG = data["dfCAC40_ESG"]
    dfIndices_ESG = data["dfIndices_ESG"]

# Fonction pour afficher les DataFrames dans la console (debugging)
def print_dataframes_with_names(**dfs):
    """Affichage des DataFrames pour vérification dans la console."""
    for name, df in dfs.items():
        print(f"\n###### {name} ######\n")
        print(df) 

# Affichage pour vérification (peut être retiré une fois validé)
if data:
    print_dataframes_with_names(
        dfIndices=dfIndices,
        dfSP500=dfSP500,
        dfSTOXX600=dfSTOXX600,
        dfCAC40=dfCAC40,
        dfSP500_ratios=dfSP500_ratios,
        dfSTOXX600_ratios=dfSTOXX600_ratios,
        dfCAC40_ratios=dfCAC40_ratios,
        dfSP500_BICS=dfSP500_BICS,
        dfSTOXX600_BICS=dfSTOXX600_BICS,
        dfCAC40_BICS=dfCAC40_BICS,
        dfSP500_ESG=dfSP500_ESG,
        dfSTOXX600_ESG=dfSTOXX600_ESG,
        dfCAC40_ESG=dfCAC40_ESG,
        dfIndices_ESG=dfIndices_ESG
    )

def calculer_pourcentage_manquantes(df, name):
    """
    Calcule la proportion de valeurs manquantes dans un DataFrame et l'affiche.
    
    Args:
        df (pd.DataFrame): DataFrame à analyser.
        name (str): Nom du DataFrame pour affichage.
        
    Returns:
        pd.Series: Séries triées avec le pourcentage de valeurs manquantes.
    """
    pourcentage_manquantes = df.isnull().sum().sort_values(ascending=False) / len(df) * 100
    print(f"\nProportion de valeurs absentes pour chaque membre de l'indice {name} :\n")
    print(pourcentage_manquantes)
    return pourcentage_manquantes

pourcentage_SP500 = calculer_pourcentage_manquantes(dfSP500, "S&P 500")
pourcentage_STOXX600 = calculer_pourcentage_manquantes(dfSTOXX600, "STOXX 600")
pourcentage_CAC40 = calculer_pourcentage_manquantes(dfCAC40, "CAC 40")


def tracer_distribution_na(pourcentage_manquantes, name):
    """
    Affiche un histogramme de la distribution des valeurs manquantes dans un DataFrame.
    
    Args:
        pourcentage_manquantes (pd.Series): Séries contenant les pourcentages de valeurs manquantes.
        name (str): Nom du DataFrame pour affichage.
    """
    plt.figure(figsize=(8, 5))
    pourcentage_manquantes.plot(kind='hist', bins=20, edgecolor='black')
    plt.xlabel("Proportion de valeurs manquantes (%)")
    plt.ylabel("Nombre de titres")
    plt.title(f"Distribution des valeurs manquantes - {name}")
    plt.grid()
    plt.show()

# Visualisation
tracer_distribution_na(pourcentage_SP500, "S&P 500")
tracer_distribution_na(pourcentage_STOXX600, "STOXX 600")
tracer_distribution_na(pourcentage_CAC40, "CAC 40")


def supprimer_colonnes_na(df, seuil=75):
    """
    Supprime les colonnes avec plus de `seuil`% de valeurs manquantes et retourne les titres supprimés.

    Args:
        df (pd.DataFrame): DataFrame à traiter.
        seuil (int): Seuil de pourcentage de valeurs NaN pour supprimer une colonne.

    Returns:
        tuple: (DataFrame nettoyé, set des colonnes supprimées)
    """
    # Liste des colonnes avant suppression
    titres_avant = set(df.columns)

    # Suppression des colonnes avec trop de valeurs NaN
    df = df.dropna(thresh=(1 - seuil / 100) * df.shape[0], axis=1)

    # Liste des colonnes après suppression
    titres_supprimes = titres_avant - set(df.columns)

    return df, titres_supprimes

dfSP500, titres_supprimes_SP500 = supprimer_colonnes_na(dfSP500, seuil=75)
dfSTOXX600, titres_supprimes_STOXX600 = supprimer_colonnes_na(dfSTOXX600, seuil=75)
dfCAC40, titres_supprimes_CAC40 = supprimer_colonnes_na(dfCAC40, seuil=75)

print("Titres supprimés du SP500 :", titres_supprimes_SP500)
print("Titres supprimés du STOXX600 :", titres_supprimes_STOXX600)
print("Titres supprimés du CAC40 :", titres_supprimes_CAC40)

pourcentage_SP500 = calculer_pourcentage_manquantes(dfSP500, "S&P 500")
pourcentage_STOXX600 = calculer_pourcentage_manquantes(dfSTOXX600, "STOXX 600")
pourcentage_CAC40 = calculer_pourcentage_manquantes(dfCAC40, "CAC 40")


def traitement_nan(df, name, type_data):
    """
    Remplissage des NaN avec forward fill (ffill) et suppression du warning.
    
    Args:
        df (pd.DataFrame): DataFrame à traiter.
        name (str): Nom du DataFrame pour affichage.
        type_data (str): Type de données (ex: "composants", "indices").
    
    Returns:
        pd.DataFrame: DataFrame mise à jour.
    """
    df = df.copy()  # Assure que df est une copie indépendante pour éviter SettingWithCopyWarning
    df.fillna(method="ffill", inplace=True)  # Remplit les NaN avec la dernière valeur connue
    print(f"Remplissage des NaN effectué pour {type_data} du {name}.")
    return df  

# Mise à jour des DataFrames des composants 
dfSP500 = traitement_nan(dfSP500, "S&P 500", "les composants")
dfSTOXX600 = traitement_nan(dfSTOXX600, "STOXX 600", "les composants")
dfCAC40 = traitement_nan(dfCAC40, "CAC 40", "les composants")

# Mise à jour des DataFrames des indices
dfIndices = traitement_nan(dfIndices, "S&P500, MSCI WOrld, STOXX600 et CAC40", "les indices")
dfIndices_ESG = traitement_nan(dfIndices_ESG, "CAC40 ESG et STOXX600 ESG", "les indices ESG :")


def supprimer_lignes(df, titres_a_supprimer, name):
    """
    Supprime les lignes des DataFrames qui correspondent aux titres supprimés des cours.

    Args:
        df (pd.DataFrame): DataFrame à nettoyer.
        titres_a_supprimer (set): Ensemble des index à supprimer.
        name (str): Nom du DataFrame pour affichage.

    Returns:
        pd.DataFrame: DataFrame filtré.
    """
    df_filtre = df.drop(index=titres_a_supprimer, errors="ignore")
    print(f"{name} mis à jour : {len(titres_a_supprimer)} lignes supprimées.")
    return df_filtre

dfSP500_ratios = supprimer_lignes(dfSP500_ratios, titres_supprimes_SP500, "SP500 Ratios")
dfSP500_BICS = supprimer_lignes(dfSP500_BICS, titres_supprimes_SP500, "SP500 BICS")
dfSP500_ESG = supprimer_lignes(dfSP500_ESG, titres_supprimes_SP500, "SP500 ESG")

dfSTOXX600_ratios = supprimer_lignes(dfSTOXX600_ratios, titres_supprimes_STOXX600, "STOXX600 Ratios")
dfSTOXX600_BICS = supprimer_lignes(dfSTOXX600_BICS, titres_supprimes_STOXX600, "STOXX600 BICS")
dfSTOXX600_ESG = supprimer_lignes(dfSTOXX600_ESG, titres_supprimes_STOXX600, "STOXX600 ESG")

dfCAC40_ratios = supprimer_lignes(dfCAC40_ratios, titres_supprimes_CAC40, "CAC40 Ratios")
dfCAC40_BICS = supprimer_lignes(dfCAC40_BICS, titres_supprimes_CAC40, "CAC40 BICS")
dfCAC40_ESG = supprimer_lignes(dfCAC40_ESG, titres_supprimes_CAC40, "CAC40 ESG")

print("\nVérification des dimensions des DataFrames après suppression des titres :")
print(f"SP500 : {dfSP500.shape} | SP500 Ratios : {dfSP500_ratios.shape} | SP500 BICS : {dfSP500_BICS.shape} | SP500 ESG : {dfSP500_ESG.shape}")
print(f"STOXX600 : {dfSTOXX600.shape} | STOXX600 Ratios : {dfSTOXX600_ratios.shape} | STOXX600 BICS : {dfSTOXX600_BICS.shape} | STOXX600 ESG : {dfSTOXX600_ESG.shape}")
print(f"CAC40 : {dfCAC40.shape} | CAC40 Ratios : {dfCAC40_ratios.shape} | CAC40 BICS : {dfCAC40_BICS.shape} | CAC40 ESG : {dfCAC40_ESG.shape}")


#Réunissons les données de nos 3 indices en des dataframes communs. Cela nous servira ultérieurement pour déterminer les fonds/portefeuilles de l'investisseur lorsqu'il sera indifférent au benchmark considéré.

# Titres des indices (utilisation correcte de .index)
titres_CAC40 = set(dfCAC40_ratios.index)  # Liste des titres du CAC40
titres_STOXX600 = set(dfSTOXX600_ratios.index)  # Liste des titres du STOXX600

# Identifions les titres communs
titres_communs = titres_CAC40.intersection(titres_STOXX600)

# Affichage des résultats
print(f"Titres du CAC40 également présents dans le STOXX600 : {sorted(titres_communs)}")
print(f"Nombre de titres en commun restant après filtrage détectés : {len(titres_communs)}")

# Création d'une Copie du dfCAC40 Sans les Titres Communs avec le stoxx600
dfCAC40_filtré = dfCAC40.drop(columns=titres_communs, errors="ignore")
dfCAC40_ratios_filtré = dfCAC40_ratios.drop(index=titres_communs, errors="ignore")
dfCAC40_BICS_filtré = dfCAC40_BICS.drop(index=titres_communs, errors="ignore")
dfCAC40_ESG_filtré = dfCAC40_ESG.drop(index=titres_communs, errors="ignore")

print(f"Titres supprimés du CAC40 appartenant déjà au Stoxx600 : {len(titres_communs)}")
print(f"Titres restants dans le CAC40 après filtrage et suppression des doublons : {dfCAC40_filtré.shape[1]}")

# Étape 3 : Fusion du SP500, du STOXX600 et du CAC40 Sans Doublons
dfComposants = pd.concat([dfSP500, dfSTOXX600, dfCAC40_filtré], axis=1).reindex(sorted(dfSP500.columns.union(dfSTOXX600.columns).union(dfCAC40_filtré.columns)), axis=1)
dfRatios = pd.concat([dfSP500_ratios, dfSTOXX600_ratios, dfCAC40_ratios_filtré]).sort_index()
dfBICS = pd.concat([dfSP500_BICS, dfSTOXX600_BICS, dfCAC40_BICS_filtré]).sort_index()
dfESG = pd.concat([dfSP500_ESG, dfSTOXX600_ESG, dfCAC40_ESG_filtré]).sort_index()

# Vérification après Fusion
print(f"DataFrame combiné des composants : {dfComposants.shape[1]} titres.")
print(f"DataFrame combiné des Ratios : {dfRatios.shape[0]} titres.")
print(f"DataFrame combiné des classifications BICS : {dfBICS.shape[0]} titres.")
print(f"DataFrame combiné des scores ESG : {dfESG.shape[0]} titres.")

# Affichage des DataFrames pour vérification
print_dataframes_with_names(
    Composants=dfComposants,
    Ratios=dfRatios,
    BICS=dfBICS,
    ESG=dfESG
)

# Mise en cache de tous les DataFrames finaux pour éviter le recalcul
@st.cache_data
def get_final_data():
    return {
        # Consolidation des indices
        "dfComposants": dfComposants,
        "dfRatios": dfRatios,
        "dfBICS": dfBICS,
        "dfESG": dfESG,

        # Données spécifiques au S&P 500
        "dfSP500": dfSP500,
        "dfSP500_ratios": dfSP500_ratios,
        "dfSP500_BICS": dfSP500_BICS,
        "dfSP500_ESG": dfSP500_ESG,

        # Données spécifiques au STOXX 600
        "dfSTOXX600": dfSTOXX600,
        "dfSTOXX600_ratios": dfSTOXX600_ratios,
        "dfSTOXX600_BICS": dfSTOXX600_BICS,
        "dfSTOXX600_ESG": dfSTOXX600_ESG,

        # Données spécifiques au CAC 40
        "dfCAC40": dfCAC40,
        "dfCAC40_ratios": dfCAC40_ratios,
        "dfCAC40_BICS": dfCAC40_BICS,
        "dfCAC40_ESG": dfCAC40_ESG
    }

# Chargement des données finales depuis le cache
final_data = get_final_data()

# Attribution des DataFrames aux variables pour une utilisation rapide
dfComposants = final_data["dfComposants"]
dfRatios = final_data["dfRatios"]
dfBICS = final_data["dfBICS"]
dfESG = final_data["dfESG"]

dfSP500 = final_data["dfSP500"]
dfSP500_ratios = final_data["dfSP500_ratios"]
dfSP500_BICS = final_data["dfSP500_BICS"]
dfSP500_ESG = final_data["dfSP500_ESG"]

dfSTOXX600 = final_data["dfSTOXX600"]
dfSTOXX600_ratios = final_data["dfSTOXX600_ratios"]
dfSTOXX600_BICS = final_data["dfSTOXX600_BICS"]
dfSTOXX600_ESG = final_data["dfSTOXX600_ESG"]

dfCAC40 = final_data["dfCAC40"]
dfCAC40_ratios = final_data["dfCAC40_ratios"]
dfCAC40_BICS = final_data["dfCAC40_BICS"]
dfCAC40_ESG = final_data["dfCAC40_ESG"]


# Préparation de streamlit

# Initialisation de session_state si nécessaire
if "user_choices" not in st.session_state:
    st.session_state["user_choices"] = {
        "benchmark": "Indifférent",
        "pays": [],
        "secteurs": [],
        "niveau_BICS": "BICS Niveau 1",
        "esg": {"source": "ESG par critères (Gouvernance, Social, Environnement)", 
                "gouvernance": 5, "social": 5, "environnement": 5},
        "objectif": None
    }

# Ajout des variables pour le suivi des options "Indifférent"
if "indiff_pays" not in st.session_state:
    st.session_state["indiff_pays"] = True
if "indiff_secteurs" not in st.session_state:
    st.session_state["indiff_secteurs"] = True

# Titre de l'application
st.title("🔍 Génération d'Indices Personnalisés avec Intégration ESG")

# Définition des options disponibles
liste_benchmarks = ["Indifférent", "S&P 500", "STOXX 600", "CAC 40"]
liste_niveaux_BICS = ["BICS Niveau 1", "BICS Niveau 2", "BICS Niveau 3", "BICS Niveau 4"]
liste_objectifs = [
    "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value.",
    "🛡️ Minimisation de la variance.",
    "🚀 Portefeuille efficient selon vos critères : Maximisation du Rendement selon votre niveau de risque.",
    "💎 Stratégie Value (P/E & P/B) : Maximisation de la valeur 1/PER + 1/Price to Book du portefeuille, selon votre niveau de risque."
]
liste_pays = sorted(set(dfSP500_ratios["COUNTRY"]).union(dfSTOXX600_ratios["COUNTRY"]).union(dfCAC40_ratios["COUNTRY"]))

# Définition des sources ESG
liste_sources_esg = [
    "ESG par critères (Gouvernance, Social, Environnement)",
    "ESG_SCORE (0-10 : Global)",
    "MSCI_ESG_RATING (AAA-CCC)",
    "SA_ESG_RISK_SCR (Négligeable à Très Élevé)"
]
# Interface avec des onglets incluant la Présentation
tab0, tab1, tab2, tab3, tab4 = st.tabs([
    "📖 Présentation du Projet",
    "🔍 Sélection des Critères",
    "🎯 Choix de l'Objectif",
    "📊 Résultats de l'Optimisation",
    "🧾 Bilan par Action"
])

# Initialisation de l'onglet actif s'il n'existe pas encore
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "📖 Présentation du Projet"

# Fonction pour mettre à jour l'onglet actif lorsqu'on change
def set_active_tab(tab_name):
    st.session_state["active_tab"] = tab_name

# Gestion de l'affichage selon l'onglet actif
if st.session_state["active_tab"] == "📖 Présentation du Projet":
    with tab0:
        set_active_tab("📖 Présentation du Projet")
        st.title("🌱 Bienvenue dans notre Application d'Optimisation Financière et ESG")

        st.markdown("""
        ## 💡 Pourquoi se contenter de construire un seul portefeuille ESG... quand vous pouvez explorer **des milliers de combinaisons possibles**, entièrement **personnalisées** ?

        Dans un monde en constante évolution, la **finance durable** n'est plus une option mais une **nécessité**. Notre application va plus loin : elle vous permet **non seulement** de créer un portefeuille respectueux des critères ESG, **mais aussi** d'explorer plusieurs **stratégies d'optimisation** selon vos préférences et votre tolérance au risque.

        🔎 **Analysez, optimisez, investissez...** en toute conscience.

        La base de données utilisée par l'algorithme est vaste et regroupe notamment les **cotations journalières des actifs des indices S&P 500, STOXX 600 et CAC 40**, du **3 janvier 2000 au 4 mars 2025**. La devise utilisée est le **dollar américain**. Les calendriers de cotation sont **harmonisés**.

        Nous pourrons également faire intervenir les **indices ESG** suivants lors de la comparaison de votre portefeuille à ces derniers : **CAC 40 ESG Index** et **SXXP ESG X Index**. Les données historiques sont disponibles à partir du 1er janvier 2010 pour le CAC 40 ESG et du 19 mars 2012 pour le SXXP ESG X.

        **Naviguez entre les onglets en cliquant sur leur nom !**            
                    
        ---

        ### 🔍 **Onglet 1 - Sélection des Critères**
        Ici, vous pouvez retrouver les **critères de départ** que vous avez sélectionnés dans **la barre latérale à gauche** de votre écran :
        
        - Votre **benchmark** (par exemple : S&P 500, STOXX 600, CAC 40, ou bien les trois simultanément en sélectionnant "Indifférent" — les titres du CAC 40 appartenant également au STOXX 600).
        - Les **zones géographiques** qui vous intéressent : décochez la case *"Indifférent aux pays"* pour pouvoir sélectionner des pays en particulier.
        - Le **niveau BICS** à travers lequel votre portefeuille sera analysé, puis les **secteurs** que vous désirez : décochez la case *"Indifférent aux secteurs"* pour pouvoir en choisir en particulier.
        - La **source et le niveau d'exigence ESG** que nous avons en notre possession pour chaque titre :
            - Analyse par **critères Environnementaux, Sociaux et de Gouvernance** : 3 jauges indépendantes à renseigner.
            - **Score Global ESG** : une seule jauge englobant les 3 dimensions, provenant d'une source différente.
            - **Notation MSCI** (AAA à CCC) : l'algorithme attribuera une notation selon l'échelle utilisée pour chaque titre par MSCI.
            - **Score de Risque ESG** (Négligeable à Très Élevé) : plus la notation tire vers la catégorie *"Négligeable"*, et moins votre portefeuille est exposé au risque ESG.

        **Il était crucial pour nous de vous proposer plusieurs sources et dimensions ESG différentes pour chaque titre, afin d'obtenir une évaluation la plus juste possible.**

        **Astuce :** Si vous ne touchez à rien, des valeurs par **défaut intelligentes** s'appliquent.

        ---

        ### 🎯 **Onglet 2 - Choix de l'Objectif**
        Ici, vous entrez dans le **cœur stratégique** du projet. **Quatre approches d’optimisation** s’offrent à vous :

        #### 🧹 **1. Filtrage Simple**
        👉 Construisez un **portefeuille sur mesure** en ne retenant **que les titres** respectant **individuellement** vos **critères, y compris ESG**. Il s'agit ici de l'offre la plus simple possible pour vous... et ainsi **la moins optimale**. \n
        
        Vous avez **deux options** de pondération possibles :
        - **💰 Pondération par Capitalisation Boursière** *(par défaut)* : la part de chaque titre est calculée comme le **ratio** entre sa **capitalisation boursière** et la **somme des capitalisations** de tous les titres sélectionnés.
        - **⚖️ Équipondération** *(option activable)* : chaque titre sélectionné **reçoit le même poids** dans le portefeuille, **quelle que soit sa taille ou sa capitalisation**.\n
       
        **Bonus** : Vous pouvez activer l'option **"Filtrer uniquement les titres Value"** — seuls les titres considérés comme sous-évalués (faible P/E et P/B) seront sélectionnés.

        ---

        #### 🛡️ **2. Minimisation de la Variance**
        👉 Laissez l'algorithme construire **le portefeuille le moins risqué possible selon vos critères** : idéal pour les investisseurs prudents souhaitant **maximiser la sécurité** tout en respectant les critères ESG.

        ---

        #### 🚀 **3. Portefeuille Efficient : Maximisation du Rendement selon votre niveau de risque**
        👉 Ici, **le rendement est roi**... mais **votre tolérance au risque est respectée** grâce au paramétrage du **niveau maximal de volatilité**.

        Vous construisez un portefeuille **efficient** et **personnalisé** qui maximise la performance **en fonction de vos contraintes ESG et de votre aversion au risque**.

        ⚠️ **Décochez la case "Indifférence au risque" pour pouvoir entrer la volatilité maximale que vous pourrez supporter !**

        ---

        #### 💎 **4. Stratégie Value (P/E & P/B)**
        👉 Une approche **financière historique** et toujours d'actualité : nous sélectionnons les titres **sous-évalués** selon leurs ratios **Price-to-Earnings (P/E)** et **Price-to-Book (P/B)**.

        L’algorithme maximise la valeur **1/PER + 1/Price to Book**, tout en intégrant vos critères et exigences ESG, ainsi que **votre seuil de volatilité maximale**.
        
        ⚠️ **Décochez la case "Indifférence au risque" pour pouvoir entrer la volatilité maximale que vous pourrez supporter !**

        ---

        ### 📊 **Onglet 3 - Résultats de l'Optimisation**
        ✅ Une fois vos critères et votre stratégie définis, **lancez l’algorithme** en appuyant sur le bouton dans **l'onglet 2**, et découvrez le **dashboard construit** :
        - La composition **optimale** de votre portefeuille
        - Les **performances financières attendues**
        - Le **respect des critères ESG**
        - La **répartition géographique et sectorielle** des titres composant votre portefeuille.
        - ... et toute une gamme complète de **données et comparatifs** !

        Chaque **décision d’investissement est visible et traçable**.

        ---

        ### 📈 **Onglet 4 - 🧾 Bilan par Action**
        🧐 Il s'agit ici d'un **onglet indépendant**. Explorez chaque **actif individuellement** :
        - Visualisez ses caractéristiques, sa **performance financière**
        - Analysez sa **notation ESG**
        - Parcourez l’intégralité des données disponibles pour chaque titre pour mieux comprendre **chaque opportunité d’investissement**.

        ---

        ## 🚀 Alors, prêt à construire **le portefeuille qui vous ressemble** ?
        👉 **Commencez par l'onglet "🔍 Sélection des Critères**", choisissez votre stratégie dans "**🎯 Choix de l'Objectif**", lancez l’optimisation et **analysez vos résultats** dans "**📊 Résultats de l'Optimisation**".

        ### 🎯 **Optimisez vos investissements sans jamais trahir vos convictions ESG.**
        """)

elif st.session_state["active_tab"] == "🔍 Sélection des Critères":
    with tab1:
        set_active_tab("🔍 Sélection des Critères")

elif st.session_state["active_tab"] == "🎯 Choix de l'Objectif":
    with tab2:
        set_active_tab("🎯 Choix de l'Objectif")

elif st.session_state["active_tab"] == "📊 Résultats de l'Optimisation":
    with tab3:
        set_active_tab("📊 Résultats de l'Optimisation")

elif st.session_state["active_tab"] == "🧾 Bilan par Action":
    with tab4:
        set_active_tab("🧾 Bilan par Action")


# Onglet 1 : Sélection des Critères
with tab1:

    # Sélection du Benchmark (remis à sa place initiale)
    st.sidebar.subheader("📈 Choix du Benchmark")
    benchmark = st.sidebar.selectbox(
        "📈 Sélectionner un ou plusieurs indices :",
        liste_benchmarks,
        index=liste_benchmarks.index(st.session_state["user_choices"]["benchmark"])
    )
    st.session_state["user_choices"]["benchmark"] = benchmark

    # Ajout des cases Indifférence
    st.sidebar.subheader("🌍 Sélection Géographique")
    indiff_pays = st.sidebar.checkbox("🌍 Indifférent aux pays (Tous inclus)", value=st.session_state["indiff_pays"])
    st.session_state["indiff_pays"] = indiff_pays

    pays_selectionnes = st.sidebar.multiselect(
        "🌍 Sélectionner un ou plusieurs pays :",
        liste_pays if not indiff_pays else ["Tous sélectionnés"],
        default=["Tous sélectionnés"] if indiff_pays else st.session_state["user_choices"]["pays"],
        disabled=indiff_pays
    )
    st.session_state["user_choices"]["pays"] = [] if indiff_pays else pays_selectionnes

    st.sidebar.subheader("🏢 Sélection des Secteurs via BICS")

    # Sélection du niveau BICS
    niveau_BICS = st.sidebar.selectbox("🔍 Sélectionner le niveau BICS d'analyse :", liste_niveaux_BICS,
                                       index=liste_niveaux_BICS.index(st.session_state["user_choices"]["niveau_BICS"]))
    st.session_state["user_choices"]["niveau_BICS"] = niveau_BICS

    # Case à cocher pour l'indifférence sectorielle
    indiff_secteurs = st.sidebar.checkbox("🏢 Indifférent aux secteurs (Tous inclus)", value=st.session_state["indiff_secteurs"])
    st.session_state["indiff_secteurs"] = indiff_secteurs

    # Mapping des niveaux BICS
    bics_colonne_map = {
        "BICS Niveau 1": "bics_level_1_sector_name",
        "BICS Niveau 2": "bics_level_2_industry_group_name",
        "BICS Niveau 3": "bics_level_3_industry_name",
        "BICS Niveau 4": "bics_level_4_sub_industry_name"
    }

    # Liste des secteurs disponibles
    liste_secteurs = sorted(set(dfSP500_BICS.get(bics_colonne_map[niveau_BICS], []))
                            .union(dfSTOXX600_BICS.get(bics_colonne_map[niveau_BICS], []))
                            .union(dfCAC40_BICS.get(bics_colonne_map[niveau_BICS], [])))

    secteurs_selectionnes = st.sidebar.multiselect(
        f"📌 Sélectionner un ou plusieurs secteurs ({niveau_BICS}) :",
        liste_secteurs if not indiff_secteurs else ["Tous sélectionnés"],
        default=["Tous sélectionnés"] if indiff_secteurs else st.session_state["user_choices"]["secteurs"],
        disabled=indiff_secteurs
    )
    st.session_state["user_choices"]["secteurs"] = [] if indiff_secteurs else secteurs_selectionnes

    # Sélection de la source des critères ESG
    st.sidebar.subheader("♻️ Source des critères ESG")
    source_esg = st.sidebar.selectbox("📊 Sélectionner la source des critères ESG :", liste_sources_esg,
                                      index=liste_sources_esg.index(st.session_state["user_choices"]["esg"]["source"]))
    st.session_state["user_choices"]["esg"]["source"] = source_esg

    # Gestion des valeurs ESG stockées dans `session_state`
    if "esg_values" not in st.session_state:
        st.session_state["esg_values"] = {
            "gouvernance": 5,
            "social": 5,
            "environnement": 5,
            "esg_score": 5,
            "msci_rating": "BBB",
            "sa_esg_risk": "Moyen (20-30)"
        }

    # Affichage dynamique des critères ESG
    st.sidebar.subheader("♻️ Critères ESG")

    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        gouvernance = st.sidebar.slider("🏛 Importance de la Gouvernance :", 0, 10, st.session_state["esg_values"]["gouvernance"])
        social = st.sidebar.slider("🤝 Importance du Social :", 0, 10, st.session_state["esg_values"]["social"])
        environnement = st.sidebar.slider("🌿 Importance de l'Environnement :", 0, 10, st.session_state["esg_values"]["environnement"])

        st.session_state["esg_values"]["gouvernance"] = gouvernance
        st.session_state["esg_values"]["social"] = social
        st.session_state["esg_values"]["environnement"] = environnement

        st.session_state["user_choices"]["esg"]["gouvernance"] = gouvernance
        st.session_state["user_choices"]["esg"]["social"] = social
        st.session_state["user_choices"]["esg"]["environnement"] = environnement

    elif source_esg == "ESG_SCORE (0-10 : Global)":
        esg_score = st.sidebar.slider("📊 ESG Score Global (0-10) :", 0, 10, st.session_state["esg_values"]["esg_score"])

        st.session_state["esg_values"]["esg_score"] = esg_score
        st.session_state["user_choices"]["esg"]["esg_score"] = esg_score

    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        liste_ratings = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC"]
        msci_rating = st.sidebar.selectbox("📊 Sélectionner la note MSCI ESG :", liste_ratings,
                                           index=liste_ratings.index(st.session_state["esg_values"]["msci_rating"]))

        st.session_state["esg_values"]["msci_rating"] = msci_rating
        st.session_state["user_choices"]["esg"]["msci_rating"] = msci_rating

    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        liste_risques = ["Négligeable (0-10)", "Faible (10-20)", "Moyen (20-30)", "Élevé (30-40)", "Très Élevé (40+)"]
        sa_esg_risk = st.sidebar.selectbox("⚠️ Score de risque ESG :", liste_risques,
                                           index=liste_risques.index(st.session_state["esg_values"]["sa_esg_risk"]))

        st.session_state["esg_values"]["sa_esg_risk"] = sa_esg_risk
        st.session_state["user_choices"]["esg"]["sa_esg_risk"] = sa_esg_risk

    # Ajout des noms en bas de la sidebar
    st.sidebar.markdown("""
    ---
    👥 **Réalisé par** :  
    - DESMAREST Vincent  
    - FIGUEIREDO Laeticia  
    - VAZ Alexia  
    - RAKOTOARISOA Lana-Rose
    """)

    # Affichage des choix sélectionnés
    st.write("### ✅ Vos Critères Sélectionnés")
    st.write(f"📈 **Benchmark sélectionné :** {st.session_state['user_choices']['benchmark']}")
    st.write(f"🌍 **Pays sélectionnés :** {'Tous' if indiff_pays else ', '.join(st.session_state['user_choices']['pays'])}")
    st.write(f"🏢 **Secteurs sélectionnés ({niveau_BICS}) :** {'Tous' if indiff_secteurs else ', '.join(st.session_state['user_choices']['secteurs'])}")
    st.write(f"♻️ **Source des critères ESG :** {st.session_state['user_choices']['esg']['source']}")

    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        st.markdown(f"📊 **Notations minimales :**<br>"
                    f"🏛 Gouvernance : {st.session_state['user_choices']['esg']['gouvernance']}/10<br>"
                    f"🤝 Social : {st.session_state['user_choices']['esg']['social']}/10<br>"
                    f"🌿 Environnement : {st.session_state['user_choices']['esg']['environnement']}/10",
                    unsafe_allow_html=True)
    elif source_esg == "ESG_SCORE (0-10 : Global)":
        st.write(f"📊 **Notation minimale ESG Score :** {st.session_state['user_choices']['esg']['esg_score']}/10")
    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        st.write(f"📊 **Notation minimale MSCI ESG :** {st.session_state['user_choices']['esg']['msci_rating']}")
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        st.write(f"⚠️ **Risque ESG maximal :** {st.session_state['user_choices']['esg']['sa_esg_risk']}")

with tab2:
    # Sélection de l'objectif d'optimisation
    st.subheader("🎯 Choix de l'Objectif d'Optimisation")
    objectif_optimisation = st.radio("Sélectionner un objectif :", liste_objectifs)

    # Explication dynamique de l'objectif choisi
    if objectif_optimisation == "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value.":
        st.write("🔹 **Vous avez choisi une approche Filtrage Simple : Les actions seront sélectionnées en fonction des critères ESG imposés.**")
    elif objectif_optimisation == "🛡️ Minimisation de la variance.":
        st.write("🔹 **Vous avez choisi de minimiser la variance : Le portefeuille avec le risque le plus faible sera sélectionné automatiquement.**")
    elif objectif_optimisation == "🚀 Portefeuille efficient selon vos critères : Maximisation du Rendement selon votre niveau de risque.":
        st.write("🔹 **Vous avez choisi un portefeuille efficient : Le rendement sera maximisé selon votre niveau de risque toléré.**")
    elif objectif_optimisation == "💎 Stratégie Value (P/E & P/B) : Maximisation de la valeur 1/PER + 1/Price to Book du portefeuille, selon votre niveau de risque.":
        st.write("🔹 **Vous avez choisi une approche Value : Les actions sous-évaluées seront sélectionnées en fonction de votre tolérance au risque.**")

    # Griser l'aversion au risque pour MinVariance et Filtrage Simple
    indiff_aversion_risque = st.checkbox(
        "🔄 Indifférent au degré d'aversion au risque",
        value=True,
        disabled=(objectif_optimisation == "🛡️ Minimisation de la variance." or objectif_optimisation == "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value.")
    )

    # Slider de volatilité (désactivé si indifférent ou non applicable)
    volatilite_max = st.slider(
        "📊 Définir la volatilité maximale autorisée (%) :",
        5, 50, step=1,
        disabled=(objectif_optimisation == "🛡️ Minimisation de la variance." or 
                  indiff_aversion_risque or 
                  objectif_optimisation == "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value.")
    )

    # Affichage dynamique de la volatilité uniquement si une valeur est sélectionnée
    if not indiff_aversion_risque and objectif_optimisation not in ["🛡️ Minimisation de la variance.", "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value."]:
        st.write(f"📊 **Volatilité maximale autorisée** : {volatilite_max}%")

    # Stocker les choix utilisateur
    st.session_state["user_choices"]["objectif"] = objectif_optimisation
    st.session_state["user_choices"]["volatilite_max"] = volatilite_max if not indiff_aversion_risque else None
    st.session_state["user_choices"]["indiff_aversion_risque"] = indiff_aversion_risque

# Mise en cache des statistiques financières
@st.cache_data
def pretraiter_donnees_financieres(dfIndices, dfIndices_ESG, dfSP500, dfSTOXX600, dfCAC40, dfComposants):
    """
    Prétraitement des données financières pour optimiser les portefeuilles.
    Correction des tailles incohérentes en réalignant les indices temporels.
    """

    # Nombre de jours de trading par an
    jours_trading = 252

    # Alignement des indices de temps pour uniformiser les tailles
    common_index = dfIndices.index.intersection(dfSP500.index).intersection(dfSTOXX600.index).intersection(dfCAC40.index)

    dfIndices = dfIndices.loc[common_index]
    dfIndices_ESG = dfIndices_ESG.loc[dfIndices_ESG.index.intersection(common_index)]
    dfSP500 = dfSP500.loc[common_index]
    dfSTOXX600 = dfSTOXX600.loc[common_index]
    dfCAC40 = dfCAC40.loc[common_index]
    dfComposants = dfComposants.loc[common_index]

    # Fonctions pour le calcul des statistiques
    def calculer_rendements(df):
        """Renvoie les rendements journaliers (logarithmiques possibles en option)."""
        return df.pct_change().dropna()

    def calculer_volatilite(df_rendements):
        """
        Calcule la volatilité annualisée à partir des rendements journaliers.
        """
        return df_rendements.std(ddof=1) * np.sqrt(jours_trading)

    def calculer_covariances(df_rendements):
        """
        Calcule la matrice de covariance annualisée à partir des rendements journaliers.
        """
        return df_rendements.cov() * jours_trading

    def calculer_max_drawdown(df_rendements):
        """
        Calcule le maximum drawdown d’un actif ou d’un DataFrame de rendements.
        Renvoie le drawdown maximum par colonne.
        """
        cumul_rendements = (1 + df_rendements).cumprod()
        pic = cumul_rendements.cummax()
        drawdown = (pic - cumul_rendements) / pic
        return drawdown.max()

    def calculer_ratio_sharpe(df_rendements, taux_sans_risque=0.02):
        """
        Calcule le ratio de Sharpe annualisé pour chaque actif.
        """
        rendement_annuel = (1 + df_rendements).prod() ** (jours_trading / len(df_rendements)) - 1
        volatilite_annuelle = df_rendements.std(ddof=1) * np.sqrt(jours_trading)
        return (rendement_annuel - taux_sans_risque) / volatilite_annuelle


    # Calcul des rendements journaliers
    dfRendementsIndices = calculer_rendements(dfIndices)
    dfRendementsIndicesESG = calculer_rendements(dfIndices_ESG)
    dfRendementsSP500 = calculer_rendements(dfSP500)
    dfRendementsSTOXX600 = calculer_rendements(dfSTOXX600)
    dfRendementsCAC40 = calculer_rendements(dfCAC40)
    dfRendementsConsolidés = calculer_rendements(dfComposants)

    # Décomposition des volatilités annualisées
    dfVolatiliteSP500 = calculer_volatilite(dfRendementsSP500).to_frame(name="Volatilité SP500")
    dfVolatiliteSTOXX600 = calculer_volatilite(dfRendementsSTOXX600).to_frame(name="Volatilité STOXX600")
    dfVolatiliteCAC40 = calculer_volatilite(dfRendementsCAC40).to_frame(name="Volatilité CAC40")
    dfVolatiliteConsolide = calculer_volatilite(dfRendementsConsolidés).to_frame(name="Volatilité Consolidé")

    # Matrices de covariance
    dfCovariancesSP500 = calculer_covariances(dfRendementsSP500)
    dfCovariancesSTOXX600 = calculer_covariances(dfRendementsSTOXX600)
    dfCovariancesCAC40 = calculer_covariances(dfRendementsCAC40)
    dfCovariancesConsolidées = calculer_covariances(dfRendementsConsolidés)

    # Décomposition des Drawdowns maximaux
    dfMaxDrawdownsSP500 = calculer_max_drawdown(dfRendementsSP500).to_frame(name="Max Drawdown SP500")
    dfMaxDrawdownsSTOXX600 = calculer_max_drawdown(dfRendementsSTOXX600).to_frame(name="Max Drawdown STOXX600")
    dfMaxDrawdownsCAC40 = calculer_max_drawdown(dfRendementsCAC40).to_frame(name="Max Drawdown CAC40")
    dfMaxDrawdownsConsolide = calculer_max_drawdown(dfRendementsConsolidés).to_frame(name="Max Drawdown Consolidé")

    # Décomposition des Ratios de Sharpe
    dfRatiosSharpeSP500 = calculer_ratio_sharpe(dfRendementsSP500).to_frame(name="Sharpe Ratio SP500")
    dfRatiosSharpeSTOXX600 = calculer_ratio_sharpe(dfRendementsSTOXX600).to_frame(name="Sharpe Ratio STOXX600")
    dfRatiosSharpeCAC40 = calculer_ratio_sharpe(dfRendementsCAC40).to_frame(name="Sharpe Ratio CAC40")
    dfRatiosSharpeConsolide = calculer_ratio_sharpe(dfRendementsConsolidés).to_frame(name="Sharpe Ratio Consolidé")

    return {
        "dfRendementsIndices": dfRendementsIndices,
        "dfRendementsIndicesESG": dfRendementsIndicesESG,
        "dfRendementsSP500": dfRendementsSP500,
        "dfRendementsSTOXX600": dfRendementsSTOXX600,
        "dfRendementsCAC40": dfRendementsCAC40,
        "dfRendementsConsolidés": dfRendementsConsolidés,
        "dfVolatiliteSP500": dfVolatiliteSP500,
        "dfVolatiliteSTOXX600": dfVolatiliteSTOXX600,
        "dfVolatiliteCAC40": dfVolatiliteCAC40,
        "dfVolatiliteConsolide": dfVolatiliteConsolide,
        "dfCovariancesSP500": dfCovariancesSP500,
        "dfCovariancesSTOXX600": dfCovariancesSTOXX600,
        "dfCovariancesCAC40": dfCovariancesCAC40,
        "dfCovariancesConsolidées": dfCovariancesConsolidées,
        "dfMaxDrawdownsSP500": dfMaxDrawdownsSP500,
        "dfMaxDrawdownsSTOXX600": dfMaxDrawdownsSTOXX600,
        "dfMaxDrawdownsCAC40": dfMaxDrawdownsCAC40,
        "dfMaxDrawdownsConsolide": dfMaxDrawdownsConsolide,
        "dfRatiosSharpeSP500": dfRatiosSharpeSP500,
        "dfRatiosSharpeSTOXX600": dfRatiosSharpeSTOXX600,
        "dfRatiosSharpeCAC40": dfRatiosSharpeCAC40,
        "dfRatiosSharpeConsolide": dfRatiosSharpeConsolide
    }


# Exécution du prétraitement et mise en cache
donnees_financieres = pretraiter_donnees_financieres(dfIndices, dfIndices_ESG, dfSP500, dfSTOXX600, dfCAC40, dfComposants)
# Mettre à jour final_data avec les données financières prétraitées
final_data.update(donnees_financieres)

# Vérification et affichage des tailles
print("✅ **Vérification des dimensions des DataFrames après prétraitement** ✅\n")

# Vérification des rendements journaliers
for df_name in ["dfRendementsIndices", "dfRendementsIndicesESG", "dfRendementsSP500", "dfRendementsSTOXX600", "dfRendementsCAC40", "dfRendementsConsolidés"]:
    print(f"📊 Rendements journaliers {df_name} : {donnees_financieres[df_name].shape} (lignes x colonnes)")

# Vérification des volatilités
for df_name in ["dfVolatiliteSP500", "dfVolatiliteSTOXX600", "dfVolatiliteCAC40", "dfVolatiliteConsolide"]:
    print(f"📊 Volatilité Annualisée {df_name} : {donnees_financieres[df_name].shape} (lignes x colonnes)")

# Vérification des matrices de covariance
for df_name in ["dfCovariancesSP500", "dfCovariancesSTOXX600", "dfCovariancesCAC40", "dfCovariancesConsolidées"]:
    print(f"📊 Matrice de covariance {df_name} : {donnees_financieres[df_name].shape} (lignes x colonnes)")

# Vérification des Drawdowns
for df_name in ["dfMaxDrawdownsSP500", "dfMaxDrawdownsSTOXX600", "dfMaxDrawdownsCAC40", "dfMaxDrawdownsConsolide"]:
    print(f"📉 Maximum Drawdowns {df_name} : {donnees_financieres[df_name].shape} (lignes x colonnes)")

# Vérification des Ratios de Sharpe
for df_name in ["dfRatiosSharpeSP500", "dfRatiosSharpeSTOXX600", "dfRatiosSharpeCAC40", "dfRatiosSharpeConsolide"]:
    print(f"📈 Ratios de Sharpe {df_name} : {donnees_financieres[df_name].shape} (lignes x colonnes)")


# Programmes

# Fonction de conversion des notations MSCI ESG en valeurs numériques
def convertir_notation_msci_en_valeur(msci_rating):
    mapping = {"AAA": 7, "AA": 6, "A": 5, "BBB": 4, "BB": 3, "B": 2, "CCC": 1}
    return mapping.get(msci_rating, np.nan)

# Fonction de conversion finale du score numérique du portefeuille en notation MSCI par intervalles
def classer_portefeuille_msciesg(score):
    if pd.isna(score):
        return "N.S."  # Non significatif
    elif score >= 6.5:
        return "AAA"
    elif 5.5 <= score < 6.5:
        return "AA"
    elif 4.5 <= score < 5.5:
        return "A"
    elif 3.5 <= score < 4.5:
        return "BBB"
    elif 2.5 <= score < 3.5:
        return "BB"
    elif 1.5 <= score < 2.5:
        return "B"
    else:
        return "CCC"
    
# Fonction de classement du risque ESG selon son score
def classer_risque(score):
    if score < 10: 
        return "Négligeable (0-10)"
    elif 10 <= score < 20: 
        return "Faible (10-20)"
    elif 20 <= score < 30: 
        return "Moyen (20-30)"
    elif 30 <= score < 40: 
        return "Élevé (30-40)"
    else: 
        return "Très Élevé (40+)"
    
# Convertir ISO-2 ➔ ISO-3 pour la carte interactive
def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2).alpha_3
    except:
        return None

benchmark_map = {
    "S&P 500": "SP500",
    "STOXX 600": "STOXX600",
    "CAC 40": "CAC40",
    "Indifférent": "Indifférent"
}

def afficher_comparaison_indices(df_rendements, rendement_pf, vol_pf, titre, gras=True):

    # Couleurs
    color_vert = "#d0f0c0"
    color_rouge = "#ffcccc"
    color_ref = "#e0f0ff"
    color_gris = "#f0f0f0"

    # Calcul des stats des indices
    stats = pd.DataFrame({
        "Rendement Annualisé (%)": df_rendements.mean() * 252 * 100,
        "Volatilité Annualisée (%)": df_rendements.std() * np.sqrt(252) * 100
    }).round(2)

    # Ajout du portefeuille en tête
    stats.loc["Portefeuille Optimisé"] = {
        "Rendement Annualisé (%)": rendement_pf * 100,
        "Volatilité Annualisée (%)": vol_pf * 100
    }
    stats = stats.loc[["Portefeuille Optimisé"] + [i for i in stats.index if i != "Portefeuille Optimisé"]]

    # Titre affiché
    titre_affiche = f"**{titre}**" if gras else titre
    st.markdown(f"📊 {titre_affiche}")

    # Légende couleurs
    st.markdown(f"""
    <div style='padding: 8px 16px; border-left: 5px solid #4CAF50; background-color: #f9f9f9; margin-bottom: 12px;'>
        <b>🎨 Légende des couleurs :</b><br>
        <span style="background-color:{color_ref}; padding:2px 6px; border-radius:3px;">📌 Bleu pâle</span> : Votre portefeuille optimisé (référence)<br>
        <span style="background-color:{color_vert}; padding:2px 6px; border-radius:3px;">🟩 Vert pâle</span> : Mieux que votre portefeuille (meilleur rendement ou moindre volatilité)<br>
        <span style="background-color:{color_gris}; padding:2px 6px; border-radius:3px;">⬜ Gris clair</span> : Égalité avec votre portefeuille<br>
        <span style="background-color:{color_rouge}; padding:2px 6px; border-radius:3px;">🟥 Rouge pâle</span> : Moins bien que votre portefeuille (moindre rendement ou plus grande volatilité)<br>
    </div>
    """, unsafe_allow_html=True)

    # Application des styles conditionnels
    styles = pd.DataFrame("", index=stats.index, columns=stats.columns)

    for idx in stats.index:
        if idx == "Portefeuille Optimisé":
            styles.loc[idx, :] = f"background-color: {color_ref};"
        else:
            r_i = stats.loc[idx, "Rendement Annualisé (%)"]
            v_i = stats.loc[idx, "Volatilité Annualisée (%)"]
            r_p = stats.loc["Portefeuille Optimisé", "Rendement Annualisé (%)"]
            v_p = stats.loc["Portefeuille Optimisé", "Volatilité Annualisée (%)"]

            styles.loc[idx, "Rendement Annualisé (%)"] = (
                f"background-color: {color_vert};" if r_i > r_p else
                f"background-color: {color_rouge};" if r_i < r_p else
                f"background-color: {color_gris};"
            )
            styles.loc[idx, "Volatilité Annualisée (%)"] = (
                f"background-color: {color_vert};" if v_i < v_p else
                f"background-color: {color_rouge};" if v_i > v_p else
                f"background-color: {color_gris};"
            )

    st.dataframe(
        stats.style.set_properties(**{"text-align": "center"}).apply(lambda _: styles, axis=None).format("{:.2f}"),
        height=300
    )

    # Interprétation des comparaisons
    interpretations = []
    for idx in stats.index:
        if idx == "Portefeuille Optimisé":
            continue

        r_i = stats.loc[idx, "Rendement Annualisé (%)"]
        v_i = stats.loc[idx, "Volatilité Annualisée (%)"]
        r_p = stats.loc["Portefeuille Optimisé", "Rendement Annualisé (%)"]
        v_p = stats.loc["Portefeuille Optimisé", "Volatilité Annualisée (%)"]

        if r_i > r_p and v_i < v_p:
            texte = f"🔰 <b>{idx}</b> surpasse <b>votre portefeuille</b> avec un <b>rendement plus élevé</b> et une <b>volatilité plus faible</b> : un profil <b>idéal</b>, à la fois <i>offensif</i> et <i>défensif</i>."
        elif r_i > r_p and v_i > v_p:
            texte = f"📈 <b>{idx}</b> propose une approche plus <i>offensive</i> que <b>votre portefeuille</b>, avec un <b>meilleur rendement</b> mais aussi une <b>volatilité plus élevée</b>."
        elif r_i < r_p and v_i < v_p:
            texte = f"🛡️ <b>{idx}</b> est plus <i>défensif</i> que <b>votre portefeuille</b> : <b>moins risqué</b>, mais avec un <b>rendement inférieur</b>."
        elif r_i < r_p and v_i > v_p:
            texte = f"❌ <b>{idx}</b> est inférieur à <b>votre portefeuille</b> sur les deux plans : <b>moins performant</b> et <b>plus risqué</b>."
        elif r_i == r_p and v_i == v_p:
            texte = f"🔄 <b>{idx}</b> présente un profil identique à <b>votre portefeuille</b>, tant en <b>rendement</b> qu'en <b>volatilité</b>."
        else:
            texte = f"📌 <b>{idx}</b> montre un profil <b>mixte</b> par rapport à <b>votre portefeuille</b>, sans positionnement clair comme <i>offensif</i> ou <i>défensif</i>."

        interpretations.append(f"<li>{texte}</li>")

    st.markdown("<ul>" + "\n".join(interpretations) + "</ul>", unsafe_allow_html=True)

# Ajout d'un bouton pour lancer l'optimisation ou le filtrage
with tab2:
    lancer_optimisation_minvar = lancer_optimisation_rendement = lancer_optimisation_value = lancer_filtrage_strict = False
    value_filter_strict = False  # Initialisation par défaut

    if objectif_optimisation == "🛡️ Minimisation de la variance.":
        lancer_optimisation_minvar = st.button("🚀 Lancer l'Optimisation Min Variance")

    elif objectif_optimisation == "🚀 Portefeuille efficient selon vos critères : Maximisation du Rendement selon votre niveau de risque.":
        lancer_optimisation_rendement = st.button("🚀 Lancer la Maximisation du Rendement")

    elif objectif_optimisation == "💎 Stratégie Value (P/E & P/B) : Maximisation de la valeur 1/PER + 1/Price to Book du portefeuille, selon votre niveau de risque.":
        lancer_optimisation_value = st.button("🚀 Lancer la Stratégie Value (P/E & P/B)")

    elif objectif_optimisation == "🧹 Filtrage Simple : Indice composé uniquement de titres satisfaisant individuellement les critères ESG. Possibilité de sélectionner uniquement les titres Value.":
        # Option Value
        value_filter_strict = st.checkbox("✅ Filtrer uniquement les titres Value", value=False)
        # Option équipondération
        equiponderation = st.checkbox("⚖️ Construire un portefeuille équipondéré (poids égal sur chaque titre), plutôt que pondéré selon les capitalisations boursières.", value=False)
        lancer_filtrage_strict = st.button("🚀 Lancer le Filtrage Simple")


# MinVariance
if lancer_optimisation_minvar:
    benchmark = st.session_state["user_choices"]["benchmark"]
    selected_benchmark = benchmark_map.get(benchmark, benchmark)

    # Chargement des données
    if selected_benchmark == "Indifférent":
        df_cours = final_data["dfComposants"]
        df_esg = final_data["dfESG"]
        df_cov = final_data["dfCovariancesConsolidées"]
        df_bics = final_data["dfBICS"]
        df_ratios = final_data["dfRatios"]
    else:
        df_cours = final_data[f"df{selected_benchmark}"]
        df_esg = final_data[f"df{selected_benchmark}_ESG"]
        df_cov = final_data[f"dfCovariances{selected_benchmark}"]
        df_bics = final_data[f"df{selected_benchmark}_BICS"]
        df_ratios = final_data[f"df{selected_benchmark}_ratios"]

    # Filtrage géographique
    if not st.session_state["indiff_pays"]:
        mask_pays = df_ratios["COUNTRY"].isin(st.session_state["user_choices"]["pays"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_pays] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Filtrage sectoriel
    niveau_BICS = st.session_state["user_choices"]["niveau_BICS"]
    colonne_BICS_selectionnee = bics_colonne_map[niveau_BICS]
    if not st.session_state["indiff_secteurs"]:
        mask_secteur = df_bics[colonne_BICS_selectionnee].isin(st.session_state["user_choices"]["secteurs"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_secteur] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Alignement des matrices covariance
    df_cov = df_cov.loc[df_cours.columns, df_cours.columns]

    # ESG Handling
    source_esg = st.session_state["user_choices"]["esg"]["source"]
    contraintes = []
    poids_esg = None
    seuil_esg = None

    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        df_esg["SA_ESG_RISK_SCR"] = pd.to_numeric(df_esg["SA_ESG_RISK_SCR"], errors='coerce')
        df_esg.dropna(subset=["SA_ESG_RISK_SCR"], inplace=True)
        df_cours = df_cours.loc[:, df_esg.index]
        df_cov = df_cov.loc[df_cours.columns, df_cours.columns]
        df_ratios = df_ratios.loc[df_esg.index]
        df_bics = df_bics.loc[df_esg.index]

        mapping_risk_seuil = {
            "Négligeable (0-10)": 9.99,   # Maximum < 10
            "Faible (10-20)": 19.99,      # Maximum < 20
            "Moyen (20-30)": 29.99,       # Maximum < 30
            "Élevé (30-40)": 39.99,       # Maximum < 40
            "Très Élevé (40+)": 100       # Pas de limite haute
        }

        selected_risk = st.session_state["user_choices"]["esg"].get("sa_esg_risk", "Moyen (20-30)")
        seuil_esg = mapping_risk_seuil.get(selected_risk, 30)
        poids_esg = np.array(df_esg["SA_ESG_RISK_SCR"]).reshape(-1, 1)

    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        df_esg = df_esg[df_esg["MSCI_ESG_RATING"].notna() & (df_esg["MSCI_ESG_RATING"] != "N.S.")]
        df_cours = df_cours.loc[:, df_esg.index]
        df_cov = df_cov.loc[df_cours.columns, df_cours.columns]
        df_ratios = df_ratios.loc[df_esg.index]
        df_bics = df_bics.loc[df_esg.index]
        seuil_esg = convertir_notation_msci_en_valeur(st.session_state["user_choices"]["esg"].get("msci_rating", "BBB"))
        poids_esg = np.array(df_esg["MSCI_ESG_RATING"].apply(convertir_notation_msci_en_valeur)).reshape(-1, 1)

    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        seuil_esg = np.array([
            st.session_state["user_choices"]["esg"].get("gouvernance", 5),
            st.session_state["user_choices"]["esg"].get("social", 5),
            st.session_state["user_choices"]["esg"].get("environnement", 5)
        ])
        esg_matrix = np.array([
            df_esg["GOVERNANCE_SCORE"].astype(float),
            df_esg["SOCIAL_SCORE"].astype(float),
            df_esg["ENVIRONMENTAL_SCORE"].astype(float)
        ]).T

    elif source_esg == "ESG_SCORE (0-10 : Global)":
        seuil_esg = st.session_state["user_choices"]["esg"].get("esg_score", 5)
        poids_esg = np.array(df_esg["ESG_SCORE"]).reshape(-1, 1)

    # Optimisation Minimum Variance
    N = len(df_cours.columns)
    w = cp.Variable(N)
    objectif = cp.Minimize(cp.quad_form(w, df_cov))
    contraintes = [cp.sum(w) == 1, w >= 0]

    # Application de la bonne contrainte ESG
    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        contraintes.append(esg_matrix.T @ w >= seuil_esg)
    elif source_esg in ["ESG_SCORE (0-10 : Global)", "MSCI_ESG_RATING (AAA-CCC)"]:
        contraintes.append((poids_esg.T @ w) >= seuil_esg)
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        contraintes.append((poids_esg.T @ w) <= seuil_esg)

    # Résolution
    problem = cp.Problem(objectif, contraintes)
    problem.solve()

    if w.value is None:
        st.error("❌ Nous sommes navrés, l'optimisation est impossible avec ces contraintes. Essayez d'assouplir ces dernières !")
        st.stop()

    # Résultats
    poids_opt = w.value.flatten()
    actifs_selectionnes = df_cours.columns[poids_opt > 1e-4]
    poids_selectionnes = poids_opt[poids_opt > 1e-4]

    # Calcul performance
    rendement_attendu = df_cours.pct_change().dropna().mean() * 252
    rendement_portefeuille = np.dot(rendement_attendu, poids_opt)
    vol_portefeuille = np.sqrt(np.dot(poids_opt.T, np.dot(df_cov, poids_opt)))
    sharpe = rendement_portefeuille / vol_portefeuille

    # Calcul ESG final
    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        final_esg = esg_matrix.T @ poids_opt
    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        score_msciesg = float((poids_esg.T @ poids_opt).item())
        final_esg = classer_portefeuille_msciesg(score_msciesg)
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        final_esg_score = float((poids_esg.T @ poids_opt).item())
        final_esg = classer_risque(final_esg_score)
    else:
        final_esg = float((poids_esg.T @ poids_opt).item())
    
        # Calcul des ratios financiers pondérés (P/E et P/B)
    pe_values = df_ratios.loc[actifs_selectionnes, "PE_RATIO"]
    pb_values = df_ratios.loc[actifs_selectionnes, "PX_TO_BOOK_RATIO"]
    poids_series = pd.Series(poids_selectionnes, index=actifs_selectionnes)

    # Filtrage des actifs valides pour les deux ratios
    ratios_valides = pe_values.notna() & pb_values.notna()
    actifs_valides = pe_values[ratios_valides].index

    if len(actifs_valides) > 0:
        poids_valides = poids_series.loc[actifs_valides]
        poids_valides /= poids_valides.sum()  # Renormalisation

        pe_pondere = np.dot(poids_valides, pe_values.loc[actifs_valides])
        pb_pondere = np.dot(poids_valides, pb_values.loc[actifs_valides])
    else:
        pe_pondere = np.nan
        pb_pondere = np.nan

    with tab3:
        st.header("📊 Résultats Complets de l'Optimisation Minimum Variance")
        
        # Récapitulatif des critères de sélection
        st.subheader("🧾 Récapitulatif des Critères de Sélection")

        # Benchmark
        st.markdown(f"📈 **Benchmark sélectionné** : `{st.session_state['user_choices']['benchmark']}`")

        # Géographie
        if st.session_state["indiff_pays"]:
            st.markdown("🌍 **Pays sélectionnés** : `Indifférent`")
        else:
            pays = st.session_state["user_choices"]["pays"]
            st.markdown(f"🌍 **Pays sélectionnés** : `{', '.join(pays)}`")

        # Secteurs
        if st.session_state["indiff_secteurs"]:
            st.markdown("🏢 **Secteurs sélectionnés** : `Indifférent`")
        else:
            secteurs = st.session_state["user_choices"]["secteurs"]
            niveau_bics = st.session_state["user_choices"]["niveau_BICS"]
            st.markdown(f"🏢 **Secteurs sélectionnés ({niveau_bics})** : `{', '.join(secteurs)}`")

        # ESG - selon la source
        source_esg = st.session_state["user_choices"]["esg"]["source"]
        st.markdown(f"♻️ **Source ESG sélectionnée** : `{source_esg}`")

        if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            seuil_gouv = st.session_state["user_choices"]["esg"]["gouvernance"]
            seuil_soc = st.session_state["user_choices"]["esg"]["social"]
            seuil_env = st.session_state["user_choices"]["esg"]["environnement"]
            st.markdown(
                f"🔎 **Seuils ESG exigés pour le portefeuille** : **Gouvernance** ➔ `{seuil_gouv}` | **Social** ➔ `{seuil_soc}` | **Environnement** ➔ `{seuil_env}`"
            )
        elif source_esg == "ESG_SCORE (0-10 : Global)":
            esg_score = st.session_state["user_choices"]["esg"]["esg_score"]
            st.markdown(f"🔎 **Score ESG Global minimal exigé pour le portefeuille** : `{esg_score}`")
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            msci_rating = st.session_state["user_choices"]["esg"]["msci_rating"]
            st.markdown(f"🔎 **Notation MSCI minimale exigée pour le portefeuille** : `{msci_rating}`")
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            esg_risk = st.session_state["user_choices"]["esg"]["sa_esg_risk"]
            st.markdown(f"🔎 **Risque ESG maximal autorisé pour le portefeuille** : `{esg_risk}`")

        # Performances
        st.subheader("📈 Récapitulatif des Performances")
        st.markdown(f"""
    - 🚀 **Rendement Annualisé du Portefeuille** : `{rendement_portefeuille:.2%}`  
    - 🛡️ **Volatilité** : `{vol_portefeuille:.2%}`  
    - ⚖️ **Sharpe Ratio** : `{sharpe:.2f}`  
    - 📖 **PER moyen pondéré** : `{pe_pondere:.2f}`  
    - 📖 **P/B moyen pondéré** : `{pb_pondere:.2f}`
        """)

        # ESG - Résultat final pondéré
        st.subheader("♻️ Résultat ESG Pondéré du Portefeuille")

        if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            st.markdown(f"⚠️ **Score Risque ESG Pondéré** : `{final_esg_score:.2f}`")
            st.markdown(f"🛑 **Classe de Risque ESG** : `{final_esg}`")

        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            st.markdown(f"📊 **MSCI ESG Rating pondéré du portefeuille** : `{final_esg}`")

        elif source_esg == "ESG_SCORE (0-10 : Global)":
            st.markdown(f"📊 **Score ESG Global pondéré** : `{final_esg:.2f}`")

        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            st.markdown(f"""
    - 🏛 **Gouvernance pondérée** : `{final_esg[0]:.2f}`  
    - 🤝 **Social pondéré** : `{final_esg[1]:.2f}`  
    - 🌿 **Environnement pondéré** : `{final_esg[2]:.2f}`
            """)

        # Taille finale du portefeuille
        st.subheader("📌 Taille finale du portefeuille")
        st.markdown(f"**Nombre d'actifs sélectionnés** : `{len(actifs_selectionnes)}`")

        # Composition Détaillée du Portefeuille
        st.subheader("📋 Composition Détaillée du Portefeuille")

        # Préparer les colonnes selon la source ESG
        df_detailed = pd.DataFrame(index=actifs_selectionnes)
        df_detailed["Pondération (%)"] = poids_selectionnes * 100
        df_detailed["Rendement Attendu (%)"] = rendement_attendu.loc[actifs_selectionnes].values * 100
        df_detailed["Contribution Rendement (%)"] = df_detailed["Pondération (%)"] * df_detailed["Rendement Attendu (%)"] / 100
        df_detailed["Pays"] = df_ratios.loc[actifs_selectionnes, "COUNTRY"]
        df_detailed["Secteur"] = df_bics.loc[actifs_selectionnes, colonne_BICS_selectionnee]

        # Ajout ESG dynamique selon la source sélectionnée
        if source_esg == "ESG_SCORE (0-10 : Global)":
            df_detailed["Score ESG Global"] = df_esg.loc[actifs_selectionnes, "ESG_SCORE"]
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            df_detailed["MSCI ESG Rating"] = df_esg.loc[actifs_selectionnes, "MSCI_ESG_RATING"]
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            df_detailed["Risque ESG Score"] = df_esg.loc[actifs_selectionnes, "SA_ESG_RISK_SCR"]
        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            df_detailed["Gouvernance"] = df_esg.loc[actifs_selectionnes, "GOVERNANCE_SCORE"]
            df_detailed["Social"] = df_esg.loc[actifs_selectionnes, "SOCIAL_SCORE"]
            df_detailed["Environnement"] = df_esg.loc[actifs_selectionnes, "ENVIRONMENTAL_SCORE"]

        df_detailed["P/E"] = df_ratios.loc[actifs_selectionnes, "PE_RATIO"]
        df_detailed["P/B"] = df_ratios.loc[actifs_selectionnes, "PX_TO_BOOK_RATIO"]
        df_detailed["Capitalisation (€)"] = df_ratios.loc[actifs_selectionnes, "CUR_MKT_CAP"] * 1_000_000


        # Trier par Pondération décroissante
        df_detailed = df_detailed.sort_values(by="Pondération (%)", ascending=False).reset_index().rename(columns={"index": "Actif"})

        st.dataframe(df_detailed.style.format({
            "Pondération (%)": "{:.2f}",
            "Contribution Rendement (%)": "{:.2f}",
            "Rendement Attendu (%)": "{:.2f}",
            "Score ESG Global": "{:.2f}",
            "Risque ESG Score": "{:.2f}",
            "Gouvernance": "{:.2f}",
            "Social": "{:.2f}",
            "Environnement": "{:.2f}",
            "P/E": "{:.2f}",
            "P/B": "{:.2f}",
            "Capitalisation (€)": "{:,.0f} €"
        }), height=600)

        # Répartition Géographique
        st.subheader("🌍 Répartition Géographique")

        # Préparation des données pour les graphiques
        repartition_pays_nb = df_detailed["Pays"].value_counts().reset_index()
        repartition_pays_nb.columns = ["Pays", "Nombre d'actifs"]
        repartition_pays_poids = df_detailed.groupby("Pays")["Pondération (%)"].sum().reset_index()

        repartition_secteurs_nb = df_detailed["Secteur"].value_counts().reset_index()
        repartition_secteurs_nb.columns = ["Secteur", "Nombre d'actifs"]
        repartition_secteurs_poids = df_detailed.groupby("Secteur")["Pondération (%)"].sum().reset_index()
        
        # Tri décroissant
        repartition_pays_nb = repartition_pays_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_pays_poids = repartition_pays_poids.sort_values("Pondération (%)", ascending=False)


        col1, col2 = st.columns(2)

        with col1:
            st.markdown("📌 **Nombre d'actifs par pays**")
            fig_nb_pays = px.bar(
                repartition_pays_nb,
                x="Pays", y="Nombre d'actifs",
                color="Pays",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_pays.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_pays, use_container_width=True)

        with col2:
            st.markdown("📌 **Répartition pondérée (%) par pays**")
            fig_poids_pays = px.bar(
                repartition_pays_poids,
                x="Pays", y="Pondération (%)",
                color="Pays",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_pays.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_pays, use_container_width=True)


        # Répartition Sectorielle
        st.subheader("🏢 Répartition Sectorielle")

        # Tri décroissant
        repartition_secteurs_nb = repartition_secteurs_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_secteurs_poids = repartition_secteurs_poids.sort_values("Pondération (%)", ascending=False)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("📌 **Nombre d'actifs par secteur**")
            fig_nb_sect = px.bar(
                repartition_secteurs_nb,
                x="Secteur", y="Nombre d'actifs",
                color="Secteur",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_sect.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_sect, use_container_width=True)

        with col4:
            st.markdown("📌 **Répartition pondérée (%) par secteur**")
            fig_poids_sect = px.bar(
                repartition_secteurs_poids,
                x="Secteur", y="Pondération (%)",
                color="Secteur",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_sect.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_sect, use_container_width=True)

        # Carte Interactive et Visualisation
        st.subheader("🌐 Exposition Géographique - Carte Interactive")
        repartition_geo = df_detailed.groupby('Pays')["Pondération (%)"].sum().reset_index()
        repartition_geo["ISO-3"] = repartition_geo["Pays"].apply(iso2_to_iso3)

        fig_map = px.choropleth(
            repartition_geo.dropna(subset=["ISO-3"]),
            locations="ISO-3",
            locationmode="ISO-3",
            color="Pondération (%)",
            hover_name="Pays",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            range_color=(0, repartition_geo['Pondération (%)'].max()),
            title="🌍 Exposition Géographique - Pondération (%)"
        )

        fig_map.update_geos(
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=True, coastlinecolor="lightgrey",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="LightBlue",
            projection_type="natural earth"
        )

        fig_map.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Poids (%)", tickformat=".2f")
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # Evolution Historique du Portefeuille et Comparatif avec les Indices
        st.subheader("📈 Évolution Historique du Portefeuille Optimisé vs Indices de Référence \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")

        # Récupération des rendements des indices traditionnels et ESG
        df_rendements_indices = final_data["dfRendementsIndices"].copy()
        df_rendements_indices_esg = final_data["dfRendementsIndicesESG"].copy()

        # Récupération des rendements des actifs sélectionnés
        df_rendements_portefeuille = df_cours[actifs_selectionnes].pct_change().dropna()

        # Construction du portefeuille (pondérations appliquées)
        poids_optimaux = poids_selectionnes
        perf_portefeuille = (df_rendements_portefeuille @ poids_optimaux).to_frame(name="Portefeuille Optimisé")

        # Construction des indices cumulés sans base 100 pour le moment
        perf_portefeuille_cum = (perf_portefeuille + 1).cumprod()
        indices_cum = (df_rendements_indices + 1).cumprod()
        indices_esg_cum = (df_rendements_indices_esg + 1).cumprod()

        # Concaténer pour trouver la première date commune (intersection)
        df_concat = pd.concat([perf_portefeuille_cum, indices_cum], axis=1, join='inner').dropna()
        date_base100 = df_concat.index[0]  # Première date d'intersection

        st.markdown(f"📌 **Date de base 100 alignée** : {date_base100.date()}")

        # Rebase à la date d'intersection
        perf_portefeuille_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100]) * 100
        indices_base100 = (indices_cum / indices_cum.loc[date_base100]) * 100

        # Concaténation finale
        df_comparatif = pd.concat([perf_portefeuille_base100, indices_base100], axis=1)

        # Graphique Plotly - Evolution Historique
        fig_perf = px.line(
            df_comparatif,
            labels={"value": "Indice (Base 100)", "index": "Date"},
            title="📈 Évolution Historique - Portefeuille vs Indices de Référence",
        )

        # Ajuster l'épaisseur des lignes
        for trace in fig_perf.data:
            if "Portefeuille Optimisé" in trace.name:
                trace.line.width = 3  # Portefeuille plus épais
            else:
                trace.line.width = 1.8  # Indices classiques un peu plus fins

        fig_perf.update_layout(
            legend_title_text="Indice",
            hovermode="x unified"
        )

        st.plotly_chart(fig_perf, use_container_width=True)

        # Statistiques des indices classiques
        afficher_comparaison_indices(df_rendements_indices, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices Classiques")
        
        # ESG - Même logique d'intersection et rebase
        st.subheader("🌱 Évolution Comparée avec les Indices ESG \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")

        # Rechercher la première date commune ESG / portefeuille
        df_concat_esg = pd.concat([perf_portefeuille_cum, indices_esg_cum], axis=1, join='inner').dropna()
        date_base100_esg = df_concat_esg.index[0]

        st.markdown(f"📌 **Date de base 100 ESG alignée** : {date_base100_esg.date()}")

        # Rebase ESG + indices classiques (en grisé)
        perf_portefeuille_esg_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100_esg]) * 100
        indices_esg_base100 = (indices_esg_cum / indices_esg_cum.loc[date_base100_esg]) * 100
        indices_classiques_base100_gris = (indices_cum / indices_cum.loc[date_base100_esg]) * 100

        # Fusion ESG
        df_comparatif_esg = pd.concat([perf_portefeuille_esg_base100, indices_esg_base100], axis=1)
        df_comparatif_gris = indices_classiques_base100_gris.copy()

        # Graphique ESG avec indices classiques en arrière-plan grisé
        fig_esg = go.Figure()

        # Portefeuille Optimisé - épais
        fig_esg.add_trace(go.Scatter(
            x=perf_portefeuille_esg_base100.index,
            y=perf_portefeuille_esg_base100.iloc[:, 0],
            mode='lines',
            name="Portefeuille Optimisé",
            visible=True,
            line=dict(width=3)
        ))

        # Indices Classiques - finesse moyenne mais masqués au départ
        for col in df_comparatif_gris.columns:
            fig_esg.add_trace(go.Scatter(
                x=df_comparatif_gris.index,
                y=df_comparatif_gris[col],
                mode='lines',
                name=f"{col} (Classique)",
                visible='legendonly',
                line=dict(width=1.8)
            ))

        # Indices ESG - trait standard
        for col in df_comparatif_esg.columns:
            if col != "Portefeuille Optimisé":
                fig_esg.add_trace(go.Scatter(
                    x=df_comparatif_esg.index,
                    y=df_comparatif_esg[col],
                    mode='lines',
                    name=col,
                    visible=True,
                    line=dict(width=1.8)
                ))

        fig_esg.update_layout(
            title="🌱 Performance Comparée - Portefeuille vs Indices Classiques et ESG",
            xaxis_title="Date",
            yaxis_title="Indice (Base 100)",
            legend_title_text="Indices",
            hovermode="x unified"
        )

        st.plotly_chart(fig_esg, use_container_width=True)

        # Statistiques des indices ESG
        afficher_comparaison_indices(df_rendements_indices_esg, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices ESG")

        with tab2:
            st.success("✅ Optimisation Minimum Variance terminée avec succès !")
            st.info("👉 Vous pouvez désormais consulter tous les résultats dans l'onglet **📊 Résultats de l'optimisation**.")


elif lancer_optimisation_rendement:
    benchmark = st.session_state["user_choices"]["benchmark"]
    selected_benchmark = benchmark_map.get(benchmark, benchmark)

    # Chargement des datasets selon le benchmark
    if selected_benchmark == "Indifférent":
        df_cours = final_data["dfComposants"]
        df_esg = final_data["dfESG"]
        df_cov = final_data["dfCovariancesConsolidées"]
        df_bics = final_data["dfBICS"]
        df_ratios = final_data["dfRatios"]
    else:
        df_cours = final_data[f"df{selected_benchmark}"]
        df_esg = final_data[f"df{selected_benchmark}_ESG"]
        df_cov = final_data[f"dfCovariances{selected_benchmark}"]
        df_bics = final_data[f"df{selected_benchmark}_BICS"]
        df_ratios = final_data[f"df{selected_benchmark}_ratios"]

    # Filtrage géographique
    if not st.session_state["indiff_pays"]:
        mask_pays = df_ratios["COUNTRY"].isin(st.session_state["user_choices"]["pays"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_pays] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Filtrage sectoriel
    niveau_BICS = st.session_state["user_choices"]["niveau_BICS"]
    colonne_BICS_selectionnee = bics_colonne_map[niveau_BICS]
    if not st.session_state["indiff_secteurs"]:
        mask_secteurs = df_bics[colonne_BICS_selectionnee].isin(st.session_state["user_choices"]["secteurs"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_secteurs] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Réalignement final après filtres géo/secteurs
    actifs_communs = df_cours.columns.intersection(df_esg.index).intersection(df_ratios.index).intersection(df_bics.index)
    df_cours = df_cours[actifs_communs]
    df_esg = df_esg.loc[actifs_communs]
    df_ratios = df_ratios.loc[actifs_communs]
    df_bics = df_bics.loc[actifs_communs]
    df_cov = df_cov.loc[actifs_communs, actifs_communs]

    # ESG
    source_esg = st.session_state["user_choices"]["esg"]["source"]
    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        df_esg["SA_ESG_RISK_SCR"] = pd.to_numeric(df_esg["SA_ESG_RISK_SCR"], errors='coerce')
        df_esg.dropna(subset=["SA_ESG_RISK_SCR"], inplace=True)

        # Réduction sur actifs communs après nettoyage ESG
        actifs_communs = df_cours.columns.intersection(df_esg.index)
        df_cours = df_cours[actifs_communs]
        df_esg = df_esg.loc[actifs_communs]
        df_ratios = df_ratios.loc[actifs_communs]
        df_bics = df_bics.loc[actifs_communs]
        df_cov = df_cov.loc[actifs_communs, actifs_communs]

        mapping_risk_seuil = {
            "Négligeable (0-10)": 9.99,
            "Faible (10-20)": 19.99,
            "Moyen (20-30)": 29.99,
            "Élevé (30-40)": 39.99,
            "Très Élevé (40+)": 100
        }
        selected_risk = st.session_state["user_choices"]["esg"].get("sa_esg_risk", "Moyen (20-30)")
        seuil_esg = mapping_risk_seuil.get(selected_risk, 30)
        poids_esg = np.array(df_esg["SA_ESG_RISK_SCR"]).reshape(-1, 1)

    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        df_esg = df_esg[df_esg["MSCI_ESG_RATING"].notna() & (df_esg["MSCI_ESG_RATING"] != "N.S.")]

        actifs_communs = df_cours.columns.intersection(df_esg.index)
        df_cours = df_cours[actifs_communs]
        df_esg = df_esg.loc[actifs_communs]
        df_ratios = df_ratios.loc[actifs_communs]
        df_bics = df_bics.loc[actifs_communs]
        df_cov = df_cov.loc[actifs_communs, actifs_communs]

        seuil_esg = convertir_notation_msci_en_valeur(st.session_state["user_choices"]["esg"].get("msci_rating", "BBB"))
        poids_esg = np.array(df_esg["MSCI_ESG_RATING"].apply(convertir_notation_msci_en_valeur)).reshape(-1, 1)

    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        seuil_esg = np.array([
            st.session_state["user_choices"]["esg"].get("gouvernance", 5),
            st.session_state["user_choices"]["esg"].get("social", 5),
            st.session_state["user_choices"]["esg"].get("environnement", 5)
        ])

        actifs_communs = df_cours.columns.intersection(df_esg.index)
        df_cours = df_cours[actifs_communs]
        df_esg = df_esg.loc[actifs_communs]
        df_ratios = df_ratios.loc[actifs_communs]
        df_bics = df_bics.loc[actifs_communs]
        df_cov = df_cov.loc[actifs_communs, actifs_communs]

        poids_esg = np.array([
            df_esg["GOVERNANCE_SCORE"].astype(float),
            df_esg["SOCIAL_SCORE"].astype(float),
            df_esg["ENVIRONMENTAL_SCORE"].astype(float)
        ]).T

    elif source_esg == "ESG_SCORE (0-10 : Global)":
        seuil_esg = st.session_state["user_choices"]["esg"].get("esg_score", 5)

        actifs_communs = df_cours.columns.intersection(df_esg.index)
        df_cours = df_cours[actifs_communs]
        df_esg = df_esg.loc[actifs_communs]
        df_ratios = df_ratios.loc[actifs_communs]
        df_bics = df_bics.loc[actifs_communs]
        df_cov = df_cov.loc[actifs_communs, actifs_communs]

        poids_esg = np.array(df_esg["ESG_SCORE"]).reshape(-1, 1)

    else:
        st.error(f"Source ESG '{source_esg}' non reconnue.")
        st.stop()

    # Calcul des rendements
    df_rendements = df_cours.pct_change().dropna()
    rendement_attendu = df_rendements.mean() * 252
    rendement_attendu = rendement_attendu.loc[df_cours.columns].values

    # Optimisation Maximisation Rendement
    w = cp.Variable(len(df_cours.columns))
    objectif = cp.Maximize(rendement_attendu @ w)
    contraintes = [cp.sum(w) == 1, w >= 0]

    # Contraintes ESG
    if source_esg in ["ESG par critères (Gouvernance, Social, Environnement)", "ESG_SCORE (0-10 : Global)", "MSCI_ESG_RATING (AAA-CCC)"]:
        contraintes.append((poids_esg.T @ w) >= seuil_esg)
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        contraintes.append((poids_esg.T @ w) <= seuil_esg)

    # Ajout de la contrainte de volatilité maximale si précisé par l'utilisateur
    if not st.session_state["user_choices"]["indiff_aversion_risque"]:
        seuil_volatilite = st.session_state["user_choices"]["volatilite_max"] / 100  # converti en décimal
        contraintes.append(cp.quad_form(w, df_cov) <= seuil_volatilite ** 2)

    # Résolution
    probleme = cp.Problem(objectif, contraintes)
    probleme.solve()

    if w.value is None:
        st.error("❌ Nous sommes navrés, l'optimisation est impossible avec ces contraintes. Essayez d'assouplir ces dernières !")
        st.stop()

    # Résultats
    poids_opt = w.value
    actifs_selectionnes = df_cours.columns[poids_opt > 1e-4]
    poids_selectionnes = poids_opt[poids_opt > 1e-4]
    rendement_portefeuille = np.dot(rendement_attendu, poids_opt)
    volatilite_portefeuille = np.sqrt(np.dot(poids_opt.T, np.dot(df_cov, poids_opt)))
    sharpe_ratio = rendement_portefeuille / volatilite_portefeuille

    # Calcul ESG final
    notation_esg_finale = poids_esg.T @ poids_opt
    if source_esg in ["ESG_SCORE (0-10 : Global)", "MSCI_ESG_RATING (AAA-CCC)", "SA_ESG_RISK_SCR (Négligeable à Très Élevé)"]:
        notation_esg_finale = notation_esg_finale.item()
    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        notation_esg_finale = notation_esg_finale.flatten()

    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        classe_risque = classer_risque(notation_esg_finale)

    if source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        notation_esg_finale = classer_portefeuille_msciesg(notation_esg_finale)

    # Calcul des ratios financiers pondérés (P/E et P/B)
    pe_values = df_ratios.loc[actifs_selectionnes, "PE_RATIO"]
    pb_values = df_ratios.loc[actifs_selectionnes, "PX_TO_BOOK_RATIO"]
    poids_series = pd.Series(poids_selectionnes, index=actifs_selectionnes)

    # Filtrage des actifs valides pour les deux ratios
    ratios_valides = pe_values.notna() & pb_values.notna()
    actifs_valides = pe_values[ratios_valides].index

    if len(actifs_valides) > 0:
        poids_valides = poids_series.loc[actifs_valides]
        poids_valides /= poids_valides.sum()  # Renormalisation

        pe_pondere = np.dot(poids_valides, pe_values.loc[actifs_valides])
        pb_pondere = np.dot(poids_valides, pb_values.loc[actifs_valides])
    else:
        pe_pondere = np.nan
        pb_pondere = np.nan

    # Résultats affichés dans tab3
    with tab3:
        st.header("📊 Résultats Complets de la Maximisation du Rendement")

        # Récapitulatif des critères de sélection
        st.subheader("🧾 Récapitulatif des Critères de Sélection")

        # Benchmark
        st.markdown(f"📈 **Benchmark sélectionné** : `{st.session_state['user_choices']['benchmark']}`")

        # Géographie
        if st.session_state["indiff_pays"]:
            st.markdown("🌍 **Pays sélectionnés** : `Indifférent`")
        else:
            pays = st.session_state["user_choices"]["pays"]
            st.markdown(f"🌍 **Pays sélectionnés** : `{', '.join(pays)}`")

        # Secteurs
        if st.session_state["indiff_secteurs"]:
            st.markdown("🏢 **Secteurs sélectionnés** : `Indifférent`")
        else:
            secteurs = st.session_state["user_choices"]["secteurs"]
            niveau_bics = st.session_state["user_choices"]["niveau_BICS"]
            st.markdown(f"🏢 **Secteurs sélectionnés ({niveau_bics})** : `{', '.join(secteurs)}`")

        # ESG - selon la source
        source_esg = st.session_state["user_choices"]["esg"]["source"]
        st.markdown(f"♻️ **Source ESG sélectionnée** : `{source_esg}`")

        if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            seuil_gouv = st.session_state["user_choices"]["esg"]["gouvernance"]
            seuil_social = st.session_state["user_choices"]["esg"]["social"]
            seuil_env = st.session_state["user_choices"]["esg"]["environnement"]
            st.markdown(
                f"🔎 **Seuils ESG exigés pour le portefeuille** : **Gouvernance** ➔ `{seuil_gouv}` | **Social** ➔ `{seuil_social}` | **Environnement** ➔ `{seuil_env}`"
            )
        elif source_esg == "ESG_SCORE (0-10 : Global)":
            esg_score = st.session_state["user_choices"]["esg"]["esg_score"]
            st.markdown(f"🔎 **Score ESG Global minimal exigé pour le portefeuille** : `{esg_score}`")
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            msci_rating = st.session_state["user_choices"]["esg"]["msci_rating"]
            st.markdown(f"🔎 **Notation MSCI minimale exigée pour le portefeuille** : `{msci_rating}`")
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            esg_risk = st.session_state["user_choices"]["esg"]["sa_esg_risk"]
            st.markdown(f"🔎 **Risque ESG maximal autorisé pour le portefeuille** : `{esg_risk}`")

        # Performances
        st.subheader("📈 Récapitulatif des Performances")

        if st.session_state["user_choices"]["indiff_aversion_risque"]:
            st.markdown("🎯 **Degré d'Aversion au Risque** : `Indifférent`")
        else:
            seuil_vol_utilisateur = st.session_state["user_choices"]["volatilite_max"]
            st.markdown(f"🎯 **Seuil de Volatilité maximal fixé** : `{seuil_vol_utilisateur}%`")

        st.markdown(f"""
    - 🚀 **Rendement Annualisé** : `{rendement_portefeuille:.2%}`  
    - 🛡️ **Volatilité du Portefeuille** : `{volatilite_portefeuille:.2%}` 
    - ⚖️ **Sharpe Ratio** : `{sharpe_ratio:.2f}`   
    - 📖 **PER (Price-to-Earnings Ratio) pondéré** : `{pe_pondere:.2f}`  
    - 📖 **P/B (Price-to-Book Ratio) pondéré** : `{pb_pondere:.2f}`
        """)

        # Résultat ESG pondéré
        st.subheader("♻️ Résultat ESG Pondéré du Portefeuille")

        if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            st.markdown(f"⚠️ **Score Risque ESG Pondéré** : `{notation_esg_finale:.2f}`")
            st.markdown(f"🛑 **Classe de Risque ESG** : `{classe_risque}`")

        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            st.markdown(f"📊 **MSCI ESG Rating pondéré du portefeuille** : `{notation_esg_finale}`")

        elif source_esg == "ESG_SCORE (0-10 : Global)":
            st.markdown(f"📊 **Score ESG Global pondéré** : `{notation_esg_finale:.2f}`")

        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            st.markdown(f"""
    - 🏛 **Gouvernance pondérée** : `{notation_esg_finale[0]:.2f}`  
    - 🤝 **Social pondéré** : `{notation_esg_finale[1]:.2f}`  
    - 🌿 **Environnement pondérée** : `{notation_esg_finale[2]:.2f}`
            """)

        # Nombre d'actifs retenus après optimisation
        st.subheader("📌 Taille finale du portefeuille")
        st.markdown(f"**Nombre d'actifs sélectionnés** : `{len(actifs_selectionnes)}`")

        # Calcul des rendements annualisés bien sous forme de Series indexée
        df_rendements = df_cours.pct_change().dropna()
        rendement_attendu = df_rendements.mean() * 252  # pd.Series indexée par les actifs

        # Composition Détaillée du Portefeuille
        st.subheader("📋 Composition Détaillée du Portefeuille")

        df_detailed = pd.DataFrame(index=actifs_selectionnes)
        df_detailed["Pondération (%)"] = poids_selectionnes * 100
        df_detailed["Rendement Attendu (%)"] = rendement_attendu.loc[actifs_selectionnes].values * 100
        df_detailed["Contribution Rendement (%)"] = df_detailed["Pondération (%)"] * df_detailed["Rendement Attendu (%)"] / 100
        df_detailed["Pays"] = df_ratios.loc[actifs_selectionnes, "COUNTRY"]
        df_detailed["Secteur"] = df_bics.loc[actifs_selectionnes, colonne_BICS_selectionnee]

        # Ajout ESG dynamique selon la source sélectionnée
        if source_esg == "ESG_SCORE (0-10 : Global)":
            df_detailed["Score ESG Global"] = df_esg.loc[actifs_selectionnes, "ESG_SCORE"]
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            df_detailed["MSCI ESG Rating"] = df_esg.loc[actifs_selectionnes, "MSCI_ESG_RATING"]
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            df_detailed["Risque ESG Score"] = df_esg.loc[actifs_selectionnes, "SA_ESG_RISK_SCR"]
        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            df_detailed["Gouvernance"] = df_esg.loc[actifs_selectionnes, "GOVERNANCE_SCORE"]
            df_detailed["Social"] = df_esg.loc[actifs_selectionnes, "SOCIAL_SCORE"]
            df_detailed["Environnement"] = df_esg.loc[actifs_selectionnes, "ENVIRONMENTAL_SCORE"]

        df_detailed["P/E"] = df_ratios.loc[actifs_selectionnes, "PE_RATIO"]
        df_detailed["P/B"] = df_ratios.loc[actifs_selectionnes, "PX_TO_BOOK_RATIO"]
        df_detailed["Capitalisation (€)"] = df_ratios.loc[actifs_selectionnes, "CUR_MKT_CAP"] * 1_000_000

        # Trier par Pondération décroissante
        df_detailed = df_detailed.sort_values(by="Pondération (%)", ascending=False).reset_index(drop=True)
        df_detailed.insert(0, "Actif", actifs_selectionnes.values)  # Colonne 'Actif' au début

        # Affichage
        st.dataframe(df_detailed.style.format({
            "Pondération (%)": "{:.2f}",
            "Contribution Rendement (%)": "{:.2f}",
            "Rendement Attendu (%)": "{:.2f}",
            "Score ESG Global": "{:.2f}",
            "Risque ESG Score": "{:.2f}",
            "Gouvernance": "{:.2f}",
            "Social": "{:.2f}",
            "Environnement": "{:.2f}",
            "P/E": "{:.2f}",
            "P/B": "{:.2f}",
            "Capitalisation (€)": "{:,.0f} €"
        }), height=600)

        # Répartition Géographique
        st.subheader("🌍 Répartition Géographique")

        # Préparation des données
        repartition_pays_nb = df_detailed["Pays"].value_counts().reset_index()
        repartition_pays_nb.columns = ["Pays", "Nombre d'actifs"]
        repartition_pays_poids = df_detailed.groupby("Pays")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_pays_nb = repartition_pays_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_pays_poids = repartition_pays_poids.sort_values("Pondération (%)", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("📌 **Nombre d'actifs par pays**")
            fig_nb_pays = px.bar(
                repartition_pays_nb,
                x="Pays", y="Nombre d'actifs",
                color="Pays",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_pays.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_pays, use_container_width=True)

        with col2:
            st.markdown("📌 **Répartition pondérée (%) par pays**")
            fig_poids_pays = px.bar(
                repartition_pays_poids,
                x="Pays", y="Pondération (%)",
                color="Pays",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_pays.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_pays, use_container_width=True)

        # Répartition Sectorielle
        st.subheader("🏢 Répartition Sectorielle")

        # Préparation des données
        repartition_secteurs_nb = df_detailed["Secteur"].value_counts().reset_index()
        repartition_secteurs_nb.columns = ["Secteur", "Nombre d'actifs"]
        repartition_secteurs_poids = df_detailed.groupby("Secteur")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_secteurs_nb = repartition_secteurs_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_secteurs_poids = repartition_secteurs_poids.sort_values("Pondération (%)", ascending=False)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("📌 **Nombre d'actifs par secteur**")
            fig_nb_sect = px.bar(
                repartition_secteurs_nb,
                x="Secteur", y="Nombre d'actifs",
                color="Secteur",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_sect.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_sect, use_container_width=True)

        with col4:
            st.markdown("📌 **Répartition pondérée (%) par secteur**")
            fig_poids_sect = px.bar(
                repartition_secteurs_poids,
                x="Secteur", y="Pondération (%)",
                color="Secteur",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_sect.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_sect, use_container_width=True)

        # Carte Interactive
        st.subheader("🌐 Exposition Géographique - Carte Interactive")
        repartition_geo = df_detailed.groupby('Pays')["Pondération (%)"].sum().reset_index()
        repartition_geo["ISO-3"] = repartition_geo["Pays"].apply(iso2_to_iso3)

        fig_map = px.choropleth(
            repartition_geo.dropna(subset=["ISO-3"]),
            locations="ISO-3",
            locationmode="ISO-3",
            color="Pondération (%)",
            hover_name="Pays",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            range_color=(0, repartition_geo['Pondération (%)'].max()),
            title="🌍 Exposition Géographique - Pondération (%)"
        )

        fig_map.update_geos(
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=True, coastlinecolor="lightgrey",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="LightBlue",
            projection_type="natural earth"
        )

        fig_map.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Poids (%)", tickformat=".2f")
        )

        st.plotly_chart(fig_map, use_container_width=True)


       # Evolution Historique du Portefeuille et Comparatif avec les Indices
        st.subheader("📈 Évolution Historique du Portefeuille Optimisé vs Indices de Référence \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")

        # Récupération des rendements des indices traditionnels et ESG
        df_rendements_indices = final_data["dfRendementsIndices"].copy()
        df_rendements_indices_esg = final_data["dfRendementsIndicesESG"].copy()

        # Rendements des actifs du portefeuille
        port_rendements = df_cours[actifs_selectionnes].pct_change().dropna()
        perf_portefeuille = (port_rendements @ poids_selectionnes).to_frame(name="Portefeuille Optimisé")

        # Cumuler les performances
        df_perf_cum = (perf_portefeuille + 1).cumprod()
        indices_cum = (df_rendements_indices + 1).cumprod()
        indices_esg_cum = (df_rendements_indices_esg + 1).cumprod()

        # Intersection des dates pour alignement base 100
        df_concat = pd.concat([df_perf_cum, indices_cum], axis=1, join='inner').dropna()
        date_base100 = df_concat.index[0]
        st.markdown(f"📌**Date de base 100 alignée** : `{date_base100.date()}`")

        # Rebase Portefeuille et Indices
        perf_port_base100 = (df_perf_cum / df_perf_cum.loc[date_base100]) * 100
        indices_base100 = (indices_cum / indices_cum.loc[date_base100]) * 100
        df_comparatif = pd.concat([perf_port_base100, indices_base100], axis=1)

        # Graphique Historique Classique
        fig_perf = px.line(df_comparatif, labels={"value": "Indice (Base 100)", "index": "Date"},
                        title="📌 Évolution Historique - Portefeuille Optimisé vs Indices de Référence")
        for trace in fig_perf.data:
            if "Portefeuille Optimisé" in trace.name:
                trace.line.width = 3
            else:
                trace.line.width = 1.8
                trace.line.dash = 'dot'
        fig_perf.update_layout(legend_title_text="Indice", hovermode="x unified")
        st.plotly_chart(fig_perf, use_container_width=True)

        # Comparaison avec les Indices Classiques
        afficher_comparaison_indices(df_rendements_indices, rendement_portefeuille, volatilite_portefeuille, "Comparaison : Portefeuille vs Indices Classiques")
        
        # ESG - Même logique d'intersection et rebase
        st.subheader("🌱 Évolution Comparée avec les Indices ESG \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")
        df_concat_esg = pd.concat([df_perf_cum, indices_esg_cum, indices_cum], axis=1, join='inner').dropna()
        date_base100_esg = df_concat_esg.index[0]
        st.markdown(f"📌 **Date de base 100 ESG alignée** : `{date_base100_esg.date()}`")

        # Rebase ESG et Classiques sur date ESG
        perf_port_esg_base100 = (df_perf_cum / df_perf_cum.loc[date_base100_esg]) * 100
        indices_esg_base100 = (indices_esg_cum / indices_esg_cum.loc[date_base100_esg]) * 100
        indices_classiques_gris = (indices_cum / indices_cum.loc[date_base100_esg]) * 100

        # Fusion
        df_comparatif_esg = pd.concat([perf_port_esg_base100, indices_esg_base100], axis=1)

        # Graphique ESG
        fig_esg = go.Figure()
        fig_esg.add_trace(go.Scatter(
            x=perf_port_esg_base100.index, y=perf_port_esg_base100.iloc[:, 0],
            mode='lines', name="Portefeuille Optimisé", line=dict(width=3)))

        for col in indices_classiques_gris.columns:
            fig_esg.add_trace(go.Scatter(
                x=indices_classiques_gris.index, y=indices_classiques_gris[col],
                mode='lines', name=f"{col} (Classique)", visible='legendonly',
                line=dict(width=1.8, dash='dot')))

        for col in indices_esg_base100.columns:
            fig_esg.add_trace(go.Scatter(
                x=indices_esg_base100.index, y=indices_esg_base100[col],
                mode='lines', name=col, line=dict(width=1.8)))

        fig_esg.update_layout(
            title="🌱 Performance Comparée - Portefeuille vs Indices ESG et Indices Classiques ",
            xaxis_title="Date", yaxis_title="Indice (Base 100)",
            legend_title_text="Indices", hovermode="x unified"
        )

        st.plotly_chart(fig_esg, use_container_width=True)

        
        # Comparaison avec les Indices ESG
        afficher_comparaison_indices(df_rendements_indices_esg, rendement_portefeuille, volatilite_portefeuille, "Comparaison : Portefeuille vs Indices ESG")
       
        with tab2:
            st.success("✅ Optimisation par maximisation du rendement effectuée avec succès !")
            st.info("👉 Consultez maintenant les résultats dans l'onglet **📊 Résultats de l'optimisation**.")


elif lancer_optimisation_value:
    benchmark = st.session_state["user_choices"]["benchmark"]
    selected_benchmark = benchmark_map.get(benchmark, benchmark)

    # Chargement des données
    if selected_benchmark == "Indifférent":
        df_cours = final_data["dfComposants"]
        df_esg = final_data["dfESG"]
        df_cov = final_data["dfCovariancesConsolidées"]
        df_bics = final_data["dfBICS"]
        df_ratios = final_data["dfRatios"]
    else:
        df_cours = final_data[f"df{selected_benchmark}"]
        df_esg = final_data[f"df{selected_benchmark}_ESG"]
        df_cov = final_data[f"dfCovariances{selected_benchmark}"]
        df_bics = final_data[f"df{selected_benchmark}_BICS"]
        df_ratios = final_data[f"df{selected_benchmark}_ratios"]

    # Filtres géographiques et sectoriels
    if not st.session_state["indiff_pays"]:
        mask_pays = df_ratios["COUNTRY"].isin(st.session_state["user_choices"]["pays"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_pays] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    niveau_BICS = st.session_state["user_choices"]["niveau_BICS"]
    colonne_BICS_selectionnee = bics_colonne_map[niveau_BICS]
    if not st.session_state["indiff_secteurs"]:
        mask_secteur = df_bics[colonne_BICS_selectionnee].isin(st.session_state["user_choices"]["secteurs"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_secteur] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Value Filter
    PX_Book_Ratio_Mean = df_ratios["PX_TO_BOOK_RATIO"].mean()
    PE_Ratio_Mean = df_ratios["PE_RATIO"].mean()
    mask_value = (df_ratios["PX_TO_BOOK_RATIO"] <= PX_Book_Ratio_Mean) & (df_ratios["PE_RATIO"] <= PE_Ratio_Mean)
    df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_value] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
    df_cours = df_cours.T
    df_cov = df_cov.loc[df_cours.columns, df_cours.columns]

    # Alignement ESG
    source_esg = st.session_state["user_choices"]["esg"]["source"]

    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        df_esg["SA_ESG_RISK_SCR"] = pd.to_numeric(df_esg["SA_ESG_RISK_SCR"], errors='coerce')
        df_esg.dropna(subset=["SA_ESG_RISK_SCR"], inplace=True)
        df_cours = df_cours.loc[:, df_esg.index]
        df_cov = df_cov.loc[df_cours.columns, df_cours.columns]
        df_ratios = df_ratios.loc[df_esg.index]
        df_bics = df_bics.loc[df_esg.index]

        mapping_risk_seuil = {
            "Négligeable (0-10)": 9.99,
            "Faible (10-20)": 19.99,
            "Moyen (20-30)": 29.99,
            "Élevé (30-40)": 39.99,
            "Très Élevé (40+)": 100
        }
        selected_risk = st.session_state["user_choices"]["esg"].get("sa_esg_risk", "Moyen (20-30)")
        seuil_esg = mapping_risk_seuil.get(selected_risk, 30)
        poids_esg = np.array(df_esg["SA_ESG_RISK_SCR"]).reshape(-1, 1)

    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        df_esg = df_esg[df_esg["MSCI_ESG_RATING"].notna() & (df_esg["MSCI_ESG_RATING"] != "N.S.")]
        df_cours = df_cours.loc[:, df_esg.index]
        df_cov = df_cov.loc[df_cours.columns, df_cours.columns]
        df_ratios = df_ratios.loc[df_esg.index]
        df_bics = df_bics.loc[df_esg.index]
        seuil_esg = convertir_notation_msci_en_valeur(st.session_state["user_choices"]["esg"].get("msci_rating", "BBB"))
        poids_esg = np.array(df_esg["MSCI_ESG_RATING"].apply(convertir_notation_msci_en_valeur)).reshape(-1, 1)

    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        seuil_esg = np.array([
            st.session_state["user_choices"]["esg"].get("gouvernance", 5),
            st.session_state["user_choices"]["esg"].get("social", 5),
            st.session_state["user_choices"]["esg"].get("environnement", 5)
        ])
        esg_matrix = np.array([
            df_esg["GOVERNANCE_SCORE"].astype(float),
            df_esg["SOCIAL_SCORE"].astype(float),
            df_esg["ENVIRONMENTAL_SCORE"].astype(float)
        ]).T  # (N, 3)

    elif source_esg == "ESG_SCORE (0-10 : Global)":
        seuil_esg = st.session_state["user_choices"]["esg"].get("esg_score", 5)
        poids_esg = np.array(df_esg["ESG_SCORE"]).reshape(-1, 1)

    else:
        st.error(f"Source ESG '{source_esg}' non reconnue.")
        st.stop()

    # Préparation Value (1/PER + 1/PB)
    inverse_pe = 1 / df_ratios["PE_RATIO"].replace(0, np.nan).fillna(0.01)
    inverse_pb = 1 / df_ratios["PX_TO_BOOK_RATIO"].replace(0, np.nan).fillna(0.01)
    value_score = inverse_pe + inverse_pb

    # Optimisation
    N = len(df_ratios)
    w = cp.Variable(N)
    objective = cp.Maximize(value_score.values @ w)
    constraints = [cp.sum(w) == 1, w >= 0]

    # Contraintes ESG
    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        constraints.append(esg_matrix.T @ w >= seuil_esg)
    elif source_esg in ["ESG_SCORE (0-10 : Global)", "MSCI_ESG_RATING (AAA-CCC)"]:
        constraints.append((poids_esg.T @ w) >= seuil_esg)
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        constraints.append((poids_esg.T @ w) <= seuil_esg)

    # Ajout de la contrainte de volatilité maximale si précisé par l'utilisateur
    if not st.session_state["user_choices"]["indiff_aversion_risque"]:
        seuil_volatilite = st.session_state["user_choices"]["volatilite_max"] / 100
        constraints.append(cp.quad_form(w, df_cov) <= seuil_volatilite ** 2)

    # Résolution
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if w.value is None:
        st.error("❌ Nous sommes navrés, l'optimisation est impossible avec ces contraintes. Essayez d'assouplir ces dernières !")
        st.stop()

    # Calcul et sécurisation immédiate des actifs et pondérations sélectionnés
    poids_opt = np.array(w.value).flatten()
    actifs_selectionnes = df_cours.columns[poids_opt > 1e-4]
    poids_selectionnes = poids_opt[poids_opt > 1e-4]

    # Calculs des performances du portefeuille
    rendement_attendu = df_cours.pct_change().dropna().mean() * 252
    rendement_portefeuille = np.dot(rendement_attendu, poids_opt)
    vol_portefeuille = np.sqrt(np.dot(poids_opt.T, np.dot(df_cov, poids_opt)))
    sharpe = rendement_portefeuille / vol_portefeuille

    # Ratios Value pondérés
    pe_ratio_portefeuille = np.dot(poids_opt, df_ratios["PE_RATIO"].values)
    pb_ratio_portefeuille = np.dot(poids_opt, df_ratios["PX_TO_BOOK_RATIO"].values)

    # Résultat ESG final et conversion MSCI ou Risk
    if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        final_esg = esg_matrix.T @ poids_opt
    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        score_msciesg = float((poids_esg.T @ poids_opt).item())
        final_esg = classer_portefeuille_msciesg(score_msciesg)
    elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        final_esg_score = float((poids_esg.T @ poids_opt).item())
        final_esg_class = classer_risque(final_esg_score)
        risk_respecte = final_esg_score <= seuil_esg
    else:
        final_esg = float((poids_esg.T @ poids_opt).item())

    # Affichage dans tab3
    with tab3:
        st.header("📊 Résultats Complets de l'Optimisation Value sous Contraintes ESG")

        # Récapitulatif des critères de sélection
        st.subheader("🧾 Récapitulatif des Critères de Sélection")

        # Benchmark
        st.markdown(f"📈 **Benchmark sélectionné** : `{st.session_state['user_choices']['benchmark']}`")

        # Géographie
        if st.session_state["indiff_pays"]:
            st.markdown("🌍 **Pays sélectionnés** : `Indifférent`")
        else:
            pays = st.session_state["user_choices"]["pays"]
            st.markdown(f"🌍 **Pays sélectionnés** : `{', '.join(pays)}`")

        # Secteurs
        if st.session_state["indiff_secteurs"]:
            st.markdown("🏢 **Secteurs sélectionnés** : `Indifférent`")
        else:
            secteurs = st.session_state["user_choices"]["secteurs"]
            niveau_bics = st.session_state["user_choices"]["niveau_BICS"]
            st.markdown(f"🏢 **Secteurs sélectionnés ({niveau_bics})** : `{', '.join(secteurs)}`")

        # ESG - selon la source
        source_esg = st.session_state["user_choices"]["esg"]["source"]
        st.markdown(f"♻️ **Source ESG sélectionnée** : `{source_esg}`")

        if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            seuil_gouv = st.session_state["user_choices"]["esg"]["gouvernance"]
            seuil_soc = st.session_state["user_choices"]["esg"]["social"]
            seuil_env = st.session_state["user_choices"]["esg"]["environnement"]
            st.markdown(
                f"🔎 **Seuils ESG exigés pour le portefeuille** : **Gouvernance** ➔ `{seuil_gouv}` | **Social** ➔ `{seuil_soc}` | **Environnement** ➔ `{seuil_env}`"
            )

        elif source_esg == "ESG_SCORE (0-10 : Global)":
            esg_score = st.session_state["user_choices"]["esg"]["esg_score"]
            st.markdown(f"🔎 **Score ESG Global minimal exigé pour le portefeuille** : `{esg_score}`")

        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            msci_rating = st.session_state["user_choices"]["esg"]["msci_rating"]
            st.markdown(f"🔎 **Notation MSCI minimale exigée pour le portefeuille** : `{msci_rating}`")

        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            esg_risk = st.session_state["user_choices"]["esg"]["sa_esg_risk"]
            st.markdown(f"🔎 **Risque ESG maximal autorisé pour le portefeuille** : `{esg_risk}`")

        # Récapitulatif Performances
        st.subheader("📈 Récapitulatif des Performances")

        if st.session_state["user_choices"]["indiff_aversion_risque"]:
            st.markdown("🎯 **Degré d'Aversion au Risque** : `Indifférent`")
        else:
            seuil_vol_utilisateur = st.session_state["user_choices"]["volatilite_max"]
            st.markdown(f"🎯 **Seuil de Volatilité maximal fixé** : `{seuil_vol_utilisateur}%`")

        st.markdown(f"""
        - 🚀 **Rendement Annualisé** : `{rendement_portefeuille:.2%}`
        - 🛡️ **Volatilité du Portefeuille** : `{vol_portefeuille:.2%}`
        - ⚖️ **Sharpe Ratio** : `{sharpe:.2f}`
        - 📖 **PER (Price-to-Earnings Ratio) pondéré** : `{pe_ratio_portefeuille:.2f}`
        - 📖 **P/B (Price-to-Book Ratio) pondéré** : `{pb_ratio_portefeuille:.2f}`
        """)

        # Résultat ESG pondéré
        st.subheader("♻️ Résultat ESG Pondéré du Portefeuille")

        if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            st.markdown(f"⚠️ **Score Risque ESG Pondéré** : `{final_esg_score:.2f}`")
            st.markdown(f"🛑 **Classe de Risque ESG** : `{final_esg_class}`")

        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            st.markdown(f"📊 **MSCI ESG Rating pondéré du portefeuille** : `{final_esg}`")

        elif source_esg == "ESG_SCORE (0-10 : Global)":
            st.markdown(f"📊 **Score ESG Global pondéré** : `{final_esg:.2f}`")

        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            # Décomposition des scores s'ils ne l'ont pas encore été
            score_g, score_s, score_e = final_esg[0], final_esg[1], final_esg[2]
            st.markdown(f"""
        - 🏛 **Gouvernance pondérée** : `{score_g:.2f}`  
        - 🤝 **Social pondéré** : `{score_s:.2f}`  
        - 🌿 **Environnement pondéré** : `{score_e:.2f}`
            """)

        # Taille finale du portefeuille
        st.subheader("📌 Taille finale du portefeuille")
        st.markdown(f"**Nombre d'actifs sélectionnés** : `{len(actifs_selectionnes)}`")

        # Composition Détaillée du Portefeuille
        st.subheader("📋 Composition Détaillée du Portefeuille")

        df_detailed = pd.DataFrame(index=actifs_selectionnes)
        df_detailed["Pondération (%)"] = poids_selectionnes * 100
        df_detailed["Rendement Attendu (%)"] = rendement_attendu.loc[actifs_selectionnes].values * 100
        df_detailed["Contribution Rendement (%)"] = df_detailed["Pondération (%)"] * df_detailed["Rendement Attendu (%)"] / 100
        df_detailed["Pays"] = df_ratios.loc[actifs_selectionnes, "COUNTRY"]
        df_detailed["Secteur"] = df_bics.loc[actifs_selectionnes, colonne_BICS_selectionnee]

        # Ajout ESG dynamique
        if source_esg == "ESG_SCORE (0-10 : Global)":
            df_detailed["Score ESG Global"] = df_esg.loc[actifs_selectionnes, "ESG_SCORE"]
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            df_detailed["MSCI ESG Rating"] = df_esg.loc[actifs_selectionnes, "MSCI_ESG_RATING"]
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            df_detailed["Risque ESG Score"] = df_esg.loc[actifs_selectionnes, "SA_ESG_RISK_SCR"]
        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            df_detailed["Gouvernance"] = df_esg.loc[actifs_selectionnes, "GOVERNANCE_SCORE"]
            df_detailed["Social"] = df_esg.loc[actifs_selectionnes, "SOCIAL_SCORE"]
            df_detailed["Environnement"] = df_esg.loc[actifs_selectionnes, "ENVIRONMENTAL_SCORE"]

        # Ajout Ratios Value
        df_detailed["P/E"] = df_ratios.loc[actifs_selectionnes, "PE_RATIO"]
        df_detailed["P/B"] = df_ratios.loc[actifs_selectionnes, "PX_TO_BOOK_RATIO"]
        df_detailed["Capitalisation (€)"] = df_ratios.loc[actifs_selectionnes, "CUR_MKT_CAP"] * 1_000_000

        df_detailed = df_detailed.sort_values(by="Pondération (%)", ascending=False).reset_index(drop=True)
        df_detailed.insert(0, "Actif", actifs_selectionnes.values)

        st.dataframe(df_detailed.style.format({
            "Pondération (%)": "{:.2f}",
            "Contribution Rendement (%)": "{:.2f}",
            "Rendement Attendu (%)": "{:.2f}",
            "Score ESG Global": "{:.2f}",
            "Risque ESG Score": "{:.2f}",
            "Gouvernance": "{:.2f}",
            "Social": "{:.2f}",
            "Environnement": "{:.2f}",
            "P/E": "{:.2f}",
            "P/B": "{:.2f}",
            "Capitalisation (€)": "{:,.0f} €"
        }), height=600)

        # Répartition Géographique
        st.subheader("🌍 Répartition Géographique")

        # Préparation des données
        repartition_pays_nb = df_detailed["Pays"].value_counts().reset_index()
        repartition_pays_nb.columns = ["Pays", "Nombre d'actifs"]
        repartition_pays_poids = df_detailed.groupby("Pays")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_pays_nb = repartition_pays_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_pays_poids = repartition_pays_poids.sort_values("Pondération (%)", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("📌 **Nombre d'actifs par pays**")
            fig_nb_pays = px.bar(
                repartition_pays_nb,
                x="Pays", y="Nombre d'actifs",
                color="Pays",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_pays.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_pays, use_container_width=True)

        with col2:
            st.markdown("📌 **Répartition pondérée (%) par pays**")
            fig_poids_pays = px.bar(
                repartition_pays_poids,
                x="Pays", y="Pondération (%)",
                color="Pays",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_pays.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_pays, use_container_width=True)

        # Répartition Sectorielle
        st.subheader("🏢 Répartition Sectorielle")

        # Préparation des données
        repartition_secteurs_nb = df_detailed["Secteur"].value_counts().reset_index()
        repartition_secteurs_nb.columns = ["Secteur", "Nombre d'actifs"]
        repartition_secteurs_poids = df_detailed.groupby("Secteur")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_secteurs_nb = repartition_secteurs_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_secteurs_poids = repartition_secteurs_poids.sort_values("Pondération (%)", ascending=False)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("📌 **Nombre d'actifs par secteur**")
            fig_nb_sect = px.bar(
                repartition_secteurs_nb,
                x="Secteur", y="Nombre d'actifs",
                color="Secteur",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_sect.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_sect, use_container_width=True)

        with col4:
            st.markdown("📌 **Répartition pondérée (%) par secteur**")
            fig_poids_sect = px.bar(
                repartition_secteurs_poids,
                x="Secteur", y="Pondération (%)",
                color="Secteur",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_sect.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_sect, use_container_width=True)

        # Carte Interactive
        st.subheader("🌐 Exposition Géographique - Carte Interactive")
        repartition_geo = df_detailed.groupby('Pays')["Pondération (%)"].sum().reset_index()
        repartition_geo["ISO-3"] = repartition_geo["Pays"].apply(iso2_to_iso3)

        fig_map = px.choropleth(
            repartition_geo.dropna(subset=["ISO-3"]),
            locations="ISO-3",
            locationmode="ISO-3",
            color="Pondération (%)",
            hover_name="Pays",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            range_color=(0, repartition_geo['Pondération (%)'].max()),
            title="🌍 Exposition Géographique - Pondération (%)"
        )

        fig_map.update_geos(
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=True, coastlinecolor="lightgrey",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="LightBlue",
            projection_type="natural earth"
        )

        fig_map.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Poids (%)", tickformat=".2f")
        )

        st.plotly_chart(fig_map, use_container_width=True)

        # Évolution Historique du Portefeuille Value vs Indices de Référence
        st.subheader("📈 Évolution Historique du Portefeuille Optimisé (Value) vs Indices de Référence \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")

        # Récupération des rendements des indices traditionnels et ESG
        df_rendements_indices = final_data["dfRendementsIndices"].copy()
        df_rendements_indices_esg = final_data["dfRendementsIndicesESG"].copy()

        # Rendements des actifs sélectionnés
        df_rendements_portefeuille = df_cours[actifs_selectionnes].pct_change().dropna()

        # Performance cumulée du portefeuille optimisé
        perf_portefeuille = (df_rendements_portefeuille @ poids_selectionnes).to_frame(name="Portefeuille Optimisé")
        perf_portefeuille_cum = (perf_portefeuille + 1).cumprod()
        indices_cum = (df_rendements_indices + 1).cumprod()
        indices_esg_cum = (df_rendements_indices_esg + 1).cumprod()

        # Trouver la première date commune pour l'alignement Base 100
        df_concat = pd.concat([perf_portefeuille_cum, indices_cum], axis=1, join='inner').dropna()
        date_base100 = df_concat.index[0]
        st.markdown(f"📌 **Date de base 100 alignée** : `{date_base100.date()}`")

        # Rebase à cette date
        perf_portefeuille_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100]) * 100
        indices_base100 = (indices_cum / indices_cum.loc[date_base100]) * 100

        # Fusion pour le graphique
        df_comparatif = pd.concat([perf_portefeuille_base100, indices_base100], axis=1)

        # Graphique Plotly - Historique
        fig_perf = px.line(
            df_comparatif,
            labels={"value": "Indice (Base 100)", "index": "Date"},
            title="📈 Évolution Historique - Portefeuille Optimisé (Value) vs Indices de Référence",
        )

        for trace in fig_perf.data:
            if "Portefeuille Optimisé" in trace.name:
                trace.line.width = 3  # Plus épais
            else:
                trace.line.width = 1.8  # Indices plus fins

        fig_perf.update_layout(legend_title_text="Indice", hovermode="x unified")
        st.plotly_chart(fig_perf, use_container_width=True)

        # Comparaison avec les Indices Classiques
        afficher_comparaison_indices(df_rendements_indices, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices Classiques")

        # Évolution comparée avec les Indices ESG
        st.subheader("🌱 Évolution Comparée avec les Indices ESG \nBase 100 sur la première date commune à tous les actifs du portefeuille et aux indices comparés.")

        df_concat_esg = pd.concat([perf_portefeuille_cum, indices_esg_cum], axis=1, join='inner').dropna()
        date_base100_esg = df_concat_esg.index[0]
        st.markdown(f"📌 **Date de base 100 ESG alignée** : `{date_base100_esg.date()}`")

        perf_portefeuille_esg_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100_esg]) * 100
        indices_esg_base100 = (indices_esg_cum / indices_esg_cum.loc[date_base100_esg]) * 100
        indices_classiques_base100_gris = (indices_cum / indices_cum.loc[date_base100_esg]) * 100

        # Construction du graphique ESG
        df_comparatif_esg = pd.concat([perf_portefeuille_esg_base100, indices_esg_base100], axis=1)
        df_comparatif_gris = indices_classiques_base100_gris.copy()

        fig_esg = go.Figure()

        # Portefeuille Value
        fig_esg.add_trace(go.Scatter(
            x=perf_portefeuille_esg_base100.index,
            y=perf_portefeuille_esg_base100.iloc[:, 0],
            mode='lines',
            name="Portefeuille Optimisé",
            visible=True,
            line=dict(width=3)
        ))

        # Indices Classiques grisés
        for col in df_comparatif_gris.columns:
            fig_esg.add_trace(go.Scatter(
                x=df_comparatif_gris.index,
                y=df_comparatif_gris[col],
                mode='lines',
                name=f"{col} (Classique)",
                visible='legendonly',
                line=dict(width=1.8, dash='dot')
            ))

        # Indices ESG
        for col in df_comparatif_esg.columns:
            if col != "Portefeuille Optimisé":
                fig_esg.add_trace(go.Scatter(
                    x=df_comparatif_esg.index,
                    y=df_comparatif_esg[col],
                    mode='lines',
                    name=col,
                    visible=True,
                    line=dict(width=1.8)
                ))

        fig_esg.update_layout(
            title="🌱 Performance Comparée - Portefeuille Optimisé (Value) vs Indices Classiques et ESG",
            xaxis_title="Date",
            yaxis_title="Indice (Base 100)",
            legend_title_text="Indices",
            hovermode="x unified"
        )

        st.plotly_chart(fig_esg, use_container_width=True)

        # Comparaison avec les Indices ESG
        afficher_comparaison_indices(df_rendements_indices_esg, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices ESG")

    with tab2:
        st.success("✅ Optimisation Value effectuée avec succès !")
        st.info("👉 Rendez-vous dans l'onglet **📊 Résultats de l'optimisation** pour consulter les résultats détaillés.")


# Programme de filtrage strict
elif lancer_filtrage_strict:

    # Sélection du benchmark
    benchmark = st.session_state["user_choices"]["benchmark"]
    selected_benchmark = benchmark_map.get(benchmark, benchmark)

    # Chargement des datasets selon le benchmark
    if selected_benchmark == "Indifférent":
        df_cours = final_data["dfComposants"]
        df_esg = final_data["dfESG"]
        df_bics = final_data["dfBICS"]
        df_ratios = final_data["dfRatios"]
    else:
        df_cours = final_data[f"df{selected_benchmark}"]
        df_esg = final_data[f"df{selected_benchmark}_ESG"]
        df_bics = final_data[f"df{selected_benchmark}_BICS"]
        df_ratios = final_data[f"df{selected_benchmark}_ratios"]

    # Filtrage géographique
    if not st.session_state["indiff_pays"]:
        mask_pays = df_ratios["COUNTRY"].isin(st.session_state["user_choices"]["pays"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_pays] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Filtrage sectoriel
    niveau_BICS = st.session_state["user_choices"]["niveau_BICS"]
    colonne_BICS_selectionnee = bics_colonne_map[niveau_BICS]
    if not st.session_state["indiff_secteurs"]:
        mask_secteur = df_bics[colonne_BICS_selectionnee].isin(st.session_state["user_choices"]["secteurs"])
        df_cours, df_ratios, df_bics, df_esg = [df.loc[mask_secteur] for df in [df_cours.T, df_ratios, df_bics, df_esg]]
        df_cours = df_cours.T

    # Filtrage Value (optionnel)
    if value_filter_strict:
        px_book_mean = df_ratios["PX_TO_BOOK_RATIO"].mean()
        pe_ratio_mean = df_ratios["PE_RATIO"].mean()
        value_mask = (df_ratios["PX_TO_BOOK_RATIO"] <= px_book_mean) & (df_ratios["PE_RATIO"] <= pe_ratio_mean)
    else:
        value_mask = pd.Series(True, index=df_ratios.index)

    # Filtrage ESG
    source_esg = st.session_state["user_choices"]["esg"]["source"]

    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        df_esg["SA_ESG_RISK_SCR"] = pd.to_numeric(df_esg["SA_ESG_RISK_SCR"], errors='coerce')
        df_esg.dropna(subset=["SA_ESG_RISK_SCR"], inplace=True)
        mapping_risk_seuil = {"Négligeable (0-10)": 10, "Faible (10-20)": 20, "Moyen (20-30)": 30, "Élevé (30-40)": 40, "Très Élevé (40+)": 100}
        selected_risk = st.session_state["user_choices"]["esg"].get("sa_esg_risk", "Moyen (20-30)")
        seuil_esg = mapping_risk_seuil.get(selected_risk, 30)
        esg_mask = df_esg["SA_ESG_RISK_SCR"] <= seuil_esg

    elif source_esg == "ESG_SCORE (0-10 : Global)":
        seuil_esg = st.session_state["user_choices"]["esg"].get("esg_score", 5)
        esg_mask = df_esg["ESG_SCORE"] >= seuil_esg

    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        df_esg = df_esg[df_esg["MSCI_ESG_RATING"].notna() & (df_esg["MSCI_ESG_RATING"] != "N.S.")]
        seuil_esg = convertir_notation_msci_en_valeur(st.session_state["user_choices"]["esg"].get("msci_rating", "BBB"))
        esg_mask = df_esg["MSCI_ESG_RATING"].apply(convertir_notation_msci_en_valeur) >= seuil_esg

    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        seuil_gov = st.session_state["user_choices"]["esg"].get("gouvernance", 5)
        seuil_soc = st.session_state["user_choices"]["esg"].get("social", 5)
        seuil_env = st.session_state["user_choices"]["esg"].get("environnement", 5)
        esg_mask = (
            (df_esg["GOVERNANCE_SCORE"] >= seuil_gov) &
            (df_esg["SOCIAL_SCORE"] >= seuil_soc) &
            (df_esg["ENVIRONMENTAL_SCORE"] >= seuil_env)
        )
    else:
        st.error("❌ Source ESG non reconnue.")
        st.stop()

    # Application des masques de filtrage combinés
    final_mask = value_mask & esg_mask
    df_filtered = df_cours.loc[:, final_mask]

    if "CUR_MKT_CAP" not in df_ratios.columns:
        st.error("🚨 Colonne 'CUR_MKT_CAP' absente dans les ratios financiers.")
        st.stop()

    actifs_filtrés = df_filtered.columns

    # Vérification 
    if len(actifs_filtrés) == 0:
        st.error("❌ Aucun actif ne satisfait les critères de filtrage. Veuillez élargir vos contraintes (secteurs, pays, ESG, etc.).")
        st.stop()

    caps_filtrées = df_ratios.loc[actifs_filtrés, "CUR_MKT_CAP"]

    # Nettoyage des données manquantes
    if caps_filtrées.isnull().any():
        st.warning("⚠️ Capitalisations manquantes détectées, lignes supprimées.")
        caps_filtrées = caps_filtrées.dropna()

    # Calcul des poids de chaque actif
    if equiponderation:
        poids_caps = pd.Series(1 / len(actifs_filtrés), index=actifs_filtrés)
    else:
        # Pondération par capitalisation boursière
        poids_caps = caps_filtrées / caps_filtrées.sum()

    
    # Calcul des rendements annualisés
    df_rendements = df_cours.pct_change().dropna()
    rendements_annualisés = df_rendements.mean() * 252  # Annualisé

    # Calcul de la performance du portefeuille filtré et pondéré
    rendement_portefeuille = np.dot(rendements_annualisés.loc[actifs_filtrés].values, poids_caps)

    # Calcul de la volatilité du portefeuille
    cov_filtrée = df_rendements.loc[:, actifs_filtrés].cov() * 252
    vol_portefeuille = np.sqrt(np.dot(poids_caps.T, np.dot(cov_filtrée, poids_caps)))

    # Calcul du Sharpe Ratio
    sharpe = rendement_portefeuille / vol_portefeuille

    # Calcul des contributions individuelles
    rendement_attendu_par_actif = rendements_annualisés.loc[actifs_filtrés].values * 100
    contribution_rendement = poids_caps.values * rendement_attendu_par_actif

     # Calcul ESG pondéré final
    score_esg_final = None
    if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
        score_esg_final = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "SA_ESG_RISK_SCR"].values)
        classe_risque_esg = classer_risque(score_esg_final)
    elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
        score_msciesg = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "MSCI_ESG_RATING"].apply(convertir_notation_msci_en_valeur).values)
        score_esg_final = classer_portefeuille_msciesg(score_msciesg)
    elif source_esg == "ESG_SCORE (0-10 : Global)":
        score_esg_final = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "ESG_SCORE"].values)
    elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
        score_g = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "GOVERNANCE_SCORE"].values)
        score_s = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "SOCIAL_SCORE"].values)
        score_e = np.dot(poids_caps, df_esg.loc[actifs_filtrés, "ENVIRONMENTAL_SCORE"].values)
    
    # Nettoyage des ratios financiers (P/E et P/B)
    pe_ratios = df_ratios.loc[actifs_filtrés, "PE_RATIO"]
    pb_ratios = df_ratios.loc[actifs_filtrés, "PX_TO_BOOK_RATIO"]
    
    # On garde uniquement les actifs où les deux ratios sont valides
    ratios_valides = pe_ratios.notna() & pb_ratios.notna()
    actifs_valides = actifs_filtrés[ratios_valides]

    if len(actifs_valides) > 0:
        # Extraction des poids et des valeurs valides
        poids_valides = poids_caps[ratios_valides]
        poids_valides = poids_valides / poids_valides.sum()  # Renormalisation

        # Calcul des moyennes pondérées
        pe_pondere = np.dot(poids_valides, pe_ratios[ratios_valides])
        pb_pondere = np.dot(poids_valides, pb_ratios[ratios_valides])
    else:
        pe_pondere = np.nan
        pb_pondere = np.nan

    # Résultat Dashboard du Filtrage Strict
    with tab3:
        st.header("📊 Résultats Complets du Filtrage Strict et Construction du Portefeuille Pondéré par Capitalisation")

        # Récapitulatif des critères de sélection
        st.subheader("🧾 Récapitulatif des Critères de Sélection")

        # Benchmark
        st.markdown(f"📈 **Benchmark sélectionné** : `{st.session_state['user_choices']['benchmark']}`")

        # Géographie
        if st.session_state["indiff_pays"]:
            st.markdown("🌍 **Pays sélectionnés** : `Indifférent`")
        else:
            pays = st.session_state["user_choices"]["pays"]
            st.markdown(f"🌍 **Pays sélectionnés** : `{', '.join(pays)}`")

        # Secteurs
        if st.session_state["indiff_secteurs"]:
            st.markdown("🏢 **Secteurs sélectionnés** : `Indifférent`")
        else:
            secteurs = st.session_state["user_choices"]["secteurs"]
            niveau_bics = st.session_state["user_choices"]["niveau_BICS"]
            st.markdown(f"🏢 **Secteurs sélectionnés ({niveau_bics})** : `{', '.join(secteurs)}`")

      # ESG - selon la source
        source_esg = st.session_state["user_choices"]["esg"]["source"]
        st.markdown(f"♻️ **Source ESG sélectionnée** : `{source_esg}`")

        if source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            seuil_gouv = st.session_state["user_choices"]["esg"]["gouvernance"]
            seuil_soc = st.session_state["user_choices"]["esg"]["social"]
            seuil_env = st.session_state["user_choices"]["esg"]["environnement"]
            st.markdown(
                f"🔎 **Seuils ESG exigés pour chaque titre** : **Gouvernance** ➔ `{seuil_gouv}` | **Social** ➔ `{seuil_soc}` | **Environnement** ➔ `{seuil_env}`"
            )

        elif source_esg == "ESG_SCORE (0-10 : Global)":
            esg_score = st.session_state["user_choices"]["esg"]["esg_score"]
            st.markdown(f"🔎 **Score ESG Global minimal exigé pour chaque titre** : `{esg_score}`")

        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            msci_rating = st.session_state["user_choices"]["esg"]["msci_rating"]
            st.markdown(f"🔎 **Notation MSCI minimale exigée pour chaque titre** : `{msci_rating}`")

        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            esg_risk = st.session_state["user_choices"]["esg"]["sa_esg_risk"]
            st.markdown(f"🔎 **Risque ESG maximal autorisé pour chaque titre** : `{esg_risk}`")
                    
        st.subheader("📈 Récapitulatif des Performances")
        
        # Précision sur le filtrage Value
        if value_filter_strict:
            st.markdown("ℹ️ **Filtrage Value appliqué** : *Seuls les titres avec un **P/E** et **P/B** inférieurs à la moyenne ont été retenus.*")
        else:
            st.markdown("ℹ️ **Filtrage Value non appliqué** : *Tous les titres ont été évalués sans contrainte sur les ratios **P/E** et **P/B**.*")

         # Affiche la méthode de pondération choisie
        if equiponderation:
            st.markdown("⚖️ **Méthode de Pondération** : *Portefeuille Équipondéré (poids égal sur chaque actif sélectionné).*")
        else:
            st.markdown("💰 **Méthode de Pondération** : *Portefeuille Pondéré par Capitalisation Boursière*")

        st.markdown(f"""
        - 🚀 **Rendement Annualisé du Portefeuille Pondéré** : `{rendement_portefeuille:.2%}`
        - 🛡️ **Volatilité** : `{vol_portefeuille:.2%}`
        - ⚖️ **Sharpe Ratio** : `{sharpe:.2f}`
        - 📖 **PER (Price-to-Earnings Ratio) pondéré** : `{pe_pondere:.2f}`
        - 📖 **P/B (Price-to-Book Ratio) pondéré** : `{pb_pondere:.2f}`
        """)

        # Résultat ESG Pondéré
        st.subheader("♻️ Résultat ESG Pondéré du Portefeuille")
        if source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            st.markdown(f"⚠️ **Score Risque ESG Pondéré** : `{score_esg_final:.2f}`")
            st.markdown(f"🛑 **Classe de Risque ESG** : `{classe_risque_esg}`")
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            st.markdown(f"📊 **MSCI ESG Rating pondéré du portefeuille** : `{score_esg_final}`")
        elif source_esg == "ESG_SCORE (0-10 : Global)":
            st.markdown(f"📊 **Score ESG Global pondéré** : `{score_esg_final:.2f}`")
        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            st.markdown(f"""
            - 🏛 **Gouvernance pondérée** : `{score_g:.2f}`
            - 🤝 **Social pondéré** : `{score_s:.2f}`
            - 🌿 **Environnement pondéré** : `{score_e:.2f}`
            """)

        # Taille finale du portefeuille
        st.subheader("📌 Taille finale du portefeuille")
        st.markdown(f"**Nombre d'actifs sélectionnés** : `{len(actifs_filtrés)}`")

        # Composition Détaillée du Portefeuille Pondéré
        st.subheader("📋 Composition Détaillée du Portefeuille Pondéré")
        df_details = pd.DataFrame(index=actifs_filtrés)
        df_details["Pondération (%)"] = poids_caps * 100
        df_details["Rendement Attendu (%)"] = rendement_attendu_par_actif
        df_details["Contribution Rendement (%)"] = contribution_rendement
        df_details["Pays"] = df_ratios.loc[actifs_filtrés, "COUNTRY"]
        df_details["Secteur"] = df_bics.loc[actifs_filtrés, colonne_BICS_selectionnee]

        # ESG enrichissement dynamique
        if source_esg == "ESG_SCORE (0-10 : Global)":
            df_details["Score ESG Global"] = df_esg.loc[actifs_filtrés, "ESG_SCORE"]
        elif source_esg == "MSCI_ESG_RATING (AAA-CCC)":
            df_details["MSCI ESG Rating"] = df_esg.loc[actifs_filtrés, "MSCI_ESG_RATING"]
        elif source_esg == "SA_ESG_RISK_SCR (Négligeable à Très Élevé)":
            df_details["Risque ESG Score"] = df_esg.loc[actifs_filtrés, "SA_ESG_RISK_SCR"]
        elif source_esg == "ESG par critères (Gouvernance, Social, Environnement)":
            df_details["Gouvernance"] = df_esg.loc[actifs_filtrés, "GOVERNANCE_SCORE"]
            df_details["Social"] = df_esg.loc[actifs_filtrés, "SOCIAL_SCORE"]
            df_details["Environnement"] = df_esg.loc[actifs_filtrés, "ENVIRONMENTAL_SCORE"]

        df_details["P/E"] = df_ratios.loc[actifs_filtrés, "PE_RATIO"]
        df_details["P/B"] = df_ratios.loc[actifs_filtrés, "PX_TO_BOOK_RATIO"]
        df_details["Capitalisation (€)"] = df_ratios.loc[actifs_filtrés, "CUR_MKT_CAP"] * 1_000_000

        df_details = df_details.sort_values(by="Pondération (%)", ascending=False).reset_index().rename(columns={"index": "Actif"})

        st.dataframe(df_details.style.format({
            "Pondération (%)": "{:.2f}",
            "Rendement Attendu (%)": "{:.2f}",
            "Contribution Rendement (%)": "{:.2f}",
            "Score ESG Global": "{:.2f}",
            "Risque ESG Score": "{:.2f}",
            "Gouvernance": "{:.2f}",
            "Social": "{:.2f}",
            "Environnement": "{:.2f}",
            "P/E": "{:.2f}",
            "P/B": "{:.2f}",
            "Capitalisation (€)": "{:,.0f} €"
        }), height=600)

        # Répartition Géographique
        st.subheader("🌍 Répartition Géographique")

        # Préparation des données
        repartition_pays_nb = df_details["Pays"].value_counts().reset_index()
        repartition_pays_nb.columns = ["Pays", "Nombre d'actifs"]
        repartition_pays_poids = df_details.groupby("Pays")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_pays_nb = repartition_pays_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_pays_poids = repartition_pays_poids.sort_values("Pondération (%)", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("📌 **Nombre d'actifs par pays**")
            fig_nb_pays = px.bar(
                repartition_pays_nb,
                x="Pays", y="Nombre d'actifs",
                color="Pays",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_pays.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_pays, use_container_width=True)

        with col2:
            st.markdown("📌 **Répartition pondérée (%) par pays**")
            fig_poids_pays = px.bar(
                repartition_pays_poids,
                x="Pays", y="Pondération (%)",
                color="Pays",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_pays.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_pays.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_pays, use_container_width=True)

        # Répartition Sectorielle
        st.subheader("🏢 Répartition Sectorielle")

        # Préparation des données
        repartition_secteurs_nb = df_details["Secteur"].value_counts().reset_index()
        repartition_secteurs_nb.columns = ["Secteur", "Nombre d'actifs"]
        repartition_secteurs_poids = df_details.groupby("Secteur")["Pondération (%)"].sum().reset_index()

        # Tri décroissant
        repartition_secteurs_nb = repartition_secteurs_nb.sort_values("Nombre d'actifs", ascending=False)
        repartition_secteurs_poids = repartition_secteurs_poids.sort_values("Pondération (%)", ascending=False)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("📌 **Nombre d'actifs par secteur**")
            fig_nb_sect = px.bar(
                repartition_secteurs_nb,
                x="Secteur", y="Nombre d'actifs",
                color="Secteur",
                text="Nombre d'actifs",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_nb_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_nb_sect.update_traces(textposition="outside")
            st.plotly_chart(fig_nb_sect, use_container_width=True)

        with col4:
            st.markdown("📌 **Répartition pondérée (%) par secteur**")
            fig_poids_sect = px.bar(
                repartition_secteurs_poids,
                x="Secteur", y="Pondération (%)",
                color="Secteur",
                text="Pondération (%)",
                color_discrete_sequence=px.colors.sequential.YlGnBu
            )
            fig_poids_sect.update_layout(xaxis_title="", yaxis_title="", showlegend=False)
            fig_poids_sect.update_traces(texttemplate='%{text:.2f}%', textposition="outside")
            st.plotly_chart(fig_poids_sect, use_container_width=True)

        # Carte interactive
        st.subheader("🌐 Exposition Géographique - Carte Interactive")
        geo_repartition = df_details.groupby('Pays')["Pondération (%)"].sum().reset_index()
        geo_repartition["ISO-3"] = geo_repartition["Pays"].apply(iso2_to_iso3)

        fig_geo = px.choropleth(
            geo_repartition.dropna(subset=["ISO-3"]),
            locations="ISO-3",
            locationmode="ISO-3",
            color="Pondération (%)",
            hover_name="Pays",
            color_continuous_scale=px.colors.sequential.YlGnBu,
            range_color=(0, geo_repartition["Pondération (%)"].max()),
            title="🌍 Exposition Géographique Pondérée (%)"
        )
        fig_geo.update_geos(
            showcountries=True, countrycolor="lightgrey",
            showcoastlines=True, coastlinecolor="lightgrey",
            showland=True, landcolor="whitesmoke",
            showocean=True, oceancolor="LightBlue",
            projection_type="natural earth"
        )
        fig_geo.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            coloraxis_colorbar=dict(title="Poids (%)", tickformat=".2f")
        )
        st.plotly_chart(fig_geo, use_container_width=True)

        # Évolution Historique du Portefeuille Pondéré vs Indices de Référence
        st.subheader("📈 Évolution Historique du Portefeuille Pondéré vs Indices de Référence \nBase 100 sur la première date commune à tous les actifs et indices comparés.")

        # Chargement des rendements des indices (à faire ici pour être sûr qu'ils sont définis)
        df_rendements_indices = final_data["dfRendementsIndices"].copy()
        df_rendements_indices_esg = final_data["dfRendementsIndicesESG"].copy()

        # Rendements du portefeuille (filtré et pondéré)
        df_rendements_portefeuille = df_filtered.pct_change().dropna()
        perf_portefeuille = (df_rendements_portefeuille @ poids_caps).to_frame(name="Portefeuille Pondéré")
        perf_portefeuille_cum = (perf_portefeuille + 1).cumprod()

        # Indices classiques cumulés
        indices_cum = (df_rendements_indices + 1).cumprod()
        indices_esg_cum = (df_rendements_indices_esg + 1).cumprod()

        # Alignement des bases (recherche de la 1ère date commune)
        df_concat = pd.concat([perf_portefeuille_cum, indices_cum], axis=1, join='inner').dropna()
        date_base100 = df_concat.index[0]
        st.markdown(f"📌 **Date de base 100 alignée** : `{date_base100.date()}`")

        # Rebase de la performance et des indices
        perf_portefeuille_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100]) * 100
        indices_base100 = (indices_cum / indices_cum.loc[date_base100]) * 100

        # 🛠️ Fusion pour le graphique
        df_comparatif = pd.concat([perf_portefeuille_base100, indices_base100], axis=1)

        # Graphique historique général
        fig_perf = px.line(
            df_comparatif,
            labels={"value": "Indice (Base 100)", "index": "Date"},
            title="📈 Évolution Historique - Portefeuille Pondéré vs Indices de Référence",
        )
        for trace in fig_perf.data:
            if "Portefeuille Pondéré" in trace.name:
                trace.line.width = 3
            else:
                trace.line.width = 1.8
        fig_perf.update_layout(legend_title_text="Indice", hovermode="x unified")
        st.plotly_chart(fig_perf, use_container_width=True)

        # Comparaison avec les Indices Classiques
        afficher_comparaison_indices(df_rendements_indices, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices Classiques")

        # Évolution ESG comparée
        st.subheader("🌱 Évolution Comparée avec les Indices ESG \nBase 100 sur la première date commune.")

        # Intersection pour l'alignement ESG
        df_concat_esg = pd.concat([perf_portefeuille_cum, indices_esg_cum], axis=1, join='inner').dropna()
        date_base100_esg = df_concat_esg.index[0]
        st.markdown(f"📌 **Date de base 100 ESG alignée** : `{date_base100_esg.date()}`")

        # Rebase ESG et classique
        perf_portefeuille_esg_base100 = (perf_portefeuille_cum / perf_portefeuille_cum.loc[date_base100_esg]) * 100
        indices_esg_base100 = (indices_esg_cum / indices_esg_cum.loc[date_base100_esg]) * 100
        indices_classiques_base100_gris = (indices_cum / indices_cum.loc[date_base100_esg]) * 100

        # Fusion pour le graphique ESG
        df_comparatif_esg = pd.concat([perf_portefeuille_esg_base100, indices_esg_base100], axis=1)
        df_comparatif_gris = indices_classiques_base100_gris.copy()

        # Graphique Plotly ESG
        fig_esg = go.Figure()
        fig_esg.add_trace(go.Scatter(
            x=perf_portefeuille_esg_base100.index,
            y=perf_portefeuille_esg_base100.iloc[:, 0],
            mode='lines',
            name="Portefeuille Pondéré",
            line=dict(width=3)
        ))

        # Indices Classiques grisés
        for col in df_comparatif_gris.columns:
            fig_esg.add_trace(go.Scatter(
                x=df_comparatif_gris.index,
                y=df_comparatif_gris[col],
                mode='lines',
                name=f"{col} (Classique)",
                visible='legendonly',
                line=dict(width=1.8, dash='dot')
            ))

        # Indices ESG
        for col in df_comparatif_esg.columns:
            if col != "Portefeuille Pondéré":
                fig_esg.add_trace(go.Scatter(
                    x=df_comparatif_esg.index,
                    y=df_comparatif_esg[col],
                    mode='lines',
                    name=col,
                    line=dict(width=1.8)
                ))

        fig_esg.update_layout(
            title="🌱 Performance Comparée - Portefeuille Pondéré vs Indices Classiques et ESG",
            xaxis_title="Date",
            yaxis_title="Indice (Base 100)",
            legend_title_text="Indices",
            hovermode="x unified"
        )

        st.plotly_chart(fig_esg, use_container_width=True)

        # Comparaison avec les Indices ESG
        afficher_comparaison_indices(df_rendements_indices_esg, rendement_portefeuille, vol_portefeuille, "Comparaison : Portefeuille vs Indices ESG")
    
    with tab2:
        st.success("✅ Filtrage terminé avec succès !")
        st.info("👉 Vous pouvez maintenant consulter les résultats détaillés dans l'onglet **📊 Résultats de l'optimisation**.")


with tab4:
    st.header("🧾 Bilan par Action")

    # --- Sélection ---
    titre_indice = st.selectbox("Indice de référence :", ["S&P 500", "Stoxx 600", "CAC 40"])
    df_selected_index = {"S&P 500": dfSP500, "Stoxx 600": dfSTOXX600, "CAC 40": dfCAC40}[titre_indice]
    indice_mapping = {"S&P 500": "SPX Index", "Stoxx 600": "SXXP Index", "CAC 40": "CAC Index"}
    esg_mapping = {"Stoxx 600": "SXXPESGX Index", "CAC 40": "CACESG Index"}

    actions_disponibles = df_selected_index.columns.tolist()
    actions_selectionnees = st.multiselect("Sélectionnez des actions :", actions_disponibles, default=[actions_disponibles[0]])
    if not actions_selectionnees:
        st.warning("Veuillez sélectionner au moins une action.")
        st.stop()

    ponderation = st.radio("Méthode de pondération :", ["Égalitaire", "Par capitalisation boursière"])

    min_date = df_selected_index.index.min()
    max_date = df_selected_index.index.max()
    col1, col2 = st.columns(2)
    with col1:
        date_debut = st.date_input("📅 Date de début", value=max_date - datetime.timedelta(days=365), min_value=min_date, max_value=max_date)
    with col2:
        date_fin = st.date_input("📅 Date de fin", value=max_date, min_value=min_date, max_value=max_date)

    if date_debut >= date_fin:
        st.warning("La date de début doit être antérieure à la date de fin.")
        st.stop()

    # --- Données & Calculs ---
    df_filtered = df_selected_index.loc[date_debut:date_fin]
    prices = df_filtered[actions_selectionnees].dropna()
    returns = prices.pct_change().dropna()
    if returns.empty:
        st.warning("Pas de données suffisantes.")
        st.stop()

    cum_returns = (1 + returns).cumprod()
    annualized_returns = (1 + returns).prod() ** (252 / len(returns)) - 1
    annualized_volatilities = returns.std() * np.sqrt(252)

    if ponderation == "Par capitalisation boursière":
        try:
            caps = dfRatios.loc[actions_selectionnees]["CUR_MKT_CAP"]
            weights = caps / caps.sum()
        except:
            st.warning("Erreur sur la capitalisation, on passe en égalitaire.")
            weights = np.repeat(1 / len(actions_selectionnees), len(actions_selectionnees))
    else:
        weights = np.repeat(1 / len(actions_selectionnees), len(actions_selectionnees))

    weighted_returns = returns @ weights
    portfolio_cum = (1 + weighted_returns).cumprod()
    annualized_return_portfolio = (1 + weighted_returns).prod() ** (252 / len(weighted_returns)) - 1
    annualized_volatility_portfolio = weighted_returns.std() * np.sqrt(252)

    # Benchmarks
    market_prices = dfIndices[indice_mapping[titre_indice]].dropna().loc[date_debut:date_fin]
    market_returns = market_prices.pct_change().dropna()
    indice_annualized_return = (1 + market_returns).prod() ** (252 / len(market_returns)) - 1
    indice_annualized_volatility = market_returns.std() * np.sqrt(252)
    market_cum = (1 + market_returns).cumprod()

    benchmark_esg_perf = None
    if titre_indice in esg_mapping:
        try:
            esg_prices = dfIndices_ESG[esg_mapping[titre_indice]].dropna().loc[date_debut:date_fin]
            esg_returns = esg_prices.pct_change().dropna()
            benchmark_esg_perf = (1 + esg_returns).prod() ** (252 / len(esg_returns)) - 1
        except:
            pass

    # Sharpe, Beta, Drawdown
    taux_sans_risque = st.number_input("Taux sans risque (%)", 0.0, 10.0, 2.0) / 100
    sharpe_ratios = (annualized_returns - taux_sans_risque) / annualized_volatilities
    betas = returns.apply(lambda x: x.cov(market_returns) / market_returns.var(), axis=0)
    drawdown = (cum_returns - cum_returns.cummax()) / cum_returns.cummax()
    max_drawdowns = drawdown.min()

    # --- Statistiques Globales ---
    st.markdown("### 📊 Statistiques Globales")
    cols = st.columns(5)
    cols[0].metric("📈 Rendement Portefeuille", f"{annualized_return_portfolio:.2%}")
    cols[1].metric("📉 Volatilité Portefeuille", f"{annualized_volatility_portfolio:.2%}")
    cols[2].metric("📊 Sharpe", f"{(annualized_return_portfolio - taux_sans_risque) / annualized_volatility_portfolio:.2f}")
    cols[3].metric(f"📈 {titre_indice}", f"{indice_annualized_return:.2%}")
    cols[4].metric(f"📉 Volatilité {titre_indice}", f"{indice_annualized_volatility:.2%}")

    if benchmark_esg_perf:
        st.info(f"📘 Rendement {titre_indice} ESG : **{benchmark_esg_perf:.2%}**")

    # Rebase des performances cumulées en base 100
    portfolio_base100 = (portfolio_cum / portfolio_cum.iloc[0]) * 100
    market_base100 = (market_cum / market_cum.iloc[0]) * 100
    esg_base100 = None
    if benchmark_esg_perf:
        esg_cum = (1 + esg_returns).cumprod()
        esg_base100 = (esg_cum / esg_cum.iloc[0]) * 100

    # --- Graphique Historique ---
    st.markdown("### 📉 Comparatif Historique : Portefeuille vs Benchmark")
    fig, ax = plt.subplots(figsize=(10, 5))
    portfolio_base100.plot(ax=ax, label="Portefeuille")
    market_base100.plot(ax=ax, label=titre_indice)
    if esg_base100 is not None:
        esg_base100.plot(ax=ax, label=f"{titre_indice} ESG")
    ax.set_ylabel("Indice (Base 100)")
    ax.set_title("Performances Cumulées (Base 100)")
    ax.legend()
    st.pyplot(fig)

    # --- Tableau des Stats ---
    st.markdown("### 🧾 Détails Statistiques par Action")

    df_perf = pd.DataFrame({
        "Rendement Annualisé (%)": annualized_returns * 100,
        "Volatilité Annualisée (%)": annualized_volatilities * 100,
        "Sharpe": sharpe_ratios,
        "Beta": betas,
        "Max Drawdown (%)": max_drawdowns * 100,
    })

    # Ajout des ratios fondamentaux
    for col in ["PE_RATIO", "PX_TO_BOOK_RATIO", "CUR_MKT_CAP"]:
        if col in dfRatios.columns:
            df_perf[col] = dfRatios.loc[actions_selectionnees, col]

    # ─── Tri décroissant par performance (rendement annuel) ─────────────────────
    df_perf = df_perf.sort_values(by="Rendement Annualisé (%)", ascending=False)

    # ─── Style avec dégradés sélectionnés ───────────────────────────────────────
    styled_perf = df_perf.style\
        .background_gradient(cmap="Greens", subset=["Rendement Annualisé (%)"])\
        .background_gradient(cmap="YlGnBu", subset=["Sharpe"])\
        .background_gradient(cmap="RdPu_r", subset=["Max Drawdown (%)"])\
        .background_gradient(cmap="PuBuGn", subset=["PE_RATIO", "PX_TO_BOOK_RATIO"])\
        .background_gradient(cmap="GnBu", subset=["CUR_MKT_CAP"])\
        .format({
            "Rendement Annualisé (%)": "{:.2f}",
            "Volatilité Annualisée (%)": "{:.2f}",
            "Sharpe": "{:.2f}",
            "Beta": "{:.2f}",
            "Max Drawdown (%)": "{:.2f}",
            "PE_RATIO": "{:.2f}",
            "PX_TO_BOOK_RATIO": "{:.2f}",
            "CUR_MKT_CAP": "{:,.0f} M€"
        })

    # ─── Détails interactifs avec sous-parties ────────────────────────────────────────
    st.markdown("### 🔎 Cliquer pour cacher le détail par Action")

    for action in actions_selectionnees:
        with st.expander(f"{action}",  expanded=True):
            # Récupération des données
            pe_ratio = dfRatios.loc[action, "PE_RATIO"] if "PE_RATIO" in dfRatios.columns else np.nan
            pb_ratio = dfRatios.loc[action, "PX_TO_BOOK_RATIO"] if "PX_TO_BOOK_RATIO" in dfRatios.columns else np.nan
            mkt_cap = dfRatios.loc[action, "CUR_MKT_CAP"] * 1_000_000 if "CUR_MKT_CAP" in dfRatios.columns else np.nan

            bics_1 = dfBICS.loc[action, "bics_level_1_sector_name"] if "bics_level_1_sector_name" in dfBICS.columns else "N/A"
            bics_2 = dfBICS.loc[action, "bics_level_2_industry_group_name"] if "bics_level_2_industry_group_name" in dfBICS.columns else "N/A"
            bics_3 = dfBICS.loc[action, "bics_level_3_industry_name"] if "bics_level_3_industry_name" in dfBICS.columns else "N/A"
            bics_4 = dfBICS.loc[action, "bics_level_4_sub_industry_name"] if "bics_level_4_sub_industry_name" in dfBICS.columns else "N/A"

            env_score = dfESG.loc[action, "ENVIRONMENTAL_SCORE"] if "ENVIRONMENTAL_SCORE" in dfESG.columns and action in dfESG.index else "Indisponible"
            soc_score = dfESG.loc[action, "SOCIAL_SCORE"] if "SOCIAL_SCORE" in dfESG.columns and action in dfESG.index else "Indisponible"
            gov_score = dfESG.loc[action, "GOVERNANCE_SCORE"] if "GOVERNANCE_SCORE" in dfESG.columns and action in dfESG.index else "Indisponible"
            global_esg = dfESG.loc[action, "ESG_SCORE"] if "ESG_SCORE" in dfESG.columns and action in dfESG.index else "Indisponible"
            risk_score = dfESG.loc[action, "SA_ESG_RISK_SCR"] if "SA_ESG_RISK_SCR" in dfESG.columns and action in dfESG.index else "Indisponible"
            msci_score = dfESG.loc[action, "MSCI_ESG_RATING"] if "MSCI_ESG_RATING" in dfESG.columns and action in dfESG.index else "Indisponible"

            st.markdown(f"""
    #### 🏢 Secteur & Capitalisation
    - 🏷️ **Secteur BICS Niveau 1** : {bics_1}  
    - 🧱 **Niveau 2** : {bics_2}  
    - 🧬 **Niveau 3** : {bics_3}  
    - 🔬 **Niveau 4** : {bics_4}  
    - 💰 **Capitalisation boursière** : {mkt_cap:,.0f} €

    #### 📊 Données Financières
    - 📈 **Rendement annualisé** : {annualized_returns[action]:.2%}  
    - 📉 **Volatilité annualisée** : {annualized_volatilities[action]:.2%}  
    - 📖 **PER (Price-to-Earnings)** : {pe_ratio:.2f}  
    - 📘 **P/B (Price-to-Book)** : {pb_ratio:.2f}  
    - ⚖️ **Sharpe** : {sharpe_ratios[action]:.2f}  
    - 📊 **Bêta** : {betas[action]:.2f}  
    - 🩸 **Max Drawdown** : {max_drawdowns[action]:.2%}

    #### ♻️ Scores ESG
    - 🌿 **Score Environnemental** : {env_score}  
    - 🤝 **Score Social** : {soc_score}  
    - 🏛️ **Score Gouvernance** : {gov_score}  
    - 📊 **ESG Global (0-10)** : {global_esg}  
    - 🔺 **Risque ESG (SA)** : {risk_score}  
    - 🏷️ **Notation MSCI** : {msci_score}
            """)

            fig, ax = plt.subplots()
            prices[action].plot(ax=ax, title=f"{action} - Cours Réels", color="#1f77b4")
            ax.set_ylabel("Cours ($)")
            ax.grid(True, linestyle="--", alpha=0.6)
            st.pyplot(fig)
            
    # Palette pastel plus homogène et esthétique
    pastel_palette = sns.color_palette("pastel")

    # ─── Répartition Sectorielle avec niveau BICS ────────────────────────────────
    st.markdown("### 🏢 Répartition Sectorielle")

    try:
        niveau_bics = st.selectbox(
            "Sélectionnez le niveau de secteur BICS :", 
            options=[
                ("bics_level_1_sector_name", "Niveau 1 - Secteur"),
                ("bics_level_2_industry_group_name", "Niveau 2 - Groupe Industriel"),
                ("bics_level_3_industry_name", "Niveau 3 - Industrie"),
                ("bics_level_4_sub_industry_name", "Niveau 4 - Sous-Industrie")
            ],
            format_func=lambda x: x[1]
        )

        bics_column = niveau_bics[0]
        secteurs = dfBICS.loc[actions_selectionnees].dropna(subset=[bics_column])[bics_column]
        sector_counts = secteurs.value_counts(normalize=True).mul(100)

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        bars = ax1.barh(
            sector_counts.index,
            sector_counts.values,
            color=pastel_palette[:len(sector_counts)]
        )
        ax1.set_xlabel("Pondération (%)")
        ax1.set_title(f"Répartition par {niveau_bics[1]}")
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontsize=10)

        st.pyplot(fig1)

        sector_table = pd.DataFrame({"Secteur": secteurs}).reset_index().groupby("Secteur")["index"].apply(list).reset_index()
        sector_table.columns = [niveau_bics[1], "Actions associées"]
        st.dataframe(sector_table)

    except Exception as e:
        st.warning(f"Erreur dans la répartition sectorielle : {e}")

    # ─── Répartition Géographique ───────────────────────────────────────────────
    st.markdown("### 🌍 Répartition Géographique")
    try:
        pays = dfRatios.loc[actions_selectionnees].dropna(subset=["COUNTRY"])["COUNTRY"]
        pays_counts = pays.value_counts(normalize=True).mul(100)

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        bars = ax2.barh(
            pays_counts.index,
            pays_counts.values,
            color=pastel_palette[:len(pays_counts)]
        )
        ax2.set_xlabel("Pondération (%)")
        ax2.set_title("Répartition par Pays")
        ax2.invert_yaxis()
        ax2.grid(axis='x', linestyle='--', alpha=0.5)

        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, f"{width:.2f}%", va='center', fontsize=10)

        st.pyplot(fig2)

        geo_table = pd.DataFrame({"Pays": pays}).reset_index().groupby("Pays")["index"].apply(list).reset_index()
        geo_table.columns = ["Pays", "Actions associées"]
        st.dataframe(geo_table)
    except Exception as e:
        st.warning(f"Erreur dans la répartition géographique : {e}")

    # ─── Top 3 Sharpe ──────────────────────────────────────────────────────────
    st.markdown("### 💡 Top 3 Actions selon Sharpe")
    top_actions = sharpe_ratios.sort_values(ascending=False).head(3)
    st.dataframe(top_actions.to_frame(name="Sharpe").style.background_gradient(cmap="PuBuGn"))
            
    # ─── Scores ESG (avec pondération + affichage intelligent) ─────────────────────
    st.markdown("### ♻️ Moyennes des Scores ESG")

    try:
        sources_possibles = {
            "ESG Triple Score (E/S/G)": ["ENVIRONMENTAL_SCORE", "SOCIAL_SCORE", "GOVERNANCE_SCORE"],
            "ESG Global Score (0-10)": ["ESG_SCORE"],
            "ESG Risk Score (SA)": ["SA_ESG_RISK_SCR"],
            "MSCI ESG Rating": ["MSCI_ESG_RATING"]
        }

        sources_disponibles = {
            name: cols for name, cols in sources_possibles.items()
            if all(col in dfESG.columns for col in cols)
        }

        source_choisie = st.selectbox("Choisissez la source ESG à analyser :", list(sources_disponibles.keys()))
        colonnes_esg = sources_disponibles[source_choisie]

        # Pondérations
        if ponderation == "Par capitalisation boursière":
            try:
                caps = dfRatios.loc[actions_selectionnees]["CUR_MKT_CAP"]
                poids = caps / caps.sum()
            except:
                poids = pd.Series(np.repeat(1 / len(actions_selectionnees), len(actions_selectionnees)), index=actions_selectionnees)
        else:
            poids = pd.Series(np.repeat(1 / len(actions_selectionnees), len(actions_selectionnees)), index=actions_selectionnees)

        # ESG Triple
        if source_choisie == "ESG Triple Score (E/S/G)":
            df_scores = dfESG.loc[actions_selectionnees, colonnes_esg]
            scores_pondérés = (df_scores.T * poids).T
            moyennes = scores_pondérés.sum()

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ["#2E8B57", "#3CB371", "#66CDAA"]
            bars = ax.barh(moyennes.index.str.replace("_SCORE", ""), moyennes.values, color=colors, edgecolor='black')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
            ax.set_xlim(0, 10)
            ax.set_xlabel("Score moyen (sur 10)")
            ax.set_title("Moyennes pondérées - E / S / G")
            ax.invert_yaxis()
            st.pyplot(fig)

            df_affichage = df_scores.copy()
            df_affichage.columns = ["Environnement", "Social", "Gouvernance"]
            df_affichage = df_affichage.reset_index().rename(columns={"index": "Action"})
            st.dataframe(df_affichage)

        # MSCI ESG
        elif source_choisie == "MSCI ESG Rating":
            ratings = dfESG.loc[actions_selectionnees]["MSCI_ESG_RATING"]
            scores_numeriques = ratings.map(convertir_notation_msci_en_valeur)
            score_pondere = (scores_numeriques * poids).sum()
            notation = classer_portefeuille_msciesg(score_pondere)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(["MSCI ESG Rating"], [score_pondere], color="#91C8E4", edgecolor='black')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{notation} ({width:.2f})", va='center')
            ax.set_xlim(0, 7)
            ax.set_xlabel("Score converti (1 à 7)")
            ax.set_title("MSCI ESG Rating pondéré")
            ax.invert_yaxis()
            st.pyplot(fig)

            df_affichage = pd.DataFrame({
                "Action": ratings.index,
                "Notation MSCI": ratings.values,
                "Score MSCI (1-7)": scores_numeriques.values
            })
            st.dataframe(df_affichage)

        # Risk Score (SA)
        elif source_choisie == "ESG Risk Score (SA)":
            scores = dfESG.loc[actions_selectionnees]["SA_ESG_RISK_SCR"]
            score_pondere = (scores * poids).sum()
            classe_risque = classer_risque(score_pondere)

            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(["Risque ESG"], [score_pondere], color="#F4A261", edgecolor='black')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{classe_risque} ({width:.2f})", va='center')
            ax.set_xlim(0, 40)
            ax.set_xlabel("Score de risque ESG")
            ax.set_title("Risque ESG Moyen pondéré")
            ax.invert_yaxis()
            st.pyplot(fig)

            df_affichage = pd.DataFrame({
                "Action": scores.index,
                "Score de Risque ESG": scores.values
            })
            st.dataframe(df_affichage)

        # ESG Global
        elif source_choisie == "ESG Global Score (0-10)":
            scores = dfESG.loc[actions_selectionnees]["ESG_SCORE"]
            score_pondere = (scores * poids).sum()

            fig, ax = plt.subplots(figsize=(8, 3.5))
            bars = ax.barh(["Score ESG Global"], [score_pondere], color="#6FB98F", edgecolor='black')
            for bar in bars:
                width = bar.get_width()
                ax.text(width + 0.2, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center')
            ax.set_xlim(0, 10)
            ax.set_xlabel("Score (0 à 10)")
            ax.set_title("Score ESG Global pondéré")
            ax.invert_yaxis()
            st.pyplot(fig)

            df_affichage = pd.DataFrame({
                "Action": scores.index,
                "Score ESG Global (0-10)": scores.values
            })
            st.dataframe(df_affichage)

    except Exception as e:
        st.warning(f"Erreur dans l'affichage des scores ESG : {e}")


    # ─── Synthèse par Action vs Benchmark ──────────────────────────────────────────
    st.markdown("### 🧠 Analyse Synthétique par Action")

    try:
        commentaires = []

        for action in df_perf.index:
            r = df_perf.loc[action, "Rendement Annualisé (%)"]
            v = df_perf.loc[action, "Volatilité Annualisée (%)"]
            sharpe = df_perf.loc[action, "Sharpe"]
            beta = df_perf.loc[action, "Beta"]

            benchmark_name = titre_indice

            phrase = f"🔹 **{action}** *(vs {benchmark_name})* : "

            # Rendement vs Benchmark
            if r > indice_annualized_return * 100:
                phrase += f"🚀 Surperforme le benchmark (**{r:.2f}%** vs **{indice_annualized_return*100:.2f}%**)"
            elif r < indice_annualized_return * 100:
                phrase += f"⚠️ Sous-performe le benchmark (**{r:.2f}%** vs **{indice_annualized_return*100:.2f}%**)"
            else:
                phrase += f"🔸 A un rendement équivalent au benchmark (**{r:.2f}%**)"

            # Volatilité vs Benchmark
            if v > indice_annualized_volatility * 100:
                phrase += f", 📉 avec une volatilité plus élevée (**{v:.2f}%** vs **{indice_annualized_volatility*100:.2f}%**)"
            elif v < indice_annualized_volatility * 100:
                phrase += f", 🛡️ avec une volatilité plus faible (**{v:.2f}%** vs **{indice_annualized_volatility*100:.2f}%**)"
            else:
                phrase += f", 🔸 avec une volatilité équivalente (**{v:.2f}%**)"

            # Sharpe + Bêta
            phrase += f", 📊 Sharpe : **{sharpe:.2f}**, ⚖️ Bêta : **{beta:.2f}**."

            commentaires.append(phrase)

        # Affichage avec saut de ligne entre les phrases
        st.info("\n\n".join(commentaires))

    except Exception as e:
        st.warning(f"Erreur dans la synthèse individuelle : {e}")



