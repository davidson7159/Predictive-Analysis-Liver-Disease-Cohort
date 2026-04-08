# 🫀 Entre vie, greffe et décès : Analyse prédictive appliquée à une cohorte hépatique

Modèle de classification multiclasse pour prédire le pronostic vital de patients atteints de maladies hépatiques, à partir de données cliniques et biologiques.

---

## 📋 Contexte

La prédiction du pronostic vital chez les patients atteints de maladies hépatiques représente un enjeu majeur en santé publique, tant pour optimiser la prise en charge médicale que pour anticiper les évolutions cliniques.

L'objectif est de prédire, pour chaque patient, la probabilité d'appartenir à l'une des trois catégories suivantes :

| Classe | Label | Description |
|--------|-------|-------------|
| `Status_C` | 0 | Patient vivant |
| `Status_D` | 1 | Patient décédé |
| `Status_CL` | 2 | Patient vivant ayant bénéficié d'une transplantation hépatique |

---

## 📁 Dataset

- **Source :** Kaggle (`classification-multi-classes`)
- **Table d'apprentissage :** 15 000 observations × 20 variables
- **Table de test :** 10 000 observations × 19 variables (sans `Status`)
- **Variables :** 7 catégorielles, 13 numériques (dont `id`)
- **Contrainte principale :** Plus de la moitié des variables présentent des taux de valeurs manquantes entre 42% et 55%

---

## 🔬 Méthodologie

### Preprocessing — Table d'apprentissage

**Variables numériques :**
- Test de normalité Kolmogorov-Smirnov → toutes les distributions sont non normales
- Test de Kruskal-Wallis → toutes les variables sont statistiquement informatives (p < 0.05)
- Imputation par la **médiane par classe** (`Status`) pour les variables avec > 30% de valeurs manquantes
- Imputation par la **médiane globale** pour les variables avec < 30% de valeurs manquantes (`Platelets`, `Prothrombin`)

**Variables catégorielles :**
- Correction des modalités aberrantes (`Ascites`, `Hepatomegaly`, `Spiders`)
- Test du Chi-deux → toutes les variables sont informatives (p < 0.05)
- Imputation par le **mode par classe** (`Status`)

**Preprocessing — Table de test :**
- Variables catégorielles : mode global calculé sur la table d'apprentissage
- Variables numériques : médiane globale de la table d'apprentissage

### Feature Engineering
- **One-Hot Encoding** (drop='first') pour les variables catégorielles
- **RobustScaler** pour les variables numériques (robuste aux valeurs extrêmes)
- La variable `Stage` traitée séparément (variable ordinale)
- Split : 70% train / 30% évaluation (stratifié)

---

## 🏗️ Modèles testés

### 1. Régression Logistique Multinomiale

- **GridSearchCV** — 10-fold StratifiedKFold, 18 combinaisons
- Meilleurs hyperparamètres : `C=100`, `solver='newton-cg'`
- Calibration des probabilités (méthode sigmoïde)
- **Log loss validation croisée : 0.3286**
- Log loss Kaggle public : 0.64

### 2. XGBoost

- **GridSearchCV** sur `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- Scoring : log loss (neg)
- **Log loss train : 0.3553**
- **Log loss Kaggle public : 0.3479** ✅ (meilleur modèle)

### 3. Réseau de neurones (MLP — PyTorch)

- Architecture : 2 couches cachées (128 → 64), activation ReLU
- 50 epochs, optimiseur Adam
- Log loss Kaggle public : **0.84** ❌ (surapprentissage / mauvaise généralisation)

---

## 📊 Résultats comparatifs

| Modèle | Log Loss (validation) | Log Loss (Kaggle public) |
|--------|----------------------|--------------------------|
| Régression Logistique | 0.3286 | 0.64 |
| **XGBoost** | 0.3553 | **0.3479** ✅ |
| MLP (PyTorch) | — | 0.84 |

> **Métrique d'évaluation :** Log loss multiclasse — mesure la qualité des probabilités prédites, pas seulement la classe.

---

## 🛠️ Tech Stack

| Catégorie | Librairies |
|-----------|-----------|
| Data manipulation | `pandas`, `numpy` |
| Statistiques | `scipy` (KS test, Kruskal-Wallis, Chi-deux) |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (LogisticRegression, GridSearchCV, RobustScaler, OHE) |
| Boosting | `xgboost` |
| Deep Learning | `PyTorch` |
| Environnement | Kaggle Notebooks |

---

## 🚀 Reproduire le projet

### 1. Cloner le repo
```bash
git clone https://github.com/your-username/predictive-analysis-liver.git
cd predictive-analysis-liver
```

### 2. Installer les dépendances
```bash
pip install pandas numpy scipy scikit-learn xgboost torch matplotlib seaborn
```

### 3. Télécharger les données
Depuis Kaggle : [classification-multi-classes](https://www.kaggle.com)

Placer les fichiers dans :
```
data/
├── train.csv
└── test.csv
```

### 4. Lancer le notebook
```bash
jupyter notebook Predictive_Analysis_Liver.ipynb
```

---

## 🔮 Pistes d'amélioration

- Feature engineering : créer des ratios biologiques (ex. Bilirubin/Albumin)
- Gérer le déséquilibre de la classe `Status_D` avec SMOTE ou class weights
- Tester d'autres modèles : LightGBM, CatBoost, Random Forest
- Optimiser le MLP : dropout, batch normalization, plus d'epochs
- Stacking / ensemble des trois modèles

---

## 📌 Structure du repo

```
predictive-analysis-liver/
│
├── Predictive_Analysis_Liver.ipynb   # Notebook principal
├── README.md                          # Ce fichier
├── data/
│   ├── train.csv
│   └── test.csv
├── submissions/
│   ├── submission_mlr.csv             # Prédictions Régression Logistique
│   ├── submission_xgb.csv             # Prédictions XGBoost
│   └── submission_RN.csv              # Prédictions MLP
```

---

## 📄 Licence

Ce projet est réalisé dans un cadre académique — Master 2 Data Science, Université Paris 1 Panthéon-Sorbonne.
