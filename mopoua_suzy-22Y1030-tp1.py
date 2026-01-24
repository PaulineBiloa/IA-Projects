# TP N°1 : Classification des fleurs iris
# Theme : Rose et Violet clair
# -*- coding: utf-8 -*-

import sys
import io

# Correction pour l'encodage Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Bibliotheques pour la manipulation des donnees
import pandas as pd
import numpy as np

# Bibliotheques pour la visualisation
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration pour de meilleurs graphiques avec theme rose-violet
plt.style.use('seaborn-v0_8-darkgrid')

# Palette de couleurs rose-violet personnalisee
PALETTE_ROSE_VIOLET = ['#FFB3D9', '#E6B3FF', '#D4A5D4', '#F8B8D8', '#E0B3E6']
COULEUR_PRINCIPALE = '#E6B3FF'
COULEUR_SECONDAIRE = '#FFB3D9'
COULEUR_TERTIAIRE = '#D4A5D4'

print("OK - Toutes les bibliotheques ont ete importees avec succes!")
print("=" * 60)

# ====================================================================
# ETAPE 2 : CHARGEMENT ET EXPLORATION DES DONNEES
# ====================================================================

# Charger le dataset Iris depuis sklearn
from sklearn.datasets import load_iris

# Charger les donnees
iris = load_iris()

# Creer un DataFrame pandas pour faciliter la manipulation
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Mapper les numeros aux noms d'especes
species_names = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
df['species'] = df['species'].map(species_names)

# Renommer les colonnes pour simplifier
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print("\n[APERCU DES DONNEES]")
print("=" * 60)
print("\n1. Les 5 premieres lignes du dataset :")
print(df.head())

print("\n2. Informations sur le dataset :")
print(df.info())

print("\n3. Statistiques descriptives :")
print(df.describe())

print("\n4. Verification des valeurs manquantes :")
print(df.isnull().sum())

print("\n5. Nombre d'echantillons par espece :")
print(df['species'].value_counts())

# ====================================================================
# VISUALISATION DE LA REPARTITION DES ESPECES
# ====================================================================

print("\n[CREATION DES GRAPHIQUES...]")

# Creer une figure avec plusieurs sous-graphiques
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#FFF5FA')

# Graphique 1 : Diagramme en barres (countplot)
sns.countplot(x='species', data=df, ax=axes[0], palette=PALETTE_ROSE_VIOLET)
axes[0].set_title('Distribution des especes d\'iris', fontsize=14, fontweight='bold', color='#8B4789')
axes[0].set_xlabel('Espece', fontsize=12, color='#8B4789')
axes[0].set_ylabel('Nombre d\'echantillons', fontsize=12, color='#8B4789')
axes[0].set_facecolor('#FFF5FA')

# Graphique 2 : Diagramme circulaire (pie chart)
species_counts = df['species'].value_counts()
axes[1].pie(species_counts, labels=species_counts.index, autopct='%1.1f%%', 
            startangle=90, colors=PALETTE_ROSE_VIOLET,
            textprops={'fontsize': 11, 'color': '#8B4789'})
axes[1].set_title('Repartition en pourcentage', fontsize=14, fontweight='bold', color='#8B4789')

plt.tight_layout()
plt.savefig('distribution_especes.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : distribution_especes.png")

# ====================================================================
# EXERCICE 1 : DIFFERENTES VISUALISATIONS DES ESPECES
# ====================================================================

print("\n[EXERCICE 1 : Visualisations variees]")
print("=" * 60)

# Afficher les effectifs
print("\n1. Effectifs de chaque modalite :")
effectifs = df['species'].value_counts()
print(effectifs)

# Creer une figure avec 4 types de visualisations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#FFF5FA')

# a) Histogramme
axes[0, 0].bar(effectifs.index, effectifs.values, color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
axes[0, 0].set_title('a) Histogramme', fontsize=12, fontweight='bold', color='#8B4789')
axes[0, 0].set_ylabel('Effectif', color='#8B4789')
axes[0, 0].set_xlabel('Espece', color='#8B4789')
axes[0, 0].set_facecolor('#FFF5FA')
for i, v in enumerate(effectifs.values):
    axes[0, 0].text(i, v + 1, str(v), ha='center', fontweight='bold', color='#8B4789')

# b) Diagramme en secteurs (Pie chart)
axes[0, 1].pie(effectifs.values, labels=effectifs.index, autopct='%1.1f%%', 
               startangle=90, colors=PALETTE_ROSE_VIOLET,
               explode=(0.05, 0.05, 0.05),
               textprops={'fontsize': 10, 'color': '#8B4789'})
axes[0, 1].set_title('b) Diagramme en secteurs', fontsize=12, fontweight='bold', color='#8B4789')

# c) Barres horizontales groupees
axes[1, 0].barh(effectifs.index, effectifs.values, color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
axes[1, 0].set_title('c) Barres horizontales', fontsize=12, fontweight='bold', color='#8B4789')
axes[1, 0].set_xlabel('Effectif', color='#8B4789')
axes[1, 0].set_ylabel('Espece', color='#8B4789')
axes[1, 0].set_facecolor('#FFF5FA')
for i, v in enumerate(effectifs.values):
    axes[1, 0].text(v + 1, i, str(v), va='center', fontweight='bold', color='#8B4789')

# d) Graphique en cascade (waterfall-style)
cumsum = np.cumsum([0] + list(effectifs.values))
axes[1, 1].bar(range(len(effectifs)), effectifs.values, 
               bottom=cumsum[:-1], color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
axes[1, 1].set_title('d) Graphique en cascade', fontsize=12, fontweight='bold', color='#8B4789')
axes[1, 1].set_xticks(range(len(effectifs)))
axes[1, 1].set_xticklabels(effectifs.index)
axes[1, 1].set_ylabel('Effectif cumule', color='#8B4789')
axes[1, 1].set_facecolor('#FFF5FA')

plt.tight_layout()
plt.savefig('exercice1_visualisations.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : exercice1_visualisations.png")
print("\nMeilleure representation : Le diagramme en barres (histogramme)")
print("Raison : Il permet de comparer facilement les effectifs egaux entre especes.")

# ====================================================================
# EXERCICE 2 : ANALYSE DES VARIABLES QUANTITATIVES
# ====================================================================

print("\n" + "=" * 60)
print("[EXERCICE 2 : Analyse des variables quantitatives]")
print("=" * 60)

# 1. Resume statistique de la longueur du petale
print("\n1. Resume statistique - Longueur du petale (petal_length):")
print("-" * 60)
print(df['petal_length'].describe())
print(f"\nMode : {df['petal_length'].mode()[0]}")
print(f"Variance : {df['petal_length'].var():.4f}")
print(f"Ecart-type : {df['petal_length'].std():.4f}")

# 2. Visualisation des 4 variables quantitatives
print("\n2. Creation des histogrammes pour toutes les variables...")

# Creer une figure avec 4 histogrammes
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Distribution des variables quantitatives', fontsize=16, fontweight='bold', color='#8B4789')
fig.patch.set_facecolor('#FFF5FA')

# Liste des variables quantitatives
variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
titres = ['Longueur du Sepale (cm)', 'Largeur du Sepale (cm)', 
          'Longueur du Petale (cm)', 'Largeur du Petale (cm)']

# Creer les histogrammes
for idx, (var, titre) in enumerate(zip(variables, titres)):
    row = idx // 2
    col = idx % 2
    
    # Histogramme avec courbe de densite
    axes[row, col].hist(df[var], bins=20, color=COULEUR_PRINCIPALE, edgecolor='#8B4789', alpha=0.7)
    axes[row, col].set_title(titre, fontsize=12, fontweight='bold', color='#8B4789')
    axes[row, col].set_xlabel('Valeur (cm)', fontsize=10, color='#8B4789')
    axes[row, col].set_ylabel('Frequence', fontsize=10, color='#8B4789')
    axes[row, col].grid(axis='y', alpha=0.3, color='#D4A5D4')
    axes[row, col].set_facecolor('#FFF5FA')
    
    # Ajouter une ligne verticale pour la moyenne
    mean_val = df[var].mean()
    axes[row, col].axvline(mean_val, color='#C71585', linestyle='--', linewidth=2, label=f'Moyenne: {mean_val:.2f}')
    axes[row, col].legend()

plt.tight_layout()
plt.savefig('exercice2_histogrammes.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice2_histogrammes.png")

# 3. Tableau recapitulatif des statistiques
print("\n3. Tableau recapitulatif des statistiques descriptives :")
print("-" * 60)
stats_summary = df[variables].describe().T
stats_summary['variance'] = df[variables].var()
print(stats_summary)

# ====================================================================
# EXERCICE 3 : ETUDE BIVARIEE - NUAGE DE POINTS
# ====================================================================

print("\n" + "=" * 60)
print("[EXERCICE 3 : Etude bivariee - Nuage de points]")
print("=" * 60)

# 1. Nuage de points : Longueur vs Largeur du petale
print("\n1. Nuage de points : Longueur vs Largeur du petale")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FFF5FA')

# Graphique 1 : Sans distinction d'espece
axes[0].scatter(df['petal_length'], df['petal_width'], alpha=0.7, s=60, color=COULEUR_PRINCIPALE, edgecolors='#8B4789')
axes[0].set_xlabel('Longueur du petale (cm)', fontsize=12, color='#8B4789')
axes[0].set_ylabel('Largeur du petale (cm)', fontsize=12, color='#8B4789')
axes[0].set_title('Relation Longueur-Largeur du petale (toutes especes)', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].grid(True, alpha=0.3, color='#D4A5D4')
axes[0].set_facecolor('#FFF5FA')

# Graphique 2 : Avec distinction par espece
colors = {'setosa': '#FFB3D9', 'versicolor': '#E6B3FF', 'virginica': '#D4A5D4'}
for species in df['species'].unique():
    subset = df[df['species'] == species]
    axes[1].scatter(subset['petal_length'], subset['petal_width'], 
                   label=species, alpha=0.7, s=60, color=colors[species], edgecolors='#8B4789')

axes[1].set_xlabel('Longueur du petale (cm)', fontsize=12, color='#8B4789')
axes[1].set_ylabel('Largeur du petale (cm)', fontsize=12, color='#8B4789')
axes[1].set_title('Relation Longueur-Largeur du petale (par espece)', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].legend()
axes[1].grid(True, alpha=0.3, color='#D4A5D4')
axes[1].set_facecolor('#FFF5FA')

plt.tight_layout()
plt.savefig('exercice3_nuage_points_petale.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice3_nuage_points_petale.png")
print("\nCommentaire : On observe une correlation positive forte entre la longueur")
print("et la largeur du petale. Les trois especes forment des groupes distincts.")

# 2. Autre croisement : Longueur du sepale vs Longueur du petale
print("\n2. Autre croisement : Longueur du sepale vs Longueur du petale")

fig = plt.figure(figsize=(10, 6))
fig.patch.set_facecolor('#FFF5FA')
ax = fig.add_subplot(111)
ax.set_facecolor('#FFF5FA')

for species in df['species'].unique():
    subset = df[df['species'] == species]
    ax.scatter(subset['sepal_length'], subset['petal_length'], 
               label=species, alpha=0.7, s=60, color=colors[species], edgecolors='#8B4789')

ax.set_xlabel('Longueur du sepale (cm)', fontsize=12, color='#8B4789')
ax.set_ylabel('Longueur du petale (cm)', fontsize=12, color='#8B4789')
ax.set_title('Relation Longueur du sepale vs Longueur du petale', fontsize=14, fontweight='bold', color='#8B4789')
ax.legend()
ax.grid(True, alpha=0.3, color='#D4A5D4')
plt.tight_layout()
plt.savefig('exercice3_sepale_petale.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice3_sepale_petale.png")
print("\nCommentaire : Setosa a des petales courts quelle que soit la longueur du sepale.")
print("Versicolor et Virginica montrent une correlation plus marquee.")

# ====================================================================
# EXERCICE 4 : BOITES A MOUSTACHES (BOXPLOT)
# ====================================================================

print("\n" + "=" * 60)
print("[EXERCICE 4 : Boites a moustaches - Variables quantitatives vs Espece]")
print("=" * 60)

# Creer une figure avec 4 boxplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Boxplots : Variables quantitatives par espece', fontsize=16, fontweight='bold', color='#8B4789')
fig.patch.set_facecolor('#FFF5FA')

variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
titres = ['Longueur du Sepale', 'Largeur du Sepale', 'Longueur du Petale', 'Largeur du Petale']

for idx, (var, titre) in enumerate(zip(variables, titres)):
    row = idx // 2
    col = idx % 2
    
    # Creer le boxplot
    sns.boxplot(x='species', y=var, data=df, ax=axes[row, col], palette=PALETTE_ROSE_VIOLET)
    axes[row, col].set_title(f'{titre} par espece', fontsize=12, fontweight='bold', color='#8B4789')
    axes[row, col].set_xlabel('Espece', fontsize=10, color='#8B4789')
    axes[row, col].set_ylabel(f'{titre} (cm)', fontsize=10, color='#8B4789')
    axes[row, col].grid(axis='y', alpha=0.3, color='#D4A5D4')
    axes[row, col].set_facecolor('#FFF5FA')

plt.tight_layout()
plt.savefig('exercice4_boxplots.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice4_boxplots.png")

# Commentaires detailles
print("\n--- ANALYSE DES BOXPLOTS ---")
print("\n1. Longueur du petale par espece :")
print("   - Setosa : petales tres courts (1-2 cm)")
print("   - Versicolor : petales moyens (3-5 cm)")
print("   - Virginica : petales longs (4.5-7 cm)")
print("   => Cette variable discrimine tres bien les 3 especes")

print("\n2. Largeur du sepale par espece :")
print("   - Setosa : plus large")
print("   - Versicolor et Virginica : plus etroites et similaires")
print("   => Moins discriminante pour versicolor et virginica")

# ====================================================================
# EXERCICE 5 : CORRELATIONS ET VISUALISATIONS AVANCEES
# ====================================================================

print("\n" + "=" * 60)
print("[EXERCICE 5 : Correlations et visualisations avancees]")
print("=" * 60)

# 1. Matrice de correlation
print("\n1. Matrice de correlation entre variables quantitatives :")
correlation_matrix = df[variables].corr()
print(correlation_matrix)

# Visualisation de la matrice de correlation
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#FFF5FA')
ax = fig.add_subplot(111)
sns.heatmap(correlation_matrix, annot=True, cmap='PuRd', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Matrice de correlation des variables', fontsize=14, fontweight='bold', color='#8B4789')
plt.tight_layout()
plt.savefig('exercice5_correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice5_correlation_matrix.png")

# 2. Pairplot - Toutes les combinaisons de variables
print("\n2. Pairplot - Vue d'ensemble de toutes les relations...")
pairplot_fig = sns.pairplot(df, hue='species', palette=colors, diag_kind='hist', 
                             plot_kws={'alpha': 0.7, 'edgecolor': '#8B4789', 's': 40}, 
                             height=2.5)
pairplot_fig.fig.suptitle('Pairplot : Toutes les relations entre variables', 
                          y=1.02, fontsize=16, fontweight='bold', color='#8B4789')
pairplot_fig.fig.patch.set_facecolor('#FFF5FA')
plt.savefig('exercice5_pairplot.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice5_pairplot.png")

# 3. Violinplot - Distribution detaillee
print("\n3. Violinplot pour une visualisation detaillee des distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Violinplots : Distribution detaillee par espece', fontsize=16, fontweight='bold', color='#8B4789')
fig.patch.set_facecolor('#FFF5FA')

for idx, (var, titre) in enumerate(zip(variables, titres)):
    row = idx // 2
    col = idx % 2
    
    sns.violinplot(x='species', y=var, data=df, ax=axes[row, col], palette=PALETTE_ROSE_VIOLET)
    axes[row, col].set_title(f'{titre} par espece', fontsize=12, fontweight='bold', color='#8B4789')
    axes[row, col].set_xlabel('Espece', fontsize=10, color='#8B4789')
    axes[row, col].set_ylabel(f'{titre} (cm)', fontsize=10, color='#8B4789')
    axes[row, col].set_facecolor('#FFF5FA')

plt.tight_layout()
plt.savefig('exercice5_violinplots.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : exercice5_violinplots.png")

# Resume des correlations
print("\n--- ANALYSE DES CORRELATIONS ---")
print("\nCorrelations fortes (> 0.8) :")
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            print(f"  - {correlation_matrix.columns[i]} <-> {correlation_matrix.columns[j]}: {correlation_matrix.iloc[i, j]:.3f}")

print("\nConclusion : Les dimensions des petales sont fortement correlees entre elles,")
print("ainsi qu'avec la longueur du sepale. Ces variables sont donc tres informatives")
print("pour la classification des especes d'iris.")

print("\n" + "=" * 60)
print("[FIN DE LA PARTIE VISUALISATION]")
print("Tous les graphiques ont ete sauvegardes avec le theme rose-violet !")
print("=" * 60)

# ====================================================================
# ETAPE 3 : PREPARATION DES DONNEES POUR LE MODELE
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 3 : Preparation des donnees pour le Machine Learning]")
print("=" * 60)

# Importer la fonction de separation des donnees
from sklearn.model_selection import train_test_split

# Etape 3.1 : Separer les caracteristiques (X) et la cible (y)
print("\n--- Etape 3.1 : Separation caracteristiques / cible ---")

# X contient toutes les colonnes SAUF 'species'
X = df.drop('species', axis=1)
print("\nCaracteristiques (X) :")
print(f"  - Forme : {X.shape} (150 lignes x 4 colonnes)")
print(f"  - Colonnes : {list(X.columns)}")
print("\nApercu de X :")
print(X.head())

# y contient UNIQUEMENT la colonne 'species'
y = df['species']
print("\n\nCible (y) :")
print(f"  - Forme : {y.shape} (150 lignes)")
print(f"  - Valeurs uniques : {y.unique()}")
print("\nApercu de y :")
print(y.head())

# Visualisation de la separation
print("\n\nVisualisation de la structure des donnees...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Structure des donnees : X (caracteristiques) et y (cible)', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Heatmap des caracteristiques (5 premieres lignes)
sns.heatmap(X.head(10), annot=True, fmt='.1f', cmap='PuRd', 
            cbar_kws={'label': 'Valeur (cm)'}, ax=axes[0])
axes[0].set_title('X : Caracteristiques (10 premieres lignes)', 
                  fontsize=12, fontweight='bold', color='#8B4789')
axes[0].set_ylabel('Numero de ligne', color='#8B4789')

# Graphique 2 : Distribution de la cible
y_counts = y.value_counts()
axes[1].bar(y_counts.index, y_counts.values, color=PALETTE_ROSE_VIOLET, 
           edgecolor='#8B4789', linewidth=2)
axes[1].set_title('y : Distribution de la cible', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xlabel('Espece', color='#8B4789')
axes[1].set_ylabel('Nombre d\'echantillons', color='#8B4789')
axes[1].set_facecolor('#FFF5FA')
for i, v in enumerate(y_counts.values):
    axes[1].text(i, v + 1, str(v), ha='center', fontweight='bold', color='#8B4789')

plt.tight_layout()
plt.savefig('etape3_separation_donnees.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("OK - Graphique sauvegarde : etape3_separation_donnees.png")

# ====================================================================
# Etape 3.2 : Division en ensemble d'entrainement et de test
# ====================================================================

print("\n--- Etape 3.2 : Division Train/Test (80% / 20%) ---")

# Diviser les donnees : 80% pour l'entrainement, 20% pour le test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% pour le test
    random_state=42,    # Pour avoir toujours les memes resultats
    stratify=y          # Garder la meme proportion d'especes dans train et test
)

print("\nResultat de la division :")
print(f"  - Ensemble d'ENTRAINEMENT : {X_train.shape[0]} echantillons (80%)")
print(f"  - Ensemble de TEST        : {X_test.shape[0]} echantillons (20%)")
print(f"  - Total                    : {X_train.shape[0] + X_test.shape[0]} echantillons")

# Verifier la repartition des especes
print("\nRepartition des especes dans l'ensemble d'ENTRAINEMENT :")
print(y_train.value_counts().sort_index())

print("\nRepartition des especes dans l'ensemble de TEST :")
print(y_test.value_counts().sort_index())

# Visualisation de la division
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Division des donnees : Entrainement (80%) vs Test (20%)', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Taille des ensembles
sizes = [X_train.shape[0], X_test.shape[0]]
labels = [f'Entrainement\n({X_train.shape[0]} echantillons)', 
          f'Test\n({X_test.shape[0]} echantillons)']
colors_pie = ['#E6B3FF', '#FFB3D9']
axes[0].pie(sizes, labels=labels, autopct='%1.0f%%', startangle=90, 
           colors=colors_pie, textprops={'fontsize': 11, 'color': '#8B4789'})
axes[0].set_title('Proportion Train/Test', fontsize=12, fontweight='bold', color='#8B4789')

# Graphique 2 : Repartition des especes
x_pos = np.arange(len(y_train.value_counts()))
width = 0.35
train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()

bars1 = axes[1].bar(x_pos - width/2, train_counts, width, 
                   label='Entrainement', color='#E6B3FF', edgecolor='#8B4789', linewidth=1.5)
bars2 = axes[1].bar(x_pos + width/2, test_counts, width, 
                   label='Test', color='#FFB3D9', edgecolor='#8B4789', linewidth=1.5)

axes[1].set_xlabel('Espece', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Nombre d\'echantillons', fontsize=11, color='#8B4789')
axes[1].set_title('Repartition par espece', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(train_counts.index)
axes[1].legend()
axes[1].set_facecolor('#FFF5FA')
axes[1].grid(axis='y', alpha=0.3, color='#D4A5D4')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontweight='bold', color='#8B4789', fontsize=9)

plt.tight_layout()
plt.savefig('etape3_division_train_test.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape3_division_train_test.png")

print("\n💡 Point important :")
print("   Le parametre 'stratify=y' garantit que chaque espece est")
print("   representee proportionnellement dans train et test.")

# ====================================================================
# Etape 3.3 : Normalisation des caracteristiques
# ====================================================================

print("\n--- Etape 3.3 : Normalisation des donnees ---")

# Importer le normaliseur
from sklearn.preprocessing import StandardScaler

# Regardons les donnees AVANT normalisation
print("\nDonnees AVANT normalisation (5 premieres lignes de X_train) :")
print(X_train.head())
print("\nStatistiques AVANT normalisation :")
stats_avant = X_train.describe().T[['mean', 'std']].round(2)
print(stats_avant)

# Creer le normaliseur
scaler = StandardScaler()

# Entrainer le normaliseur sur les donnees d'entrainement et transformer
X_train_scaled = scaler.fit_transform(X_train)

# Transformer les donnees de test (SANS fit !)
X_test_scaled = scaler.transform(X_test)

# Convertir en DataFrame pour mieux visualiser
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

print("\n\nDonnees APRES normalisation (5 premieres lignes) :")
print(X_train_scaled_df.head())
print("\nStatistiques APRES normalisation :")
stats_apres = X_train_scaled_df.describe().T[['mean', 'std']].round(2)
print(stats_apres)

# Visualisation : Avant vs Apres normalisation
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Effet de la normalisation sur les donnees', 
             fontsize=16, fontweight='bold', color='#8B4789')

variables = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
titres = ['Longueur Sepale', 'Largeur Sepale', 'Longueur Petale', 'Largeur Petale']

for idx, (var, titre) in enumerate(zip(variables, titres)):
    row = idx // 2
    col = idx % 2
    
    # Donnees avant et apres
    avant = X_train[var]
    apres = X_train_scaled_df[var]
    
    # Position des boites
    positions = [1, 2]
    data_to_plot = [avant, apres]
    
    bp = axes[row, col].boxplot(data_to_plot, positions=positions, widths=0.5,
                                patch_artist=True, 
                                boxprops=dict(facecolor='#E6B3FF', edgecolor='#8B4789', linewidth=1.5),
                                medianprops=dict(color='#C71585', linewidth=2),
                                whiskerprops=dict(color='#8B4789', linewidth=1.5),
                                capprops=dict(color='#8B4789', linewidth=1.5))
    
    axes[row, col].set_title(titre, fontsize=12, fontweight='bold', color='#8B4789')
    axes[row, col].set_xticklabels(['Avant', 'Apres'], color='#8B4789')
    axes[row, col].set_ylabel('Valeur', fontsize=10, color='#8B4789')
    axes[row, col].grid(axis='y', alpha=0.3, color='#D4A5D4')
    axes[row, col].set_facecolor('#FFF5FA')
    
    # Ajouter les moyennes
    axes[row, col].axhline(y=0, color='#C71585', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
plt.savefig('etape3_normalisation.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape3_normalisation.png")

print("\n💡 Ce qu'il faut retenir :")
print("   - Avant : les valeurs sont differentes (0-8 cm)")
print("   - Apres : moyenne = 0, ecart-type = 1 pour toutes les variables")
print("   - Cela permet au modele de traiter toutes les variables equitablement")

# ====================================================================
# ETAPE 4 : CREATION ET ENTRAINEMENT DU MODELE KNN
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 4 : Creation et entrainement du modele KNN]")
print("=" * 60)

# Importer le modele KNN
from sklearn.neighbors import KNeighborsClassifier

print("\n--- Qu'est-ce que le modele KNN ? ---")
print("KNN = K-Nearest Neighbors (K plus proches voisins)")
print("Principe : Pour classifier une nouvelle fleur, on regarde les K fleurs")
print("           les plus proches et on vote pour l'espece majoritaire.")
print("\nExemple : Si K=3 et que les 3 voisins les plus proches sont des setosa,")
print("          alors la nouvelle fleur sera classee comme setosa.")

# Etape 4.1 : Creer le modele avec K=3
print("\n--- Etape 4.1 : Creation du modele ---")
print("Parametres choisis : n_neighbors = 3 (on regarde les 3 voisins les plus proches)")

knn = KNeighborsClassifier(n_neighbors=3)
print("\nModele KNN cree avec succes !")
print(f"Type du modele : {type(knn)}")

# Etape 4.2 : Entrainer le modele
print("\n--- Etape 4.2 : Entrainement du modele ---")
print("Le modele va apprendre sur les 120 echantillons d'entrainement...")

# Entrainer le modele
knn.fit(X_train_scaled, y_train)

print("\nEntrainement termine avec succes !")
print(f"Le modele a appris sur {X_train_scaled.shape[0]} echantillons")
print(f"Nombre de caracteristiques utilisees : {X_train_scaled.shape[1]}")
print(f"Nombre de classes (especes) : {len(knn.classes_)}")
print(f"Classes reconnues : {list(knn.classes_)}")

# Etape 4.3 : Faire des predictions
print("\n--- Etape 4.3 : Predictions sur l'ensemble de test ---")
print("Le modele va maintenant predire l'espece des 30 fleurs de test...")

# Predire les especes
y_pred = knn.predict(X_test_scaled)

print("\nPredictions terminees !")
print(f"Nombre de predictions : {len(y_pred)}")

# Comparer les predictions avec les vraies valeurs
print("\nComparaison des 10 premieres predictions :")
print("-" * 60)
comparison_df = pd.DataFrame({
    'Vraie espece': y_test.values[:10],
    'Prediction': y_pred[:10],
    'Correct?': ['✓' if y_test.values[i] == y_pred[i] else '✗' for i in range(10)]
})
print(comparison_df.to_string(index=False))

# Visualisation des predictions
print("\nVisualisation des predictions...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Predictions du modele KNN sur l\'ensemble de test', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Comparaison des vraies valeurs vs predictions
species_order = ['setosa', 'versicolor', 'virginica']
y_test_counts = pd.Series(y_test).value_counts()[species_order]
y_pred_counts = pd.Series(y_pred).value_counts().reindex(species_order, fill_value=0)

x_pos = np.arange(len(species_order))
width = 0.35

bars1 = axes[0].bar(x_pos - width/2, y_test_counts, width, 
                   label='Vraies valeurs', color='#E6B3FF', 
                   edgecolor='#8B4789', linewidth=1.5)
bars2 = axes[0].bar(x_pos + width/2, y_pred_counts, width, 
                   label='Predictions', color='#FFB3D9', 
                   edgecolor='#8B4789', linewidth=1.5)

axes[0].set_xlabel('Espece', fontsize=11, color='#8B4789')
axes[0].set_ylabel('Nombre', fontsize=11, color='#8B4789')
axes[0].set_title('Vraies valeurs vs Predictions', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(species_order)
axes[0].legend()
axes[0].set_facecolor('#FFF5FA')
axes[0].grid(axis='y', alpha=0.3, color='#D4A5D4')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}', ha='center', va='bottom', 
                    fontweight='bold', color='#8B4789', fontsize=9)

# Graphique 2 : Predictions correctes vs incorrectes
correct = sum(y_test.values == y_pred)
incorrect = len(y_pred) - correct
sizes = [correct, incorrect]
labels = [f'Correctes\n({correct})', f'Incorrectes\n({incorrect})']
colors_pie = ['#E6B3FF', '#FFB3D9']

axes[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90,
           colors=colors_pie, textprops={'fontsize': 11, 'color': '#8B4789'})
axes[1].set_title('Taux de reussite', fontsize=12, fontweight='bold', color='#8B4789')

plt.tight_layout()
plt.savefig('etape4_predictions.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape4_predictions.png")
print(f"\nPremier apercu des performances : {correct}/{len(y_pred)} predictions correctes ({100*correct/len(y_pred):.1f}%)")

# ====================================================================
# ETAPE 5 : EVALUATION DETAILLEE DU MODELE
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 5 : Evaluation detaillee du modele]")
print("=" * 60)

# Importer les metriques d'evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Etape 5.1 : Exactitude (Accuracy)
print("\n--- Etape 5.1 : Exactitude globale ---")

accuracy = accuracy_score(y_test, y_pred)
print(f"\nExactitude (Accuracy) : {accuracy * 100:.2f}%")
print(f"Cela signifie que le modele a correctement predit {accuracy * 100:.2f}% des fleurs.")

# Etape 5.2 : Matrice de confusion
print("\n--- Etape 5.2 : Matrice de confusion ---")
print("\nLa matrice de confusion montre :")
print("  - En LIGNES : les vraies especes")
print("  - En COLONNES : les predictions du modele")
print("  - DIAGONALE : predictions correctes")
print("  - HORS DIAGONALE : erreurs")

conf_matrix = confusion_matrix(y_test, y_pred, labels=species_order)
print("\nMatrice de confusion :")
print(conf_matrix)

# Visualisation de la matrice de confusion
fig = plt.figure(figsize=(10, 8))
fig.patch.set_facecolor('#FFF5FA')
ax = fig.add_subplot(111)

# Creer la heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='PuRd', 
            xticklabels=species_order, yticklabels=species_order,
            cbar_kws={'label': 'Nombre de predictions'},
            linewidths=2, linecolor='#8B4789', ax=ax)

ax.set_title('Matrice de confusion du modele KNN', 
            fontsize=14, fontweight='bold', color='#8B4789', pad=20)
ax.set_xlabel('Predictions', fontsize=12, color='#8B4789', fontweight='bold')
ax.set_ylabel('Vraies especes', fontsize=12, color='#8B4789', fontweight='bold')

# Colorer les axes
ax.tick_params(colors='#8B4789')

plt.tight_layout()
plt.savefig('etape5_matrice_confusion.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape5_matrice_confusion.png")

# Analyser la matrice
print("\n--- Analyse de la matrice de confusion ---")
for i, species in enumerate(species_order):
    correct_predictions = conf_matrix[i, i]
    total_species = conf_matrix[i, :].sum()
    print(f"  {species.capitalize()} : {correct_predictions}/{total_species} correctes")
    
    # Identifier les erreurs
    errors = []
    for j, other_species in enumerate(species_order):
        if i != j and conf_matrix[i, j] > 0:
            errors.append(f"{conf_matrix[i, j]} confondues avec {other_species}")
    if errors:
        print(f"    Erreurs : {', '.join(errors)}")
    else:
        print(f"    Aucune erreur !")

# Etape 5.3 : Rapport de classification
print("\n--- Etape 5.3 : Rapport de classification detaille ---")
print("\nLe rapport contient pour chaque espece :")
print("  - Precision : % de predictions correctes parmi celles faites")
print("  - Recall (Rappel) : % de vraies especes correctement trouvees")
print("  - F1-score : moyenne harmonique de precision et recall")
print("  - Support : nombre d'echantillons reels de cette espece")

print("\nRapport de classification :")
print(classification_report(y_test, y_pred, target_names=species_order))

# Visualisation du rapport de classification
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=species_order)

# Creer un graphique comparatif
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Metriques de performance par espece', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Precision, Recall, F1-score
x_pos = np.arange(len(species_order))
width = 0.25

bars1 = axes[0].bar(x_pos - width, precision, width, 
                   label='Precision', color='#FFB3D9', 
                   edgecolor='#8B4789', linewidth=1.5)
bars2 = axes[0].bar(x_pos, recall, width, 
                   label='Recall', color='#E6B3FF', 
                   edgecolor='#8B4789', linewidth=1.5)
bars3 = axes[0].bar(x_pos + width, f1, width, 
                   label='F1-score', color='#D4A5D4', 
                   edgecolor='#8B4789', linewidth=1.5)

axes[0].set_xlabel('Espece', fontsize=11, color='#8B4789')
axes[0].set_ylabel('Score', fontsize=11, color='#8B4789')
axes[0].set_title('Precision, Recall et F1-score', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(species_order)
axes[0].legend()
axes[0].set_ylim([0, 1.1])
axes[0].set_facecolor('#FFF5FA')
axes[0].grid(axis='y', alpha=0.3, color='#D4A5D4')

# Ajouter les valeurs sur les barres
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', 
                    fontweight='bold', color='#8B4789', fontsize=8)

# Graphique 2 : Support (nombre d'echantillons)
bars = axes[1].bar(species_order, support, color=PALETTE_ROSE_VIOLET, 
                   edgecolor='#8B4789', linewidth=1.5)
axes[1].set_xlabel('Espece', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Nombre d\'echantillons', fontsize=11, color='#8B4789')
axes[1].set_title('Support (echantillons de test)', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_facecolor('#FFF5FA')
axes[1].grid(axis='y', alpha=0.3, color='#D4A5D4')

for bar in bars:
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', 
                fontweight='bold', color='#8B4789', fontsize=10)

plt.tight_layout()
plt.savefig('etape5_metriques_performance.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape5_metriques_performance.png")

# Resume final
print("\n" + "=" * 60)
print("RESUME DE L'EVALUATION")
print("=" * 60)
print(f"\n✓ Exactitude globale : {accuracy * 100:.2f}%")
print(f"✓ Nombre total de predictions : {len(y_pred)}")
print(f"✓ Predictions correctes : {sum(y_test.values == y_pred)}")
print(f"✓ Predictions incorrectes : {sum(y_test.values != y_pred)}")
print("\nConclusion : Le modele KNN performe tres bien sur ce dataset !")
print("Les erreurs sont minimes et principalement entre versicolor et virginica.")

# ====================================================================
# ETAPE 6 : OPTIMISATION DU MODELE - RECHERCHE DU MEILLEUR K
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 6 : Optimisation du modele - Recherche du meilleur K]")
print("=" * 60)

print("\n--- Pourquoi optimiser K ? ---")
print("K = nombre de voisins a considerer pour la prediction")
print("  - K trop petit (ex: K=1) : sensible au bruit, risque de surajustement")
print("  - K trop grand (ex: K=50) : perd en precision, sous-ajustement")
print("  - Il faut trouver le K optimal !")

# Etape 6.1 : Tester differentes valeurs de K
print("\n--- Etape 6.1 : Test de differentes valeurs de K ---")
print("Nous allons tester K de 1 a 30...")

# Liste des valeurs de K a tester
k_values = range(1, 31)
train_scores = []
test_scores = []

# Tester chaque valeur de K
for k in k_values:
    # Creer et entrainer le modele
    knn_temp = KNeighborsClassifier(n_neighbors=k)
    knn_temp.fit(X_train_scaled, y_train)
    
    # Calculer les scores
    train_score = knn_temp.score(X_train_scaled, y_train)
    test_score = knn_temp.score(X_test_scaled, y_test)
    
    train_scores.append(train_score)
    test_scores.append(test_score)

# Trouver le meilleur K
best_k = list(k_values)[test_scores.index(max(test_scores))]
best_score = max(test_scores)

print(f"\nTest termine !")
print(f"Meilleur K trouve : {best_k}")
print(f"Meilleure exactitude sur test : {best_score * 100:.2f}%")

# Visualisation de l'evolution des scores
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Optimisation du parametre K', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Evolution des scores
axes[0].plot(k_values, train_scores, marker='o', linewidth=2, 
            color='#E6B3FF', label='Score entrainement', markersize=6)
axes[0].plot(k_values, test_scores, marker='s', linewidth=2, 
            color='#FFB3D9', label='Score test', markersize=6)
axes[0].axvline(x=best_k, color='#C71585', linestyle='--', linewidth=2, 
               label=f'Meilleur K = {best_k}')
axes[0].set_xlabel('Nombre de voisins (K)', fontsize=11, color='#8B4789')
axes[0].set_ylabel('Exactitude', fontsize=11, color='#8B4789')
axes[0].set_title('Evolution de l\'exactitude selon K', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].legend()
axes[0].grid(True, alpha=0.3, color='#D4A5D4')
axes[0].set_facecolor('#FFF5FA')
axes[0].set_ylim([0.85, 1.02])

# Graphique 2 : Top 5 des meilleurs K
# Trouver les 5 meilleurs K
k_scores = list(zip(k_values, test_scores))
k_scores_sorted = sorted(k_scores, key=lambda x: x[1], reverse=True)[:5]
top_k = [x[0] for x in k_scores_sorted]
top_scores = [x[1] for x in k_scores_sorted]

bars = axes[1].bar(range(len(top_k)), top_scores, 
                  color=PALETTE_ROSE_VIOLET[:5], 
                  edgecolor='#8B4789', linewidth=1.5)
axes[1].set_xlabel('Rang', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Exactitude', fontsize=11, color='#8B4789')
axes[1].set_title('Top 5 des meilleurs K', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xticks(range(len(top_k)))
axes[1].set_xticklabels([f'{i+1}' for i in range(len(top_k))])
axes[1].set_facecolor('#FFF5FA')
axes[1].grid(axis='y', alpha=0.3, color='#D4A5D4')
axes[1].set_ylim([0.85, 1.02])

# Ajouter les valeurs K et scores sur les barres
for i, (bar, k, score) in enumerate(zip(bars, top_k, top_scores)):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height,
                f'K={k}\n{score*100:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color='#8B4789', fontsize=9)

plt.tight_layout()
plt.savefig('etape6_optimisation_k.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape6_optimisation_k.png")

# Afficher le tableau des 5 meilleurs K
print("\n--- Top 5 des meilleurs K ---")
print("-" * 40)
print(f"{'Rang':<6} {'K':<6} {'Exactitude':<12}")
print("-" * 40)
for i, (k, score) in enumerate(k_scores_sorted[:5], 1):
    print(f"{i:<6} {k:<6} {score*100:.2f}%")
print("-" * 40)

# Etape 6.2 : Entrainer le modele final avec le meilleur K
print("\n--- Etape 6.2 : Entrainement du modele final optimise ---")
print(f"Nous allons maintenant creer un nouveau modele avec K = {best_k}")

# Creer le modele optimise
knn_optimized = KNeighborsClassifier(n_neighbors=best_k)
knn_optimized.fit(X_train_scaled, y_train)

# Predictions avec le modele optimise
y_pred_optimized = knn_optimized.predict(X_test_scaled)

# Evaluer le modele optimise
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
conf_matrix_optimized = confusion_matrix(y_test, y_pred_optimized, labels=species_order)

print(f"\nModele optimise entraine avec succes !")
print(f"Exactitude du modele optimise : {accuracy_optimized * 100:.2f}%")

# Comparaison avant/apres optimisation
print("\n--- Comparaison des modeles ---")
print(f"Modele initial (K=3)     : {accuracy * 100:.2f}%")
print(f"Modele optimise (K={best_k})  : {accuracy_optimized * 100:.2f}%")
if accuracy_optimized > accuracy:
    improvement = (accuracy_optimized - accuracy) * 100
    print(f"Amelioration             : +{improvement:.2f}%")
elif accuracy_optimized < accuracy:
    print(f"Performance similaire")
else:
    print(f"Performance identique")

# Visualisation de la comparaison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Comparaison : Modele initial vs Modele optimise', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Comparaison des exactitudes
models = ['Initial\n(K=3)', f'Optimise\n(K={best_k})']
accuracies = [accuracy, accuracy_optimized]
bars = axes[0].bar(models, accuracies, color=['#E6B3FF', '#FFB3D9'], 
                  edgecolor='#8B4789', linewidth=2)
axes[0].set_ylabel('Exactitude', fontsize=11, color='#8B4789')
axes[0].set_title('Exactitude des modeles', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].set_ylim([0.9, 1.02])
axes[0].set_facecolor('#FFF5FA')
axes[0].grid(axis='y', alpha=0.3, color='#D4A5D4')

for bar in bars:
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color='#8B4789', fontsize=11)

# Graphique 2 : Matrice de confusion du modele optimise
sns.heatmap(conf_matrix_optimized, annot=True, fmt='d', cmap='PuRd', 
            xticklabels=species_order, yticklabels=species_order,
            cbar_kws={'label': 'Nombre'},
            linewidths=2, linecolor='#8B4789', ax=axes[1])
axes[1].set_title(f'Matrice de confusion (K={best_k})', 
                 fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xlabel('Predictions', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Vraies especes', fontsize=11, color='#8B4789')

plt.tight_layout()
plt.savefig('etape6_comparaison_modeles.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape6_comparaison_modeles.png")

# ====================================================================
# ETAPE 7 : OPTIMISATION DU MODELE ET COMPARAISON
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 7 : Optimisation et comparaison de plusieurs modeles]")
print("=" * 60)

# Importer les modeles et outils necessaires
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import time

# ====================================================================
# PARTIE 7.1 : OPTIMISATION DES HYPER-PARAMETRES DU KNN
# ====================================================================

print("\n--- Partie 7.1 : Optimisation des hyper-parametres du KNN ---")
print("Nous allons tester differentes combinaisons de K et de distances...")

# Definir la grille de parametres a tester
param_grid_knn = {
    'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Creer le modele de base
knn_base = KNeighborsClassifier()

# Recherche en grille avec validation croisee
print("\nRecherche en grille en cours...")
grid_search_knn = GridSearchCV(
    knn_base, 
    param_grid_knn, 
    cv=5,  # Validation croisee a 5 plis
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)

# Entrainer avec toutes les combinaisons
start_time = time.time()
grid_search_knn.fit(X_train_scaled, y_train)
end_time = time.time()

print(f"Recherche terminee en {end_time - start_time:.2f} secondes")
print(f"\nMeilleurs parametres trouves : {grid_search_knn.best_params_}")
print(f"Meilleur score (validation croisee) : {grid_search_knn.best_score_ * 100:.2f}%")

# Predictions avec le meilleur modele KNN
best_knn = grid_search_knn.best_estimator_
y_pred_best_knn = best_knn.predict(X_test_scaled)
accuracy_best_knn = accuracy_score(y_test, y_pred_best_knn)

print(f"Exactitude sur l'ensemble de test : {accuracy_best_knn * 100:.2f}%")

# Visualiser les resultats de la recherche en grille
results_df = pd.DataFrame(grid_search_knn.cv_results_)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Optimisation des hyper-parametres du KNN', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Graphique 1 : Score selon K pour chaque metrique
for metric in ['euclidean', 'manhattan', 'minkowski']:
    mask = results_df['param_metric'] == metric
    data = results_df[mask]
    axes[0].plot(data['param_n_neighbors'], data['mean_test_score'], 
                marker='o', label=metric, linewidth=2, markersize=8)

axes[0].set_xlabel('Nombre de voisins (K)', fontsize=11, color='#8B4789')
axes[0].set_ylabel('Score moyen (validation croisee)', fontsize=11, color='#8B4789')
axes[0].set_title('Performance selon K et la distance', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].legend()
axes[0].grid(True, alpha=0.3, color='#D4A5D4')
axes[0].set_facecolor('#FFF5FA')

# Graphique 2 : Heatmap des performances
pivot_data = results_df.pivot_table(
    values='mean_test_score', 
    index='param_metric', 
    columns='param_n_neighbors'
)
sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='PuRd', 
            linewidths=1, linecolor='#8B4789', ax=axes[1], cbar_kws={'label': 'Score'})
axes[1].set_title('Heatmap des performances', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xlabel('Nombre de voisins (K)', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Metrique de distance', fontsize=11, color='#8B4789')

plt.tight_layout()
plt.savefig('etape7_optimisation_knn.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape7_optimisation_knn.png")

# ====================================================================
# PARTIE 7.2 : ENTRAINEMENT ET COMPARAISON DE PLUSIEURS MODELES
# ====================================================================

print("\n--- Partie 7.2 : Comparaison de differents modeles ---")
print("Nous allons entrainer 6 modeles differents et comparer leurs performances...\n")

# Dictionnaire des modeles a tester
models = {
    'KNN Optimise': best_knn,
    'Regression Logistique': LogisticRegression(max_iter=200, random_state=42),
    'Arbre de Decision': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', random_state=42),
    'Reseau de Neurones': MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
}

# Stocker les resultats
results = {
    'Modele': [],
    'Exactitude Train': [],
    'Exactitude Test': [],
    'Score CV': [],
    'Temps (s)': []
}

# Entrainer et evaluer chaque modele
print("Entrainement des modeles en cours...\n")
print("-" * 70)
print(f"{'Modele':<25} {'Train':<12} {'Test':<12} {'CV Score':<12} {'Temps':<10}")
print("-" * 70)

for name, model in models.items():
    # Mesurer le temps
    start_time = time.time()
    
    # Entrainer le modele (sauf KNN deja entraine)
    if name != 'KNN Optimise':
        model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Scores
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # Validation croisee
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_mean = cv_scores.mean()
    
    # Temps
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Stocker les resultats
    results['Modele'].append(name)
    results['Exactitude Train'].append(train_acc)
    results['Exactitude Test'].append(test_acc)
    results['Score CV'].append(cv_mean)
    results['Temps (s)'].append(elapsed_time)
    
    # Afficher
    print(f"{name:<25} {train_acc*100:>6.2f}%     {test_acc*100:>6.2f}%     {cv_mean*100:>6.2f}%     {elapsed_time:>6.3f}s")

print("-" * 70)
print("\nEntrainement termine !\n")

# Creer un DataFrame des resultats
results_df = pd.DataFrame(results)
print("Tableau recapitulatif des performances :")
print(results_df.to_string(index=False))

# Identifier le meilleur modele
best_model_idx = results_df['Exactitude Test'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Modele']
best_model_score = results_df.loc[best_model_idx, 'Exactitude Test']

print(f"\n🏆 Meilleur modele : {best_model_name} avec {best_model_score*100:.2f}% d'exactitude")

# ====================================================================
# VISUALISATION COMPARATIVE DES MODELES
# ====================================================================

print("\nCreation des graphiques comparatifs...")

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle('Comparaison des performances des modeles', 
             fontsize=16, fontweight='bold', color='#8B4789')

# Graphique 1 : Exactitude Train vs Test
ax1 = plt.subplot(2, 2, 1)
x_pos = np.arange(len(results_df))
width = 0.35

bars1 = ax1.bar(x_pos - width/2, results_df['Exactitude Train'], width,
               label='Train', color='#E6B3FF', edgecolor='#8B4789', linewidth=1.5)
bars2 = ax1.bar(x_pos + width/2, results_df['Exactitude Test'], width,
               label='Test', color='#FFB3D9', edgecolor='#8B4789', linewidth=1.5)

ax1.set_xlabel('Modele', fontsize=10, color='#8B4789')
ax1.set_ylabel('Exactitude', fontsize=10, color='#8B4789')
ax1.set_title('Exactitude Train vs Test', fontsize=12, fontweight='bold', color='#8B4789')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(results_df['Modele'], rotation=45, ha='right', fontsize=9)
ax1.legend()
ax1.set_facecolor('#FFF5FA')
ax1.grid(axis='y', alpha=0.3, color='#D4A5D4')
ax1.set_ylim([0.8, 1.05])

# Ajouter les valeurs
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%', ha='center', va='bottom', 
                fontweight='bold', color='#8B4789', fontsize=7)

# Graphique 2 : Score de validation croisee
ax2 = plt.subplot(2, 2, 2)
bars = ax2.barh(results_df['Modele'], results_df['Score CV'], 
               color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
ax2.set_xlabel('Score CV (moyenne)', fontsize=10, color='#8B4789')
ax2.set_title('Score de validation croisee', fontsize=12, fontweight='bold', color='#8B4789')
ax2.set_facecolor('#FFF5FA')
ax2.grid(axis='x', alpha=0.3, color='#D4A5D4')
ax2.set_xlim([0.8, 1.05])

for i, bar in enumerate(bars):
    width = bar.get_width()
    ax2.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width*100:.2f}%', ha='left', va='center', 
            fontweight='bold', color='#8B4789', fontsize=9, 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8B4789'))

# Graphique 3 : Temps d'execution
ax3 = plt.subplot(2, 2, 3)
bars = ax3.bar(results_df['Modele'], results_df['Temps (s)'], 
              color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
ax3.set_xlabel('Modele', fontsize=10, color='#8B4789')
ax3.set_ylabel('Temps (secondes)', fontsize=10, color='#8B4789')
ax3.set_title('Temps d\'execution', fontsize=12, fontweight='bold', color='#8B4789')
ax3.set_xticklabels(results_df['Modele'], rotation=45, ha='right', fontsize=9)
ax3.set_facecolor('#FFF5FA')
ax3.grid(axis='y', alpha=0.3, color='#D4A5D4')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}s', ha='center', va='bottom', 
            fontweight='bold', color='#8B4789', fontsize=8)

# Graphique 4 : Classement global
ax4 = plt.subplot(2, 2, 4)
# Creer un score composite (Test + CV) / 2
results_df['Score Composite'] = (results_df['Exactitude Test'] + results_df['Score CV']) / 2
results_sorted = results_df.sort_values('Score Composite', ascending=True)

bars = ax4.barh(results_sorted['Modelez'], results_sorted['Score Composite'], 
               color=PALETTE_ROSE_VIOLET, edgecolor='#8B4789', linewidth=1.5)
ax4.set_xlabel('Score composite', fontsize=10, color='#8B4789')
ax4.set_title('Classement global (Test + CV)/2', fontsize=12, fontweight='bold', color='#8B4789')
ax4.set_facecolor('#FFF5FA')
ax4.grid(axis='x', alpha=0.3, color='#D4A5D4')
ax4.set_xlim([0.8, 1.05])

# Mettre en evidence le meilleur
for i, bar in enumerate(bars):
    width = bar.get_width()
    if results_sorted.iloc[i]['Modele'] == best_model_name:
        bar.set_color('#C71585')
        bar.set_linewidth(3)
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{width*100:.2f}%', ha='left', va='center', 
            fontweight='bold', color='#8B4789', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='#8B4789'))

plt.tight_layout()
plt.savefig('etape7_comparaison_modeles.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape7_comparaison_modeles.png")

# ====================================================================
# ANALYSE DETAILLEE DU MEILLEUR MODELE
# ====================================================================

print("\n--- Analyse detaillee du meilleur modele ---")
print(f"Modele selectionne : {best_model_name}\n")

# Recuperer le meilleur modele
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

# Matrice de confusion
conf_matrix_best = confusion_matrix(y_test, y_pred_best, labels=species_order)

# Rapport de classification
print("Rapport de classification du meilleur modele :")
print(classification_report(y_test, y_pred_best, target_names=species_order))

# Visualisation
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.patch.set_facecolor('#FFF5FA')
fig.suptitle(f'Analyse detaillee : {best_model_name}', 
             fontsize=14, fontweight='bold', color='#8B4789')

# Matrice de confusion
sns.heatmap(conf_matrix_best, annot=True, fmt='d', cmap='PuRd', 
            xticklabels=species_order, yticklabels=species_order,
            cbar_kws={'label': 'Nombre'},
            linewidths=2, linecolor='#8B4789', ax=axes[0])
axes[0].set_title('Matrice de confusion', fontsize=12, fontweight='bold', color='#8B4789')
axes[0].set_xlabel('Predictions', fontsize=11, color='#8B4789')
axes[0].set_ylabel('Vraies especes', fontsize=11, color='#8B4789')

# Metriques par classe
precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred_best, labels=species_order)

x_pos = np.arange(len(species_order))
width = 0.25

bars1 = axes[1].bar(x_pos - width, precision, width, label='Precision', 
                   color='#FFB3D9', edgecolor='#8B4789', linewidth=1.5)
bars2 = axes[1].bar(x_pos, recall, width, label='Recall', 
                   color='#E6B3FF', edgecolor='#8B4789', linewidth=1.5)
bars3 = axes[1].bar(x_pos + width, f1, width, label='F1-score', 
                   color='#D4A5D4', edgecolor='#8B4789', linewidth=1.5)

axes[1].set_xlabel('Espece', fontsize=11, color='#8B4789')
axes[1].set_ylabel('Score', fontsize=11, color='#8B4789')
axes[1].set_title('Metriques par espece', fontsize=12, fontweight='bold', color='#8B4789')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(species_order)
axes[1].legend()
axes[1].set_ylim([0, 1.1])
axes[1].set_facecolor('#FFF5FA')
axes[1].grid(axis='y', alpha=0.3, color='#D4A5D4')

plt.tight_layout()
plt.savefig('etape7_meilleur_modele.png', dpi=300, bbox_inches='tight', facecolor='#FFF5FA')
plt.show()

print("\nOK - Graphique sauvegarde : etape7_meilleur_modele.png")

# ====================================================================
# RESUME FINAL DE L'ETAPE 7
# ====================================================================

print("\n" + "=" * 60)
print("RESUME DE L'ETAPE 7")
print("=" * 60)
print("\n✓ Optimisation des hyper-parametres du KNN realisee")
print(f"  - Meilleurs parametres KNN : {grid_search_knn.best_params_}")
print(f"\n✓ Comparaison de 6 modeles differents effectuee")
print(f"  - Meilleur modele : {best_model_name}")
print(f"  - Exactitude : {best_model_score*100:.2f}%")
print(f"\n✓ Tous les graphiques comparatifs ont ete generes")
print("=" * 60)

# ====================================================================
# ETAPE 8 : DEPLOIEMENT DU MODELE
# ====================================================================

print("\n" + "=" * 60)
print("[ETAPE 8 : Deploiement du modele]")
print("=" * 60)

# ====================================================================
# PARTIE 8.1 : SAUVEGARDE DU MODELE ET DU SCALER
# ====================================================================

print("\n--- Partie 8.1 : Sauvegarde du modele ---")

import pickle
import joblib

# Sauvegarder le meilleur modele avec pickle
print("\nSauvegarde du meilleur modele et du scaler...")

# Methode 1 : Avec pickle
with open('best_model.pkl', 'wb') as file:
    pickle.dump(best_model, file)
print("✓ Modele sauvegarde : best_model.pkl")

# Sauvegarder aussi le scaler (tres important !)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("✓ Scaler sauvegarde : scaler.pkl")

# Methode 2 : Avec joblib (alternative, plus efficace pour les gros modeles)
joblib.dump(best_model, 'best_model_joblib.pkl')
joblib.dump(scaler, 'scaler_joblib.pkl')
print("✓ Version joblib aussi sauvegardee")

# Sauvegarder aussi les informations sur le modele
model_info = {
    'model_name': best_model_name,
    'accuracy': best_model_score,
    'features': list(X.columns),
    'species': species_order,
    'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
}

with open('model_info.pkl', 'wb') as file:
    pickle.dump(model_info, file)
print("✓ Informations du modele sauvegardees : model_info.pkl")

# Tester le chargement
print("\nTest de chargement du modele...")
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Faire une prediction de test
test_prediction = loaded_model.predict(loaded_scaler.transform(X_test[:1]))
print(f"✓ Modele charge avec succes ! Prediction test : {test_prediction[0]}")

print("\n" + "=" * 60)
print("FICHIERS CREES :")
print("  - best_model.pkl (modele principal)")
print("  - scaler.pkl (normaliseur)")
print("  - model_info.pkl (informations)")
print("=" * 60)