# =========================
# 1. Import des bibliothèques
# =========================
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "voitures_avito_clean.csv")

df = pd.read_csv(file_path)
print(df.head())
print(df.info())

# =========================
# 3. Étude de la distribution des prix et d'autres variables
# =========================
cols_num = [
    "Prix", "Année-Modèle", "Kilométrage", "Puissance fiscale", "État"
]

plt.figure(figsize=(15, 10))
sns.set(style="whitegrid")
for i, col in enumerate(cols_num, 1):
    plt.subplot(2, 3, i)
    sns.histplot(data=df, x=col, bins=30, kde=True, color='cornflowerblue')
    plt.title(f'Distribution de {col}', fontsize=12, fontweight='bold')
    plt.xlabel(col)
    plt.ylabel('Fréquence')

plt.tight_layout()
plt.show()

# =========================
# 4. Boxplots des variables numériques
# =========================
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[cols_num])
plt.title('Boxplot des variables numériques')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# =========================
# 5. Matrice de corrélation (Heatmap)
# =========================
colonnes_corr = [
    "Année-Modèle",
    "Kilométrage",
    "Nombre de portes",
    "Première main",
    "Puissance fiscale",
    "État",
    "Prix"
]

correlation_matrix = df[colonnes_corr].corr(numeric_only=True)

plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap="coolwarm",
    fmt=".2f",
)

plt.title("Matrice de corrélation", fontsize=16, pad=20)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# 6. Corrélation des variables avec le prix (Barplot)
# =========================
price_corr = correlation_matrix['Prix'].drop('Prix')
price_corr_sorted = price_corr.abs().sort_values(ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(x=price_corr_sorted.values, y=price_corr_sorted.index, palette="coolwarm")
plt.title("Corrélation avec le prix", fontsize=16)
plt.xlabel("Coefficient de corrélation")
plt.ylabel("Variables")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# =========================
# 7. Détection et traitement des valeurs aberrantes
# =========================

def remove_outliers_iqr(df, target_columns, verbose=True):

    filtered_df = df.copy()
    initial_count = len(filtered_df)
    removed_counts = {}

    masks = []
    for col in target_columns:
        if col not in filtered_df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")

        Q1 = filtered_df[col].quantile(0.25)
        Q3 = filtered_df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        mask = filtered_df[col].between(lower_bound, upper_bound)
        masks.append(mask)

    combined_mask = pd.concat(masks, axis=1).all(axis=1)

    for i, col in enumerate(target_columns):
        removed_counts[col] = (~masks[i]).sum()
        if verbose:
            print(f"[{col}]: {removed_counts[col]} outliers removed.")

    filtered_df = filtered_df.loc[combined_mask].reset_index(drop=True)

    if verbose:
        total_removed = initial_count - len(filtered_df)
        print(f"Total rows removed: {total_removed} (from {initial_count} to {len(filtered_df)})." )

    return filtered_df, removed_counts

  
colonnes_cibles = [
    "Prix", "Année-Modèle", "Kilométrage",
    "Nombre de portes", "Puissance fiscale", "État"
]

# print(df.shape)

df = remove_outliers_iqr(df, colonnes_cibles)

# print(df.shape)

# Plot the boxplot for cleaned data
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=df[colonnes_cibles])
# plt.title('Boxplot des variables numériques')
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# df.to_csv('voitures_avito_clean.csv', index=False)