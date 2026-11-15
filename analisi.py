import os
import re
from io import StringIO
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

MINUTI_MINIMI = 900    # Minimo 10 partite intere giocate
N_CLUSTER = 4

html_path = os.path.expanduser("~/Desktop/fbref.html")
with open(html_path, "r", encoding="utf-8") as f:
    html = f.read()

html_clean = re.sub(r'<!--|-->', '', html)
soup = BeautifulSoup(html_clean, 'lxml')
table = soup.find('table', {'id': 'stats_standard'})

df = pd.read_html(StringIO(str(table)))[0]

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[1] if 'Unnamed' in str(col[0]) else f"{col[0]}_{col[1]}" for col in df.columns]

df = df[df['Player'] != 'Player'].copy()

# Pulizia numeri e filtro minutaggio (escludendo i portieri)
df['Min_clean'] = pd.to_numeric(df['Playing Time_Min'].astype(str).str.replace(',', ''), errors='coerce')
df['MP_clean'] = pd.to_numeric(df['Playing Time_MP'], errors='coerce')
df = df[(df['Min_clean'] >= MINUTI_MINIMI) & (~df['Pos'].astype(str).str.contains('GK', na=False))].copy()

df['90s'] = df['Min_clean'] / 90.0
df['Gol_Azione_p90'] = pd.to_numeric(df['Performance_G-PK'], errors='coerce') / df['90s']
df['Assist_p90'] = pd.to_numeric(df['Performance_Ast'], errors='coerce') / df['90s']
df['Gialli_p90'] = pd.to_numeric(df['Performance_CrdY'], errors='coerce') / df['90s']
df['Min_per_Partita'] = df['Min_clean'] / df['MP_clean']
df['Rigori_p90'] = pd.to_numeric(df['Performance_PKatt'], errors='coerce') / df['90s']

features = ['Gol_Azione_p90', 'Assist_p90', 'Gialli_p90', 'Min_per_Partita', 'Rigori_p90']
X = df[features].fillna(0)

# Standardizzazione e K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=N_CLUSTER, init='k-means++', random_state=42, n_init=15)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df['PC1'] = coords[:, 0]
df['PC2'] = coords[:, 1]

print(f"\nAnalizzati {len(df)} giocatori della Serie A 2025/26 (min. {MINUTI_MINIMI}' giocati)")
print("\n--- ANTEPRIMA CLUSTER ---")
for c in range(N_CLUSTER):
    subset_c = df[df['Cluster'] == c].copy()
    subset_c['score'] = subset_c['Gol_Azione_p90'] + subset_c['Assist_p90'] + (subset_c['Min_clean'] / 3000)
    top_players = subset_c.sort_values(by='score', ascending=False)['Player'].head(6).tolist()
    print(f"Cluster {c} ({len(subset_c)} giocatori): {', '.join(top_players)}")

df[['Player', 'Squad', 'Pos', 'Min_clean', 'Cluster'] + features].to_csv('dati_cluster_serie_a.csv', index=False)

nomi_cluster = {
    0: "Jolly & Agonismo (56)",
    1: "Decisivi su Azione (89)",
    2: "Pilastri Titolari (142)",
    3: "Specialisti & Rigoristi (20)"
}
df['Profilo'] = df['Cluster'].map(nomi_cluster)
palette_social = ['#00F5D4', '#FF007F', '#FEE440', '#9B5DE5']

# GRAFICO 1: SCATTER PLOT (Formato 4:5 IG)
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 12.5))
fig.patch.set_facecolor('#0D1117')
ax.set_facecolor('#0D1117')

sns.scatterplot(
    data=df, x='PC1', y='PC2', hue='Profilo',
    palette=palette_social, s=160, alpha=0.85, edgecolor='white', linewidth=0.8, ax=ax
)

for c in range(N_CLUSTER):
    subset = df[df['Cluster'] == c].copy()
    subset['score'] = subset['Gol_Azione_p90'] + subset['Assist_p90'] + (subset['Min_clean'] / 3000)
    for _, row in subset.sort_values(by='score', ascending=False).head(4).iterrows():
        cognome = str(row['Player']).split()[-1]
        ax.text(
            row['PC1'] + 0.08, row['PC2'] + 0.05, cognome,
            fontsize=11, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#161B22', edgecolor='none', alpha=0.7)
        )

ax.set_title('SERIE A 2025/26 • MAPPA PROFILI GIOCATORI', fontsize=18, fontweight='bold', pad=20, color='white')
ax.set_xlabel('Impatto Offensivo (Gol e Assist / 90\') →', fontsize=11, color='#8B949E')
ax.set_ylabel('← Disciplina / Interdizione vs Continuità →', fontsize=11, color='#8B949E')
ax.grid(True, linestyle='--', alpha=0.15)
ax.legend(title='Cluster K-Means', title_fontsize=12, fontsize=10, loc='upper left', frameon=True, facecolor='#161B22')

plt.figtext(0.5, 0.01, 'Dati FBref (min. 900\' giocati) • Analisi K-Means + PCA', ha='center', fontsize=9, color='#8B949E')
plt.savefig('1_mappa_cluster_serie_a.png', dpi=300, bbox_inches='tight')
plt.close()

# GRAFICO 2: RADAR CHART DEGLI IDENTIKIT
scaler_radar = MinMaxScaler()
df_radar = pd.DataFrame(scaler_radar.fit_transform(df[features]), columns=features)
df_radar['Cluster'] = df['Cluster'].values
cluster_means = df_radar.groupby('Cluster').mean()

etichette_radar = ['Gol su Azione\n(/90\')', 'Assist\n(/90\')', 'Aggressività\n(Gialli /90\')', 'Titolarità\n(Min/Partita)', 'Rigori\n(/90\')']
num_vars = len(etichette_radar)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, axes = plt.subplots(2, 2, figsize=(12, 12), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#0D1117')
axes = axes.flatten()

for idx in range(N_CLUSTER):
    ax = axes[idx]
    ax.set_facecolor('#161B22')
    values = cluster_means.loc[idx].tolist()
    values += values[:1]
    
    colore = palette_social[idx]
    ax.plot(angles, values, color=colore, linewidth=2.5)
    ax.fill(angles, values, color=colore, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(etichette_radar, color='white', size=10, fontweight='bold')
    ax.set_yticklabels([])
    ax.spines['polar'].set_color('#30363D')
    ax.grid(color='#30363D', linestyle='--', alpha=0.5)
    ax.set_title(nomi_cluster[idx], size=13, color=colore, fontweight='bold', pad=18)

plt.suptitle('IDENTIKIT MEDIO DEI 4 PROFILI • SERIE A 2025/26', fontsize=18, fontweight='bold', color='white', y=1.02)
plt.tight_layout()
plt.savefig('2_identikit_cluster_serie_a.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nCompletato! Apro la cartella con i grafici...")
