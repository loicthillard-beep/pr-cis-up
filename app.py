import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import requests
import io
import zipfile
import os
import numpy as np
import shutil
import itertools
from datetime import datetime, date, timedelta
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════
# 1. CONFIG PAGE & STYLE
# ══════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Précis'Up", layout="wide", page_icon="🌱",
                    initial_sidebar_state="expanded")

# ── Imports critiques protégés : jamais un écran "Oh no" vide ─────────────
try:
    import shapefile as pyshp  # pyshp — lecture pure Python, sans GDAL
    HAS_PYSHP = True
    PYSHP_ERROR = None
except Exception as e:
    HAS_PYSHP = False
    PYSHP_ERROR = str(e)

try:
    import statsmodels.formula.api as smf
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

if not HAS_PYSHP:
    st.error(
        "❌ Le module **pyshp** n'a pas pu être chargé.\n\n"
        f"Détail technique : `{PYSHP_ERROR}`\n\n"
        "Ajoutez `pyshp` à votre `requirements.txt`, puis **Manage app → Reboot app**."
    )
    st.stop()

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #fafbf9; }

.hero {
    background: linear-gradient(120deg, #1b4332 0%, #2d6a4f 55%, #40916c 100%);
    border-radius: 18px; padding: 28px 32px; margin-bottom: 22px;
    box-shadow: 0 8px 24px rgba(27,67,50,0.25);
}
.hero h1 { color: #fff; margin: 0; font-size: 2.1rem; font-weight: 800; }
.hero p  { color: #d8f3dc; margin: 4px 0 0 0; font-size: 1rem; }

[data-testid="stMetricValue"] { font-size: 1.55rem; font-weight: 800; color:#1b4332; }
[data-testid="stMetricLabel"] { font-weight: 600; color:#52796f; }
[data-testid="stMetric"] {
    background: #ffffff; border-radius: 16px; padding: 16px 18px;
    box-shadow: 0 3px 14px rgba(27,67,50,0.07); border: 1px solid #eef2ef;
    transition: transform .15s ease;
}
[data-testid="stMetric"]:hover { transform: translateY(-2px); }

h1, h2, h3 { color:#1b4332; font-weight: 800; }
h2 { border-bottom: 3px solid #d8f3dc; padding-bottom: 6px; }

.verdict-sig {
    background: linear-gradient(135deg,#d8f3dc,#b7e4c7); border-left:6px solid #2d6a4f;
    padding:18px 22px; border-radius:14px; color:#1b4332; font-size:1.05rem;
    box-shadow: 0 4px 14px rgba(45,106,79,0.12);
}
.verdict-nosig {
    background: linear-gradient(135deg,#fde2e2,#f8c6c6); border-left:6px solid #c0392b;
    padding:18px 22px; border-radius:14px; color:#7b241c; font-size:1.05rem;
    box-shadow: 0 4px 14px rgba(192,57,43,0.10);
}
.verdict-dominant {
    background: linear-gradient(135deg,#fff3cd,#ffe8a1); border-left:6px solid #b8860b;
    padding:18px 22px; border-radius:14px; color:#5c4400; font-size:1.02rem;
    box-shadow: 0 4px 14px rgba(184,134,11,0.12);
}
.stress-high { background:#fdecea; border-left:6px solid #e74c3c; padding:14px 18px; border-radius:10px; color:#7b241c; }
.stress-low  { background:#eafaf1; border-left:6px solid #27ae60; padding:14px 18px; border-radius:10px; color:#145a32; }

.vulgarisation {
    background:#f1f5f2; border-left:4px solid #74a892; padding:14px 18px;
    margin-bottom:14px; border-radius:10px; font-size:0.92rem; color:#33433a;
}
.badge { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.78rem; font-weight:700; margin-right:6px; }
.badge-orange { background:#ffe8cc; color:#a65c00; }
.badge-blue { background:#d6e9f8; color:#1b4f72; }
.badge-gold { background:#fff3cd; color:#5c4400; }
.badge-ref { background:#fde2e2; color:#7b241c; }

hr { border-top: 1px solid #e0e4e1; }
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { background:#f1f5f2; border-radius:10px 10px 0 0; padding:10px 18px; font-weight:600; }
.stTabs [aria-selected="true"] { background:#2d6a4f !important; color:#fff !important; }
</style>
""", unsafe_allow_html=True)


def clear_temp():
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.makedirs("temp")


ALPHA_LEVELS = {"5 % (standard)": 0.05, "1 % (strict)": 0.01, "10 % (exploratoire)": 0.10}
PARAM_CULTURES = {
    "Blé Tendre": {"t_echaudage": 25, "t_critique": 30, "t_gel": -2, "precip_min_jour": 0.5},
    "Maïs":       {"t_echaudage": 32, "t_critique": 36, "t_gel": 0,  "precip_min_jour": 0.5},
    "Orge":       {"t_echaudage": 25, "t_critique": 30, "t_gel": -3, "precip_min_jour": 0.5},
    "Colza":      {"t_echaudage": 27, "t_critique": 32, "t_gel": -5, "precip_min_jour": 0.5},
    "Tournesol":  {"t_echaudage": 30, "t_critique": 35, "t_gel": -2, "precip_min_jour": 0.5},
}
PALETTE_BANDES = ['#2d6a4f', '#40916c', '#74c69d', '#95d5b2', '#52b788', '#081c15', '#1b4332', '#a3b18a']
COULEUR_REFERENCE = '#c0392b'

# ══════════════════════════════════════════════════════════════════════════
# 2. SIDEBAR (partie fixe, indépendante du fichier chargé)
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌱 Précis'Up")
    st.caption("Analyse statistique d'essais en bandes — v8.0")
    st.divider()

    with st.expander("📥 IMPORTATION DONNÉES", expanded=True):
        uploaded_file = st.file_uploader("Fichier QGIS (.zip)", type=["zip"])
        st.caption("Colonnes attendues : `bande`, `rdt`, `potentiel` (optionnel), + toute variable de sol numérique (pH, argile, MO, CEC, altitude, NDVI…)")

    with st.expander("🌾 CONFIGURATION ESSAI", expanded=True):
        culture = st.selectbox("Culture", list(PARAM_CULTURES.keys()))
        d_semis = st.date_input("Date de Semis", date(2024, 10, 20))
        d_appli = st.date_input("Date d'Application produit", date(2025, 3, 10))
        d_recolt = st.date_input("Date de Récolte", date(2025, 7, 15))
        alpha = st.selectbox("Seuil de significativité α", list(ALPHA_LEVELS.keys()))
        alpha_v = ALPHA_LEVELS[alpha]
        clean_iqr = st.checkbox("Nettoyage strict des outliers (IQR 1.2)", value=True)

    with st.expander("📊 OPTIONS STATISTIQUES", expanded=True):
        run_anova = st.checkbox("Activer l'ANOVA spatiale (si ≥ 2 zones de potentiel)", value=True)
        st.caption("Chaque bande est comparée à la bande de référence avec le test le plus adapté, choisi automatiquement.")

    with st.expander("💰 ÉCONOMIE", expanded=True):
        prix_vente = st.number_input("Prix de vente (€/T)", value=210)
        cout_prod = st.number_input("Coût Produit (€/ha)", value=45)

    with st.expander("🌦️ MÉTÉO (position de la parcelle)", expanded=True):
        st.caption("Indiquez les coordonnées GPS approximatives de votre parcelle (clic droit sur Google Maps → copier les coordonnées).")
        lat_input = st.number_input("Latitude", value=48.8566, format="%.4f")
        lon_input = st.number_input("Longitude", value=2.3522, format="%.4f")

    with st.expander("🌡️ SEUILS DE STRESS (ajustables)", expanded=True):
        st.caption("Réglez vous-même les seuils selon votre culture, votre région ou votre variété.")
        t_echaudage = st.slider("Seuil chaleur — échaudage (°C)", 15, 45, 25)
        t_critique = st.slider("Seuil chaleur — critique (°C)", t_echaudage, 50, max(t_echaudage + 5, 30))
        t_gel = st.slider("Seuil de gel (°C)", -15, 5, -2)
        precip_min_jour = st.slider("Pluie minimale considérée comme utile (mm/jour)", 0.0, 5.0, 0.5, step=0.1)
        jours_secheresse = st.slider("Nb de jours secs consécutifs = séquence de sécheresse", 3, 21, 7)


# ══════════════════════════════════════════════════════════════════════════
# 3. MOTEUR STATISTIQUE — sélection automatique et adaptative du test
# ══════════════════════════════════════════════════════════════════════════
def format_pval(p):
    """Évite l'affichage trompeur '0.0000' : une p-value n'est jamais exactement 0,
    elle peut juste être extrêmement petite (gros échantillons, effet très net)."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "—"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def cohen_d(a, b):
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 + (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "négligeable"
    if d < 0.5: return "faible"
    if d < 0.8: return "moyen"
    if d < 1.2: return "fort"
    return "très fort"


def run_main_test(data_a, data_b, alpha_v=0.05):
    """
    Sélectionne automatiquement le test le plus adapté à l'échantillon pour comparer
    deux bandes entre elles :
    - n trop petit (< 8 par groupe) : Shapiro non fiable -> Mann-Whitney directement.
    - n suffisant : Shapiro (normalité) + Levene (homogénéité) -> Student / Welch / Mann-Whitney.
    """
    n_a, n_b = len(data_a), len(data_b)
    small_sample = n_a < 8 or n_b < 8
    p_shap_a = p_shap_b = p_lev = None

    if small_sample:
        t_stat, p_main = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
        test_nom = "Mann-Whitney U (échantillon réduit → test robuste)"
    else:
        _, p_shap_a = stats.shapiro(data_a)
        _, p_shap_b = stats.shapiro(data_b)
        _, p_lev = stats.levene(data_a, data_b)
        normal = p_shap_a > alpha_v and p_shap_b > alpha_v
        homog = p_lev > alpha_v

        if normal and homog:
            t_stat, p_main = stats.ttest_ind(data_a, data_b)
            test_nom = "Test de Student (paramétrique)"
        elif normal:
            t_stat, p_main = stats.ttest_ind(data_a, data_b, equal_var=False)
            test_nom = "Test de Welch (variances inégales)"
        else:
            t_stat, p_main = stats.mannwhitneyu(data_a, data_b, alternative='two-sided')
            test_nom = "Mann-Whitney U (non-paramétrique)"

    d = cohen_d(data_a.values, data_b.values)
    return {
        'name': test_nom, 'p': p_main, 'd': d, 'label': interpret_d(d),
        'shapiro_a': p_shap_a, 'shapiro_b': p_shap_b, 'levene_p': p_lev,
        'small_sample': small_sample,
    }


def run_anova_analysis(df_final, alpha_v=0.05):
    if not HAS_STATSMODELS:
        return None, "statsmodels non installé.", None, False
    if 'potentiel' not in df_final.columns:
        return None, "Colonne 'potentiel' absente du fichier : l'ANOVA spatiale nécessite des zones de sol.", None, False
    nb_zones = df_final['potentiel'].dropna().nunique()
    if nb_zones <= 1:
        return None, ("Une seule zone de potentiel détectée. L'ANOVA à 2 facteurs n'a de sens que pour "
                       "croiser le traitement avec le sol — sans variation de sol elle se réduirait au "
                       "test principal déjà calculé. Ajoutez au moins 2 zones de potentiel pour l'activer."), None, False

    formula = "rdt ~ C(grp) + C(potentiel) + C(grp):C(potentiel)"
    try:
        model = smf.ols(formula, data=df_final).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)
        return anova_t, "📐 ANOVA à 2 facteurs : Bande (traitement) × Zone de potentiel", model, True
    except Exception as e:
        return None, f"Erreur lors du calcul statistique : {e}", None, False


# ══════════════════════════════════════════════════════════════════════════
# 3-bis. QUI A LE PLUS D'EFFET : LE SOL OU LE TRAITEMENT (BANDES) ?
# ══════════════════════════════════════════════════════════════════════════
def compare_soil_vs_product(anova_t, alpha_v=0.05):
    """
    Répond à la question centrale : est-ce le SOL (zone de potentiel) ou le TRAITEMENT
    (facteur bande) qui explique le plus la variation de rendement observée ?

    Méthode : à partir du tableau ANOVA à 2 facteurs (Bande, Sol, Interaction, Résidus),
    on calcule l'eta² de chaque facteur = SCE_facteur / SCE_totale, c'est-à-dire la part
    de la variabilité totale du rendement expliquée par ce facteur. On compare ensuite :
    - la significativité (p-value < alpha) de chaque facteur,
    - puis, si les deux sont significatifs, l'ampleur (eta²) de chacun.
    """
    if anova_t is None:
        return None
    try:
        sce_total = anova_t['sum_sq'].sum()
        row_grp = anova_t.loc['C(grp)']
        row_pot = anova_t.loc['C(potentiel)']
        row_inter = anova_t.loc['C(grp):C(potentiel)']

        eta2_grp = float(row_grp['sum_sq'] / sce_total)
        eta2_pot = float(row_pot['sum_sq'] / sce_total)
        eta2_inter = float(row_inter['sum_sq'] / sce_total)

        p_grp = float(row_grp['PR(>F)'])
        p_pot = float(row_pot['PR(>F)'])
        p_inter = float(row_inter['PR(>F)'])

        sig_grp = p_grp < alpha_v
        sig_pot = p_pot < alpha_v
        sig_inter = p_inter < alpha_v

        if not sig_grp and not sig_pot:
            dominant = "aucun"
            verdict = ("Ni le **traitement (bandes)**, ni le **sol** ne montrent d'effet statistiquement démontré "
                       "sur le rendement dans ce jeu de données : la variabilité observée reste dans "
                       "les limites du hasard d'échantillonnage.")
        elif sig_grp and not sig_pot:
            dominant = "produit"
            verdict = (f"C'est le **TRAITEMENT (bandes)** qui explique la variation de rendement observée "
                       f"(effet significatif, eta² = {eta2_grp:.1%} de la variance totale), "
                       f"le sol n'apportant pas de différence démontrée ici.")
        elif sig_pot and not sig_grp:
            dominant = "sol"
            verdict = (f"C'est le **SOL** (zone de potentiel) qui explique la variation de rendement observée "
                       f"(effet significatif, eta² = {eta2_pot:.1%} de la variance totale), "
                       f"le traitement (bandes) n'apportant pas de différence démontrée ici.")
        else:
            if eta2_grp > eta2_pot * 1.05:
                dominant = "produit"
                verdict = (f"Les deux facteurs sont significatifs, mais le **TRAITEMENT (bandes)** explique davantage "
                           f"la variabilité du rendement (eta² = {eta2_grp:.1%}) que le **SOL** (eta² = {eta2_pot:.1%}).")
            elif eta2_pot > eta2_grp * 1.05:
                dominant = "sol"
                verdict = (f"Les deux facteurs sont significatifs, mais le **SOL** explique davantage "
                           f"la variabilité du rendement (eta² = {eta2_pot:.1%}) que le **TRAITEMENT (bandes)** (eta² = {eta2_grp:.1%}).")
            else:
                dominant = "égalité"
                verdict = (f"Les deux facteurs sont significatifs et contribuent de façon quasi équivalente "
                           f"(Traitement eta² = {eta2_grp:.1%}, Sol eta² = {eta2_pot:.1%}).")

        if sig_inter:
            verdict += (f" Par ailleurs, l'interaction Traitement × Sol est elle-même significative "
                       f"(eta² = {eta2_inter:.1%}) : l'efficacité des bandes **varie selon la zone de sol** — "
                       f"regardez le graphique par zone pour identifier où chaque bande performe le mieux.")

        return {
            'eta2_grp': eta2_grp, 'eta2_pot': eta2_pot, 'eta2_inter': eta2_inter,
            'p_grp': p_grp, 'p_pot': p_pot, 'p_inter': p_inter,
            'sig_grp': sig_grp, 'sig_pot': sig_pot, 'sig_inter': sig_inter,
            'dominant': dominant, 'verdict': verdict,
        }
    except Exception:
        return None


def order_zone_labels(series):
    """
    Ordonne les libellés de zones de potentiel (ex: 'Faible', 'Moyen', 'Fort') dans un ordre
    naturel si reconnu, sinon par ordre alphabétique — pour un affichage cohérent des tableaux.
    """
    known_order = ['tres faible', 'très faible', 'faible', 'moyen faible', 'moyen-faible',
                   'moyen', 'moyenne', 'moyen fort', 'moyen-fort', 'fort', 'eleve', 'élevé',
                   'tres fort', 'très fort']
    vals = list(pd.unique(series.dropna()))

    def key(v):
        vl = str(v).strip().lower()
        return (known_order.index(vl) if vl in known_order else len(known_order), str(v))

    return sorted(vals, key=key)


def compare_bands_within_zones(df, zone_col, ordered_zones, bande_ref, alpha_v=0.05):
    """
    Pour chaque zone (ex: zone de potentiel déclarée dans le fichier), compare chaque bande à la
    référence avec le test le plus adapté choisi automatiquement — répond directement à
    « la bande est-elle meilleure, à potentiel de sol égal ? », zone par zone.
    """
    rows = []
    for zone in ordered_zones:
        sub = df[df[zone_col] == zone]
        if bande_ref not in sub['grp'].unique():
            continue
        data_ref_zone = sub[sub['grp'] == bande_ref]['rdt'].dropna()
        if len(data_ref_zone) < 4:
            continue
        for b in sorted(sub['grp'].unique()):
            if b == bande_ref:
                continue
            data_b = sub[sub['grp'] == b]['rdt'].dropna()
            if len(data_b) < 4:
                rows.append({
                    'Zone de potentiel': zone, 'Bande': b,
                    'N bande': len(data_b), 'N référence': len(data_ref_zone),
                    'Moyenne bande': np.nan, 'Moyenne référence': round(data_ref_zone.mean(), 2),
                    'Gain (qtx)': np.nan, 'Test': "N insuffisant (< 4 obs.)",
                    'p-value': "—", 'Significatif': "—",
                })
                continue
            res = run_main_test(data_b, data_ref_zone, alpha_v=alpha_v)
            rows.append({
                'Zone de potentiel': zone, 'Bande': b,
                'N bande': len(data_b), 'N référence': len(data_ref_zone),
                'Moyenne bande': round(data_b.mean(), 2), 'Moyenne référence': round(data_ref_zone.mean(), 2),
                'Gain (qtx)': round(data_b.mean() - data_ref_zone.mean(), 2),
                'Test': res['name'], 'p-value': format_pval(res['p']),
                'Significatif': '✅ Oui' if res['p'] < alpha_v else '❌ Non',
            })
    return pd.DataFrame(rows)


def compute_pairwise_matrices(data_dict, bands_order, alpha_v=0.05):
    """
    Calcule la matrice de p-values et la matrice de Cohen's d pour toutes les paires de bandes
    d'un dictionnaire {bande: valeurs de rendement} — réutilisable pour la comparaison croisée
    globale (toutes zones confondues) et pour la comparaison croisée au sein d'une seule zone.
    """
    bands_order = [b for b in bands_order if b in data_dict and len(data_dict[b]) > 3]
    p_matrix = pd.DataFrame(index=bands_order, columns=bands_order, dtype=object)
    d_matrix = pd.DataFrame(index=bands_order, columns=bands_order, dtype=float)
    pair_results = {}
    for a in bands_order:
        for b in bands_order:
            if a == b:
                p_matrix.loc[a, b] = "—"
                d_matrix.loc[a, b] = 0.0
            else:
                res_ab = run_main_test(data_dict[a], data_dict[b], alpha_v=alpha_v)
                sig_mark = " ✅" if res_ab['p'] < alpha_v else ""
                p_matrix.loc[a, b] = f"{format_pval(res_ab['p'])}{sig_mark}"
                d_matrix.loc[a, b] = res_ab['d']
                pair_results[(a, b)] = res_ab
    return p_matrix, d_matrix, pair_results, bands_order


def best_band_summary(data_dict, bande_ref, alpha_v=0.05):
    """
    Détermine, à partir d'un dictionnaire {bande: valeurs}, la meilleure bande vs la référence
    (même logique que le calcul global du haut de page), utilisable pour une zone de potentiel
    spécifique afin de savoir qui gagne "à sol égal".
    """
    if bande_ref not in data_dict or len(data_dict[bande_ref]) < 4:
        return None, []
    data_ref = data_dict[bande_ref]
    rows = []
    for b, data_b in data_dict.items():
        if b == bande_ref or len(data_b) < 4:
            continue
        res = run_main_test(data_b, data_ref, alpha_v=alpha_v)
        gain = data_b.mean() - data_ref.mean()
        rows.append({
            'bande': b, 'n': len(data_b), 'moyenne': data_b.mean(), 'gain': gain,
            'p': res['p'], 'sig': res['p'] < alpha_v, 'd': res['d'], 'd_label': res['label'],
            'test': res['name'],
        })
    if not rows:
        return None, []
    sig_rows = [r for r in rows if r['sig']]
    best = max(sig_rows, key=lambda r: r['gain']) if sig_rows else max(rows, key=lambda r: r['gain'])
    return best, rows


def compare_zones_for_band(df, zone_col, band, zones_selected, alpha_v=0.05):
    """
    Pour une bande donnée (y compris le témoin), compare son rendement entre chaque paire de
    zones de potentiel sélectionnées — répond à « cette bande rend-elle vraiment différemment
    selon le potentiel de sol, ou est-ce le hasard ? ».
    """
    sub_band = df[df['grp'] == band]
    rows = []
    for zone_a, zone_b in itertools.combinations(zones_selected, 2):
        data_a = sub_band[sub_band[zone_col] == zone_a]['rdt'].dropna()
        data_b = sub_band[sub_band[zone_col] == zone_b]['rdt'].dropna()
        if len(data_a) < 4 or len(data_b) < 4:
            rows.append({
                'Zone A': zone_a, 'Zone B': zone_b, 'N (A)': len(data_a), 'N (B)': len(data_b),
                'Moyenne (A)': round(data_a.mean(), 2) if len(data_a) else np.nan,
                'Moyenne (B)': round(data_b.mean(), 2) if len(data_b) else np.nan,
                'Diff (A - B)': np.nan, 'Test': "N insuffisant (< 4 obs.)",
                'p-value': "—", 'Significatif': "—",
            })
            continue
        res = run_main_test(data_a, data_b, alpha_v=alpha_v)
        rows.append({
            'Zone A': zone_a, 'Zone B': zone_b, 'N (A)': len(data_a), 'N (B)': len(data_b),
            'Moyenne (A)': round(data_a.mean(), 2), 'Moyenne (B)': round(data_b.mean(), 2),
            'Diff (A - B)': round(data_a.mean() - data_b.mean(), 2),
            'Test': res['name'], 'p-value': format_pval(res['p']),
            'Significatif': '✅ Oui' if res['p'] < alpha_v else '❌ Non',
        })
    return pd.DataFrame(rows)


def best_zone_for_band(df, zone_col, band, zones_selected, alpha_v=0.05):
    """
    Pour une bande donnée (y compris le témoin), détermine la zone de potentiel où elle rend le
    mieux et celle où elle rend le moins bien parmi les zones sélectionnées, et teste si cet
    écart est statistiquement significatif — répond à « ma bande (ou mon témoin) est-elle
    meilleure dans un potentiel que dans un autre ? ».
    """
    sub_band = df[df['grp'] == band]
    means = {}
    for z in zones_selected:
        data_z = sub_band[sub_band[zone_col] == z]['rdt'].dropna()
        if len(data_z) >= 4:
            means[z] = data_z
    if len(means) < 2:
        return None
    zone_best = max(means, key=lambda z: means[z].mean())
    zone_worst = min(means, key=lambda z: means[z].mean())
    if zone_best == zone_worst:
        return None
    res = run_main_test(means[zone_best], means[zone_worst], alpha_v=alpha_v)
    return {
        'bande': band, 'zone_meilleure': zone_best, 'zone_plus_faible': zone_worst,
        'moyenne_meilleure': means[zone_best].mean(), 'moyenne_plus_faible': means[zone_worst].mean(),
        'diff': means[zone_best].mean() - means[zone_worst].mean(),
        'p': res['p'], 'sig': res['p'] < alpha_v, 'n_zones_testees': len(means),
    }


# ══════════════════════════════════════════════════════════════════════════
# 3-ter. ACP — LIEN ENTRE VARIABLES DE SOL, PRODUIT (BANDE) ET RENDEMENT
# ══════════════════════════════════════════════════════════════════════════
def detect_numeric_soil_columns(df):
    """
    Détecte automatiquement les colonnes numériques du fichier qui peuvent servir
    de variables de sol pour l'ACP (pH, argile, matière organique, CEC, altitude,
    NDVI, réserve utile...), en excluant bande/rdt/potentiel/grp et les colonnes
    techniques (id, fid, géométrie, surface...).
    """
    exclude_exact = {'bande', 'rdt', 'grp', 'potentiel'}
    exclude_patterns = ['shape', 'fid', 'objectid', 'id', 'area', 'perimeter', 'length', 'geom']
    candidates = []
    for c in df.columns:
        if c in exclude_exact:
            continue
        if any(p in c for p in exclude_patterns):
            continue
        converted = pd.to_numeric(df[c], errors='coerce')
        if converted.notna().sum() >= max(5, int(0.5 * len(df))):
            candidates.append(c)
    return candidates


def run_pca(df, cols, n_components=3):
    """
    ACP "maison" (standardisation + SVD, sans dépendance supplémentaire) :
    - scores  : coordonnées des parcelles/points sur les axes principaux
    - loadings: contribution (corrélation) de chaque variable de sol à chaque axe
    - explained_var : part de variance expliquée par chaque axe
    """
    X = df[cols].apply(pd.to_numeric, errors='coerce').dropna()
    if X.shape[0] < 4 or X.shape[1] < 2:
        return None
    mean = X.mean()
    std = X.std(ddof=0).replace(0, 1)
    Xs = (X - mean) / std
    U, S, Vt = np.linalg.svd(Xs.values, full_matrices=False)
    k = min(n_components, len(S))
    explained_var = (S ** 2) / np.sum(S ** 2)
    scores = pd.DataFrame(U[:, :k] * S[:k], columns=[f"PC{i+1}" for i in range(k)], index=X.index)
    loadings = pd.DataFrame(Vt[:k, :].T, columns=[f"PC{i+1}" for i in range(k)], index=cols)
    return {'scores': scores, 'loadings': loadings, 'explained_var': explained_var[:k], 'index': X.index}


def compare_grp_vs_pca_axis(df_pca, alpha_v=0.05):
    """
    Répond, à partir de l'ACP, à la même question que 'compare_soil_vs_product' mais en
    utilisant l'axe de sol continu PC1 (au lieu de la seule zone 'potentiel' catégorielle) :
    est-ce le SOL (gradient résumé par PC1) ou le TRAITEMENT (bande) qui pèse le plus sur
    le rendement observé ?

    Méthode : ANCOVA rdt ~ C(grp) * PC1 (bande catégorielle x sol continu), puis eta² de
    chaque terme = SCE_terme / SCE_totale, comme pour l'ANOVA classique.
    """
    if not HAS_STATSMODELS:
        return None
    if df_pca['grp'].nunique() < 2 or df_pca['PC1'].nunique() < 3:
        return None
    try:
        model = smf.ols("rdt ~ C(grp) * PC1", data=df_pca).fit()
        anova_t = sm.stats.anova_lm(model, typ=2)

        sce_total = anova_t['sum_sq'].sum()
        row_grp = anova_t.loc['C(grp)']
        row_pc1 = anova_t.loc['PC1']
        row_inter = anova_t.loc['C(grp):PC1']

        eta2_grp = float(row_grp['sum_sq'] / sce_total)
        eta2_pc1 = float(row_pc1['sum_sq'] / sce_total)
        eta2_inter = float(row_inter['sum_sq'] / sce_total)

        p_grp = float(row_grp['PR(>F)'])
        p_pc1 = float(row_pc1['PR(>F)'])
        p_inter = float(row_inter['PR(>F)'])

        sig_grp = p_grp < alpha_v
        sig_pc1 = p_pc1 < alpha_v
        sig_inter = p_inter < alpha_v

        if not sig_grp and not sig_pc1:
            dominant = "aucun"
            verdict = ("Ni le **traitement (bandes)**, ni le **gradient de sol (PC1)** n'expliquent de façon "
                       "démontrée la variation de rendement observée ici.")
        elif sig_grp and not sig_pc1:
            dominant = "produit"
            verdict = (f"C'est le **TRAITEMENT (bandes)** qui explique la variation de rendement observée "
                       f"(eta² = {eta2_grp:.1%}), le gradient de sol (PC1) n'ayant pas d'effet démontré ici.")
        elif sig_pc1 and not sig_grp:
            dominant = "sol"
            verdict = (f"C'est le **SOL** (résumé par l'axe PC1 de l'ACP) qui explique la variation de rendement "
                       f"observée (eta² = {eta2_pc1:.1%}), le traitement (bandes) n'ayant pas d'effet démontré ici.")
        else:
            if eta2_grp > eta2_pc1 * 1.05:
                dominant = "produit"
                verdict = (f"Les deux effets sont significatifs, mais le **TRAITEMENT (bandes)** pèse davantage "
                           f"(eta² = {eta2_grp:.1%}) que le **SOL** (eta² = {eta2_pc1:.1%}) sur le rendement.")
            elif eta2_pc1 > eta2_grp * 1.05:
                dominant = "sol"
                verdict = (f"Les deux effets sont significatifs, mais le **SOL** pèse davantage "
                           f"(eta² = {eta2_pc1:.1%}) que le **TRAITEMENT (bandes)** (eta² = {eta2_grp:.1%}) sur le rendement.")
            else:
                dominant = "égalité"
                verdict = (f"Les deux effets sont significatifs et contribuent de façon quasi équivalente "
                           f"(Traitement eta² = {eta2_grp:.1%}, Sol eta² = {eta2_pc1:.1%}).")

        if sig_inter:
            verdict += (f" L'interaction Bande × Sol (PC1) est elle-même significative (eta² = {eta2_inter:.1%}) : "
                       f"l'efficacité du traitement varie selon la position de la parcelle sur ce gradient de sol.")

        return {
            'eta2_grp': eta2_grp, 'eta2_pc1': eta2_pc1, 'eta2_inter': eta2_inter,
            'p_grp': p_grp, 'p_pc1': p_pc1, 'p_inter': p_inter,
            'sig_grp': sig_grp, 'sig_pc1': sig_pc1, 'sig_inter': sig_inter,
            'dominant': dominant, 'verdict': verdict,
        }
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
# 3-quater. COMPARER LES BANDES "À NIVEAU DE SOL ÉQUIVALENT" (via PC1)
# ══════════════════════════════════════════════════════════════════════════
def bin_pca_levels(df_pca, n_levels=3):
    """
    Découpe l'axe de sol continu PC1 en niveaux de potentiel comparables (tertiles par défaut),
    pour pouvoir comparer les bandes "à niveau de sol équivalent" — comme des zones de potentiel,
    mais dérivées directement de l'ACP plutôt que d'une colonne 'potentiel' déclarative.
    """
    df_pca = df_pca.copy()
    n_levels_eff = max(2, min(n_levels, df_pca['PC1'].nunique()))
    labels_map = {
        2: ["Sol faible (PC1)", "Sol fort (PC1)"],
        3: ["Sol faible (PC1)", "Sol moyen (PC1)", "Sol fort (PC1)"],
        4: ["Sol faible (PC1)", "Sol moy.-faible (PC1)", "Sol moy.-fort (PC1)", "Sol fort (PC1)"],
    }
    labels = labels_map.get(n_levels_eff, [f"Niveau {i + 1} (PC1)" for i in range(n_levels_eff)])
    try:
        df_pca['niveau_pc1'] = pd.qcut(df_pca['PC1'], q=n_levels_eff, labels=labels, duplicates='drop')
    except Exception:
        df_pca['niveau_pc1'] = pd.cut(df_pca['PC1'], bins=n_levels_eff, labels=labels)
    df_pca['niveau_pc1'] = df_pca['niveau_pc1'].astype(str)
    ordered_levels = [l for l in labels if l in df_pca['niveau_pc1'].unique()]
    return df_pca, ordered_levels


def compare_bands_within_pca_levels(df_pca_binned, ordered_levels, bande_ref, alpha_v=0.05):
    """
    Pour chaque niveau de sol (issu du découpage de PC1), compare chaque bande à la référence,
    exactement comme le tableau ANOVA Zone × Bande, mais avec des niveaux de sol continus
    dérivés de l'ACP plutôt que de la colonne déclarative 'potentiel'.
    """
    rows = []
    for niveau in ordered_levels:
        sub = df_pca_binned[df_pca_binned['niveau_pc1'] == niveau]
        if bande_ref not in sub['grp'].unique():
            continue
        data_ref_niveau = sub[sub['grp'] == bande_ref]['rdt'].dropna()
        if len(data_ref_niveau) < 4:
            continue
        for b in sorted(sub['grp'].unique()):
            if b == bande_ref:
                continue
            data_b = sub[sub['grp'] == b]['rdt'].dropna()
            if len(data_b) < 4:
                rows.append({
                    'Niveau de sol (PC1)': niveau, 'Bande': b,
                    'N bande': len(data_b), 'N référence': len(data_ref_niveau),
                    'Moyenne bande': np.nan, 'Moyenne référence': round(data_ref_niveau.mean(), 2),
                    'Gain (qtx)': np.nan, 'Test': "N insuffisant (< 4 obs.)",
                    'p-value': "—", 'Significatif': "—",
                })
                continue
            res = run_main_test(data_b, data_ref_niveau, alpha_v=alpha_v)
            rows.append({
                'Niveau de sol (PC1)': niveau, 'Bande': b,
                'N bande': len(data_b), 'N référence': len(data_ref_niveau),
                'Moyenne bande': round(data_b.mean(), 2), 'Moyenne référence': round(data_ref_niveau.mean(), 2),
                'Gain (qtx)': round(data_b.mean() - data_ref_niveau.mean(), 2),
                'Test': res['name'], 'p-value': format_pval(res['p']),
                'Significatif': '✅ Oui' if res['p'] < alpha_v else '❌ Non',
            })
    return pd.DataFrame(rows)


def run_ancova_adjusted_gain(df_pca, bande_ref, alpha_v=0.05):
    """
    Calcule le gain de chaque bande vs la référence "à niveau de sol équivalent" (sol moyen de
    l'essai), via une ANCOVA avec PC1 centré : rdt ~ C(grp, Treatment(bande_ref)) * PC1_centré.

    En centrant PC1 (PC1_centré = 0 <=> sol moyen de l'essai), le coefficient de chaque bande
    dans le modèle représente directement le gain ajusté par rapport à la référence, à sol
    équivalent (moyen), avec sa propre p-value et son intervalle de confiance à 95 %. Le terme
    d'interaction bande:PC1_centré indique si ce gain varie selon le niveau de sol (produit
    plus ou moins efficace selon la zone) plutôt que d'être constant partout.
    """
    if not HAS_STATSMODELS:
        return None
    df_m = df_pca.copy()
    if df_m['grp'].nunique() < 2 or df_m['PC1'].nunique() < 3:
        return None
    df_m['PC1_centre'] = df_m['PC1'] - df_m['PC1'].mean()
    try:
        formula = f"rdt ~ C(grp, Treatment(reference='{bande_ref}')) * PC1_centre"
        model = smf.ols(formula, data=df_m).fit()
    except Exception:
        return None

    rows = []
    for name, coef in model.params.items():
        if name.startswith("C(grp") and ":" not in name:
            bande_name = name.rsplit("[T.", 1)[-1].rstrip("]")
            pval = float(model.pvalues[name])
            ci_low, ci_high = model.conf_int().loc[name]
            inter_name = f"{name}:PC1_centre"
            p_inter = float(model.pvalues[inter_name]) if inter_name in model.pvalues else np.nan
            rows.append({
                'bande': bande_name, 'gain_ajuste': float(coef), 'p': pval,
                'ci_low': float(ci_low), 'ci_high': float(ci_high),
                'p_interaction': p_inter,
                'sig': pval < alpha_v,
                'sig_interaction': (p_inter < alpha_v) if pd.notna(p_inter) else False,
            })
    return {'model': model, 'table': pd.DataFrame(rows), 'formula': formula}


# ══════════════════════════════════════════════════════════════════════════
# 4. FONCTIONS MÉTÉO (Open-Meteo — gratuit, sans clé API)
# ══════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def fetch_weather(lat, lon, start, end):
    today = date.today()
    url_parts = []

    if start < today:
        archive_end = min(end, today - timedelta(days=1))
        if start <= archive_end:
            url_parts.append(
                "https://archive-api.open-meteo.com/v1/archive"
                f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={archive_end}"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
            )
    if end >= today:
        forecast_start = max(start, today)
        if forecast_start <= end:
            url_parts.append(
                "https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}&start_date={forecast_start}&end_date={end}"
                "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&timezone=auto"
            )

    frames = []
    for url in url_parts:
        try:
            r = requests.get(url, timeout=20)
            r.raise_for_status()
            d = r.json().get("daily", {})
            if d:
                frames.append(pd.DataFrame(d))
        except Exception:
            continue

    if not frames:
        return None
    df_w = pd.concat(frames, ignore_index=True).drop_duplicates(subset="time")
    df_w["time"] = pd.to_datetime(df_w["time"])
    return df_w.sort_values("time").reset_index(drop=True)


def compute_stress(df_w, params, jours_secheresse=7):
    df_w = df_w.copy()
    df_w["stress_chaleur"] = df_w["temperature_2m_max"] >= params["t_echaudage"]
    df_w["stress_critique"] = df_w["temperature_2m_max"] >= params["t_critique"]
    df_w["stress_gel"] = df_w["temperature_2m_min"] <= params["t_gel"]
    df_w["jour_sec"] = df_w["precipitation_sum"] < params["precip_min_jour"]
    df_w["run_id"] = (df_w["jour_sec"] != df_w["jour_sec"].shift()).cumsum()
    run_len = df_w.groupby("run_id")["jour_sec"].transform("size")
    df_w["stress_secheresse"] = df_w["jour_sec"] & (run_len >= jours_secheresse)
    return df_w


# ══════════════════════════════════════════════════════════════════════════
# 5. LECTURE FICHIER (pyshp — sans GDAL)
# ══════════════════════════════════════════════════════════════════════════
if not uploaded_file:
    st.markdown("""
    <div class="hero">
        <h1>🌱 Précis'Up</h1>
        <p>Analysez vos essais terrain en quelques secondes — comparaison statistique multi-bandes, ANOVA spatiale, ACP sol-produit et météo de la parcelle.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("👈 Importez votre fichier QGIS (.zip) dans la barre latérale pour démarrer l'analyse.")
    st.markdown("""
    **Colonnes attendues dans le fichier .shp / attributs QGIS :**

    | Colonne | Obligatoire | Description |
    |---------|-------------|-------------|
    | `bande` | ✅ | Identifiant de bande (ex: A, B, C, Témoin, Produit 1…) |
    | `rdt` | ✅ | Rendement (qtx/ha ou t/ha) |
    | `potentiel` | ⚙️ recommandé | Zone de potentiel sol (ex: Faible, Moyen, Fort) |
    | *variables de sol* | ⚙️ recommandé pour l'ACP | pH, argile, matière organique, CEC, altitude, NDVI, réserve utile… (toute colonne numérique supplémentaire) |

    **Ce que fait l'application :**
    - 📊 Sélectionnez **autant de bandes que vous voulez** et choisissez laquelle sert de **référence (témoin)**
    - Chaque bande est comparée à la référence avec le **test choisi automatiquement** selon vos données
    - 🔬 Comparaison **croisée de toutes les bandes entre elles**, pas seulement vs la référence
    - 📐 ANOVA spatiale Bande × Zone de potentiel (si au moins 2 zones disponibles)
    - 🧬 **ACP** (analyse en composantes principales) pour synthétiser les variables de sol et voir si le rendement/l'effet des bandes suit un gradient de sol
    - 🌦️ Analyse météo semis → récolte avec détection de stress thermique/hydrique
    """)
    st.stop()

try:
    clear_temp()
    with zipfile.ZipFile(io.BytesIO(uploaded_file.read())) as z:
        z.extractall("temp")

    shp_files = []
    for root, _, files in os.walk("temp"):
        shp_files += [os.path.join(root, f) for f in files if f.lower().endswith('.shp')]

    if not shp_files:
        st.error("❌ Aucun fichier .shp trouvé dans le zip.")
        st.stop()

    sf = pyshp.Reader(shp_files[0])
    field_names = [f[0] for f in sf.fields[1:]]  # on ignore le champ DeletionFlag
    records = [list(r) for r in sf.records()]
    df = pd.DataFrame(records, columns=field_names)
    df.columns = df.columns.str.lower().str.strip()

    missing = [c for c in ['bande', 'rdt'] if c not in df.columns]
    if missing:
        st.error(f"❌ Colonnes manquantes : {missing}. Colonnes disponibles : {list(df.columns)}")
        st.stop()

    df['rdt'] = pd.to_numeric(df['rdt'].astype(str).str.replace(',', '.'), errors='coerce')
    df['bande'] = df['bande'].astype(str).str.strip()
    df = df.dropna(subset=['rdt'])

    if df.empty:
        st.error("❌ Aucune valeur de rendement exploitable après nettoyage de la colonne 'rdt'.")
        st.stop()

except Exception as e:
    st.error(f"❌ Erreur lecture fichier : {e}")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# 5-bis. SÉLECTION DES BANDES À COMPARER (paramétrable) + VARIABLES DE SOL
# ══════════════════════════════════════════════════════════════════════════
bandes_dispo = sorted(df['bande'].unique().tolist())
soil_vars_candidates = detect_numeric_soil_columns(df)

with st.sidebar:
    with st.expander("🎚️ SÉLECTION DES BANDES", expanded=True):
        st.caption("Choisissez les bandes à inclure, puis celle qui sert de référence (témoin). Vous pouvez en sélectionner autant que vous voulez.")
        bandes_selectionnees = st.multiselect(
            "Bandes à inclure dans l'analyse",
            bandes_dispo, default=bandes_dispo
        )
        if len(bandes_selectionnees) >= 2:
            bande_ref = st.selectbox(
                "🔴 Bande de référence (Témoin)",
                bandes_selectionnees,
                index=0
            )
        else:
            bande_ref = bandes_selectionnees[0] if bandes_selectionnees else None
        st.caption("Toutes les autres bandes sélectionnées seront comparées à la référence, et entre elles.")

    with st.expander("🧬 VARIABLES SOL POUR L'ACP", expanded=True):
        if soil_vars_candidates:
            st.caption("Variables numériques détectées dans votre fichier — décochez celles à exclure de l'ACP.")
            soil_vars_selected = st.multiselect(
                "Variables de sol à inclure dans l'ACP",
                soil_vars_candidates, default=soil_vars_candidates
            )
        else:
            soil_vars_selected = []
            st.caption("Aucune variable de sol numérique détectée en dehors de 'potentiel'. "
                       "Ajoutez par exemple pH, argile, matière organique, CEC, altitude ou NDVI "
                       "à vos attributs QGIS pour activer l'ACP.")

if len(bandes_selectionnees) < 2:
    st.warning("⚠️ Sélectionnez au moins **2 bandes** dans la barre latérale (dont la bande de référence) pour lancer une comparaison.")
    st.stop()

df_travail = df[df['bande'].isin(bandes_selectionnees)].copy()
df_travail['grp'] = df_travail['bande']  # le nom réel de la bande sert de groupe

n_initial = len(df_travail)
if clean_iqr:
    clean_list = []
    for g in bandes_selectionnees:
        sub = df_travail[df_travail['grp'] == g]
        if not sub.empty:
            q1, q3 = sub['rdt'].quantile([0.25, 0.75])
            iqr = q3 - q1
            sub = sub[(sub['rdt'] >= q1 - 1.2 * iqr) & (sub['rdt'] <= q3 + 1.2 * iqr)]
            clean_list.append(sub)
    df_final = pd.concat(clean_list) if clean_list else df_travail.copy()
else:
    df_final = df_travail.copy()

n_removed = n_initial - len(df_final)

# ── Palette de couleurs : référence en rouge, autres bandes en verts ──────
autres_bandes = [b for b in bandes_selectionnees if b != bande_ref]
color_map = {bande_ref: COULEUR_REFERENCE}
for i, b in enumerate(autres_bandes):
    color_map[b] = PALETTE_BANDES[i % len(PALETTE_BANDES)]

data_by_band = {b: df_final[df_final['grp'] == b]['rdt'].dropna() for b in bandes_selectionnees}
n_by_band = {b: len(data_by_band[b]) for b in bandes_selectionnees}

if n_by_band.get(bande_ref, 0) <= 3:
    st.error(f"❌ Données insuffisantes sur la bande de référence '{bande_ref}' (< 4 observations).")
    st.stop()

data_ref = data_by_band[bande_ref]
bandes_valides = [b for b in autres_bandes if n_by_band[b] > 3]
bandes_ecartees = [b for b in autres_bandes if n_by_band[b] <= 3]

if not bandes_valides:
    st.error("❌ Aucune bande comparable (moins de 4 observations) en dehors de la référence. Vérifiez votre fichier.")
    st.stop()

resultats_bandes = []
for b in bandes_valides:
    data_b = data_by_band[b]
    res = run_main_test(data_b, data_ref, alpha_v=alpha_v)
    gain = data_b.mean() - data_ref.mean()
    gain_pct = (gain / data_ref.mean() * 100) if data_ref.mean() != 0 else np.nan
    marge_b = ((gain / 10) * prix_vente) - cout_prod
    resultats_bandes.append({
        'bande': b, 'n': n_by_band[b], 'moyenne': data_b.mean(), 'ecart_type': data_b.std(),
        'gain': gain, 'gain_pct': gain_pct, 'marge': marge_b,
        'test': res['name'], 'p': res['p'], 'sig': res['p'] < alpha_v,
        'd': res['d'], 'd_label': res['label'], 'small_sample': res['small_sample'],
        'shapiro_a': res['shapiro_a'], 'shapiro_b': res['shapiro_b'], 'levene_p': res['levene_p'],
    })

# ── Test global multi-bandes (si ≥ 3 bandes au total) ─────────────────────
h_stat, p_global = None, None
if len(bandes_valides) >= 2:
    try:
        groupes = [data_ref] + [data_by_band[b] for b in bandes_valides]
        h_stat, p_global = stats.kruskal(*groupes)
    except Exception:
        h_stat, p_global = None, None

sig_bands = [r for r in resultats_bandes if r['sig']]
best_band = max(sig_bands, key=lambda r: r['gain']) if sig_bands else (
    max(resultats_bandes, key=lambda r: r['gain']) if resultats_bandes else None
)

# ══════════════════════════════════════════════════════════════════════════
# 6. EN-TÊTE & KPIs
# ══════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <h1>🌱 Précis'Up</h1>
    <p>Culture analysée : <b>{culture}</b> &nbsp;·&nbsp; {len(bandes_selectionnees)} bande(s) comparée(s)
    &nbsp;·&nbsp; Référence : <b>{bande_ref}</b></p>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Bandes comparées", f"{len(bandes_selectionnees)}")
k2.metric("Bande de référence", bande_ref)
k3.metric("Meilleure bande", best_band['bande'] if best_band else "—")
k4.metric("Gain (meilleure bande)", f"{best_band['gain']:+.2f} qtx" if best_band else "—")
k5.metric("Bandes significatives", f"{len(sig_bands)}/{len(resultats_bandes)}")

if n_removed > 0:
    st.caption(f"⚠️ {n_removed} points supprimés par nettoyage IQR sur {n_initial} observations.")
if bandes_ecartees:
    st.caption(f"⚠️ Bande(s) écartée(s) des tests statistiques (< 4 obs.) : {', '.join(bandes_ecartees)}.")

if best_band is not None:
    sig = best_band['sig']
    verdict_html = f"""<div class="{'verdict-sig' if sig else 'verdict-nosig'}">
    <strong>{'✅ Meilleure bande détectée' if sig else '❌ Aucune bande significativement meilleure'}</strong>
    — La bande <b>{best_band['bande']}</b> affiche {'un gain' if best_band['gain'] >= 0 else 'un écart'} de
    <b>{best_band['gain']:+.2f} qtx/ha</b> ({best_band['gain_pct']:+.1f} %) vs la référence '{bande_ref}',
    via {best_band['test']} · p = {format_pval(best_band['p'])} · Cohen's d = {best_band['d']:.2f} ({best_band['d_label']}).
    {("<br>Cet écart n'est probablement pas dû au hasard (confiance ≥ " + str(int((1-alpha_v)*100)) + "%).") if sig else "<br>La variabilité de la parcelle empêche de conclure à un effet réel de cette bande."}
    </div>"""
    st.markdown(verdict_html, unsafe_allow_html=True)

if p_global is not None:
    st.caption(
        f"🧮 Test global multi-bandes (Kruskal-Wallis, toutes bandes confondues) : "
        f"H = {h_stat:.2f} · p = {format_pval(p_global)} · "
        + ("différence globale significative détectée entre bandes." if p_global < alpha_v
           else "aucune différence globale significative détectée entre bandes.")
    )
st.markdown("")

# Variables partagées entre onglets
anova_table_final = None
soil_vs_product_result = None
pca_summary = None
meteo_summary = {}
bandes_niveaux_pca_result = None
ancova_adjusted_result = None
potentiel_zone_stats_result = None

# ══════════════════════════════════════════════════════════════════════════
# 7. ONGLETS
# ══════════════════════════════════════════════════════════════════════════
tab_rdt, tab_anova, tab_pca, tab_meteo = st.tabs([
    "📊 Résultats & Distribution",
    "📐 ANOVA Spatiale",
    "🧬 ACP Sol-Produit",
    "🌦️ Météo & Stress",
])

# ─────────────────────────────────────────────────────────────────────────
# TAB 1 — Distribution & comparatif multi-bandes
# ─────────────────────────────────────────────────────────────────────────
with tab_rdt:
    st.markdown(
        '<div class="vulgarisation">💡 Chaque bande sélectionnée est comparée à la bande de référence '
        '(en rouge sur les graphiques) avec le test choisi automatiquement : Mann-Whitney si l\'échantillon '
        'est petit (&lt; 8 obs.), sinon Student, Welch ou Mann-Whitney selon la normalité (Shapiro) et '
        'l\'homogénéité des variances (Levene).</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([3, 2])
    with col1:
        fig_box = px.box(
            df_final, x="grp", y="rdt", color="grp", points="all", notched=True,
            color_discrete_map=color_map,
            category_orders={"grp": [bande_ref] + autres_bandes},
            labels={"grp": "Bande", "rdt": "Rendement (qtx/ha)"},
            title="Distribution des rendements par bande"
        )
        fig_box.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a', showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with col2:
        fig_viol = px.violin(
            df_final, x="grp", y="rdt", color="grp", box=True,
            color_discrete_map=color_map,
            category_orders={"grp": [bande_ref] + autres_bandes},
            labels={"grp": "Bande", "rdt": "Rendement (qtx/ha)"},
            title="Densité de probabilité"
        )
        fig_viol.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a', showlegend=False)
        st.plotly_chart(fig_viol, use_container_width=True)

    st.subheader("📋 Tableau comparatif — chaque bande vs référence")
    df_cmp = pd.DataFrame([{
        'Bande': r['bande'], 'N': r['n'], 'Moyenne (qtx)': round(r['moyenne'], 2),
        'Gain vs réf. (qtx)': round(r['gain'], 2), 'Gain (%)': round(r['gain_pct'], 1),
        'Marge nette (€/ha)': round(r['marge'], 0), 'Test utilisé': r['test'],
        'p-value': format_pval(r['p']), 'Significatif': '✅ Oui' if r['sig'] else '❌ Non',
        "Cohen's d": round(r['d'], 2), 'Ampleur effet': r['d_label'],
    } for r in resultats_bandes])
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

    desc = df_final.groupby('grp')['rdt'].describe().round(2)
    desc.columns = ['N', 'Moyenne', 'Écart-type', 'Min', 'Q25', 'Médiane', 'Q75', 'Max']
    desc = desc.reindex([bande_ref] + autres_bandes)
    with st.expander("📊 Statistiques descriptives complètes (toutes bandes)"):
        st.dataframe(desc, use_container_width=True)

    with st.expander("🔍 Détails des tests de normalité et d'homogénéité par bande"):
        detail_rows = []
        for r in resultats_bandes:
            detail_rows.append({
                'Bande': r['bande'],
                'Échantillon réduit (<8 obs.)': 'Oui' if r['small_sample'] else 'Non',
                'Shapiro (bande)': format_pval(r['shapiro_a']),
                'Shapiro (référence)': format_pval(r['shapiro_b']),
                'Levene (variances)': format_pval(r['levene_p']),
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)

    # ── Comparaison croisée entre TOUTES les bandes (pas seulement vs réf.) ──
    all_bands_valid = [bande_ref] + bandes_valides
    if len(all_bands_valid) >= 3:
        st.divider()
        st.subheader("🔬 Comparaison croisée entre toutes les bandes")
        st.markdown(
            '<div class="vulgarisation">Ce tableau compare chaque bande à chaque autre bande sélectionnée '
            '(pas uniquement à la référence), avec le même test choisi automatiquement pour chaque paire.</div>',
            unsafe_allow_html=True
        )
        p_matrix = pd.DataFrame(index=all_bands_valid, columns=all_bands_valid, dtype=object)
        d_matrix = pd.DataFrame(index=all_bands_valid, columns=all_bands_valid, dtype=float)
        for a in all_bands_valid:
            for b in all_bands_valid:
                if a == b:
                    p_matrix.loc[a, b] = "—"
                    d_matrix.loc[a, b] = 0.0
                else:
                    res_ab = run_main_test(data_by_band[a], data_by_band[b], alpha_v=alpha_v)
                    sig_mark = " ✅" if res_ab['p'] < alpha_v else ""
                    p_matrix.loc[a, b] = f"{format_pval(res_ab['p'])}{sig_mark}"
                    d_matrix.loc[a, b] = res_ab['d']

        c_p, c_d = st.columns(2)
        with c_p:
            st.caption("p-value de chaque paire (✅ = écart significatif)")
            st.dataframe(p_matrix, use_container_width=True)
        with c_d:
            st.caption("Ampleur de l'écart (Cohen's d)")
            fig_heat = px.imshow(
                d_matrix.astype(float), text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-2, zmax=2, title="Cohen's d entre chaque paire de bandes"
            )
            fig_heat.update_layout(plot_bgcolor='white', paper_bgcolor='white')
            st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Lecture : une valeur positive signifie que la bande en ligne rend mieux que la bande en colonne.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2 — ANOVA
# ─────────────────────────────────────────────────────────────────────────
with tab_anova:
    st.markdown("""
    <div class="vulgarisation">
    📐 L'ANOVA sépare ce qui revient à la qualité du sol de ce qui revient à l'effet réel du traitement
    (toutes les bandes sélectionnées, pas seulement la référence), et révèle si les bandes se comportent
    différemment selon la zone de potentiel. Elle nécessite <b>au moins 2 zones de potentiel</b> distinctes.
    </div>
    """, unsafe_allow_html=True)

    if not run_anova:
        st.info("ANOVA désactivée dans la barre latérale.")
    elif not HAS_STATSMODELS:
        st.warning("statsmodels requis : ajoutez-le à votre requirements.txt")
    else:
        anova_table, anova_title, anova_model, has_pot = run_anova_analysis(df_final, alpha_v)
        if anova_table is None:
            st.warning(f"ℹ️ {anova_title}")
        else:
            anova_table_final = anova_table
            st.subheader(anova_title)
            at = anova_table.copy()
            at.columns = [c.replace('PR(>F)', 'p-value').replace('sum_sq', 'SCE').replace('mean_sq', 'CME') for c in at.columns]
            at = at.round(4)

            def style_pval(val):
                try:
                    v = float(val)
                    if v < 0.001: return 'background-color:#d8f3dc; font-weight:bold; color:#1b4332;'
                    if v < alpha_v: return 'background-color:#ffe8cc; color:#a65c00;'
                    return ''
                except Exception:
                    return ''

            target_cols = [c for c in at.columns if 'p-value' in c.lower()]
            try:
                styled_at = at.style.applymap(style_pval, subset=target_cols) if target_cols else at.style
            except Exception:
                styled_at = at
            st.dataframe(styled_at, use_container_width=True)
            st.caption("ℹ️ Une valeur affichée à 0.0000 signifie p < 0.0001 (extrêmement significatif), jamais exactement zéro.")

            if anova_model is not None:
                col1, col2, col3 = st.columns(3)
                col1.metric("R² (variance expliquée)", f"{anova_model.rsquared:.1%}")
                col2.metric("R² ajusté", f"{anova_model.rsquared_adj:.3f}")
                col3.metric("F-Value globale", f"{anova_model.fvalue:.2f}")

            st.subheader("📊 Rendement par Zone × Bande")
            pivot = df_final.groupby(['potentiel', 'grp'])['rdt'].agg(['mean', 'std', 'count']).round(2)
            pivot.columns = ['Rendement Moyen', 'Écart-Type', 'Nombre de points']
            st.dataframe(pivot, use_container_width=True)

            fig_inter = px.box(
                df_final, x="potentiel", y="rdt", color="grp",
                color_discrete_map=color_map,
                category_orders={"grp": [bande_ref] + autres_bandes},
                title="Les bandes fonctionnent-elles mieux sur certaines zones de sol ?"
            )
            fig_inter.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
            st.plotly_chart(fig_inter, use_container_width=True)

            # ── 🎯 LA BANDE EST-ELLE MEILLEURE, À POTENTIEL DE SOL ÉGAL ? ─
            st.divider()
            st.subheader("🎯 La bande est-elle meilleure, à potentiel de sol égal ?")
            st.markdown("""
            <div class="vulgarisation">
            Le boxplot ci-dessus montre les distributions, mais ne dit pas si l'écart est réel ou dû au
            hasard. Ce tableau compare statistiquement chaque bande à la référence <b>dans chaque zone
            de potentiel</b> (colonne 'potentiel' de votre fichier), avec le test le plus adapté
            (Student/Welch/Mann-Whitney selon l'échantillon). Cela répond à : « à sol comparable, la
            bande gagne-t-elle vraiment, ou est-ce qu'elle est simplement tombée sur une meilleure zone ? ».
            </div>
            """, unsafe_allow_html=True)

            zones_potentiel_ordered = order_zone_labels(df_final['potentiel'])
            tbl_potentiel = compare_bands_within_zones(df_final, 'potentiel', zones_potentiel_ordered, bande_ref, alpha_v)
            potentiel_zone_stats_result = tbl_potentiel
            if tbl_potentiel.empty:
                st.info("Pas assez d'observations par bande et par zone de potentiel (≥ 4 requises dans "
                        "chaque zone) pour calculer ce tableau. Vérifiez le nombre de points par zone.")
            else:
                st.dataframe(tbl_potentiel, use_container_width=True, hide_index=True)
                nb_sig_zone = int((tbl_potentiel['Significatif'] == '✅ Oui').sum())
                nb_calc_zone = int((tbl_potentiel['Significatif'] != '—').sum())
                st.caption(
                    f"✅ {nb_sig_zone} comparaison(s) bande-vs-référence significative(s) à potentiel de sol "
                    f"égal, sur {nb_calc_zone} comparaison(s) calculable(s) au total."
                )

            # ── 🔎 CHOIX DE LA/DES ZONE(S) + COMPARAISON CROISÉE COMPLÈTE ──
            st.divider()
            st.subheader("🔎 Analyse détaillée d'une zone de potentiel choisie")
            st.markdown("""
            <div class="vulgarisation">
            Choisissez la ou les zones de potentiel à étudier, puis une zone précise à analyser en détail :
            vous verrez les statistiques descriptives de <b>toutes les bandes, y compris le témoin</b>,
            la comparaison croisée complète entre bandes (p-value + Cohen's d de chaque paire) dans cette
            zone, et la bande qui en ressort comme la meilleure "à sol égal".
            </div>
            """, unsafe_allow_html=True)

            zones_a_inclure = st.multiselect(
                "Zones de potentiel à inclure dans cette analyse",
                zones_potentiel_ordered, default=zones_potentiel_ordered, key="zones_potentiel_multiselect"
            )

            if not zones_a_inclure:
                st.info("Sélectionnez au moins une zone de potentiel ci-dessus.")
            else:
                zone_choisie = st.selectbox(
                    "Zone à analyser en détail", zones_a_inclure, key="zone_choisie_detail"
                )
                sub_zone = df_final[df_final['potentiel'] == zone_choisie]
                bands_order_zone = [bande_ref] + autres_bandes
                data_by_band_zone = {
                    b: sub_zone[sub_zone['grp'] == b]['rdt'].dropna() for b in bands_order_zone
                }

                desc_zone = sub_zone.groupby('grp')['rdt'].describe().round(2)
                if not desc_zone.empty:
                    desc_zone.columns = ['N', 'Moyenne', 'Écart-type', 'Min', 'Q25', 'Médiane', 'Q75', 'Max']
                    desc_zone = desc_zone.reindex([b for b in bands_order_zone if b in desc_zone.index])
                    st.markdown(f"**Statistiques descriptives — zone '{zone_choisie}' (toutes bandes, y compris témoin)**")
                    st.dataframe(desc_zone, use_container_width=True)
                else:
                    st.info(f"Aucune donnée dans la zone '{zone_choisie}'.")

                bands_valid_zone = [b for b in bands_order_zone if len(data_by_band_zone.get(b, [])) > 3]
                if len(bands_valid_zone) < 2:
                    st.info("Pas assez de bandes avec suffisamment d'observations (≥ 4) dans cette zone "
                            "pour une comparaison croisée.")
                else:
                    p_mat_zone, d_mat_zone, _, _ = compute_pairwise_matrices(data_by_band_zone, bands_valid_zone, alpha_v)
                    c_pz, c_dz = st.columns(2)
                    with c_pz:
                        st.caption(f"p-value de chaque paire — zone '{zone_choisie}' (✅ = écart significatif)")
                        st.dataframe(p_mat_zone, use_container_width=True)
                    with c_dz:
                        st.caption("Ampleur de l'écart (Cohen's d)")
                        fig_heat_zone = px.imshow(
                            d_mat_zone.astype(float), text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-2, zmax=2, title=f"Cohen's d entre bandes — zone '{zone_choisie}'"
                        )
                        fig_heat_zone.update_layout(plot_bgcolor='white', paper_bgcolor='white')
                        st.plotly_chart(fig_heat_zone, use_container_width=True)

                    best_zone, _ = best_band_summary(data_by_band_zone, bande_ref, alpha_v)
                    if best_zone is not None:
                        sig_z = best_zone['sig']
                        st.markdown(
                            f"""<div class="{'verdict-sig' if sig_z else 'verdict-nosig'}">
                            <strong>{'✅ Meilleure bande dans cette zone' if sig_z else '❌ Aucune bande significativement meilleure dans cette zone'}</strong>
                            — Dans la zone <b>'{zone_choisie}'</b>, la bande <b>{best_zone['bande']}</b> affiche
                            {'un gain' if best_zone['gain'] >= 0 else 'un écart'} de <b>{best_zone['gain']:+.2f} qtx/ha</b>
                            vs '{bande_ref}' — {best_zone['test']}, p = {format_pval(best_zone['p'])},
                            Cohen's d = {best_zone['d']:.2f} ({best_zone['d_label']}).
                            </div>""",
                            unsafe_allow_html=True
                        )
                    else:
                        st.info(f"Pas assez de données de référence ('{bande_ref}') dans la zone '{zone_choisie}' "
                                "pour désigner une meilleure bande.")

            # ── 📊 UNE MÊME BANDE (Y COMPRIS LE TÉMOIN) SUR DEUX POTENTIELS ──
            st.divider()
            st.subheader("📊 Une même bande, y compris le témoin, sur deux potentiels différents")
            st.markdown("""
            <div class="vulgarisation">
            Changement de perspective : on fixe une <b>bande</b> (le témoin ou n'importe quelle autre) et
            on regarde si <b>son propre rendement</b> varie significativement d'une zone de potentiel à
            l'autre — utile pour savoir si le sol influence fortement cette bande en particulier,
            indépendamment de l'effet du produit.
            </div>
            """, unsafe_allow_html=True)

            bande_a_regarder = st.selectbox(
                "Bande à analyser (y compris le témoin)",
                [bande_ref] + autres_bandes, index=0, key="bande_zones_selectbox"
            )
            zones_pour_bande = zones_a_inclure if zones_a_inclure else zones_potentiel_ordered
            if len(zones_pour_bande) < 2:
                st.info("Sélectionnez au moins 2 zones de potentiel plus haut pour comparer une bande entre zones.")
            else:
                tbl_zones_bande = compare_zones_for_band(df_final, 'potentiel', bande_a_regarder, zones_pour_bande, alpha_v)
                if tbl_zones_bande.empty:
                    st.info(f"Pas assez d'observations pour la bande '{bande_a_regarder}' dans au moins 2 zones.")
                else:
                    st.dataframe(tbl_zones_bande, use_container_width=True, hide_index=True)
                    nb_sig_bz = int((tbl_zones_bande['Significatif'] == '✅ Oui').sum())
                    st.caption(
                        f"✅ {nb_sig_bz} paire(s) de zones où la bande '{bande_a_regarder}' rend "
                        f"significativement différemment, sur {len(tbl_zones_bande)} paire(s) testée(s)."
                    )

            # ── 🎯 QUI A LE PLUS D'EFFET : LE SOL OU LE TRAITEMENT ? ─────
            st.divider()
            st.subheader("🎯 Qui a le plus d'effet : le sol ou le traitement (bandes) ?")
            st.markdown("""
            <div class="vulgarisation">
            Cette analyse compare la part de variance du rendement expliquée par chaque facteur
            (<b>eta²</b>, calculé à partir du tableau ANOVA ci-dessus) et sa significativité,
            pour déterminer lequel — du sol ou des bandes — pèse le plus sur le résultat.
            </div>
            """, unsafe_allow_html=True)

            soil_vs_product_result = compare_soil_vs_product(anova_table, alpha_v)
            if soil_vs_product_result is None:
                st.info("Impossible de calculer la comparaison Sol vs Traitement sur ce jeu de données.")
            else:
                e1, e2, e3 = st.columns(3)
                e1.metric("Eta² — Traitement (bandes)", f"{soil_vs_product_result['eta2_grp']:.1%}",
                          "Significatif" if soil_vs_product_result['sig_grp'] else "Non significatif")
                e2.metric("Eta² — Sol", f"{soil_vs_product_result['eta2_pot']:.1%}",
                          "Significatif" if soil_vs_product_result['sig_pot'] else "Non significatif")
                e3.metric("Eta² — Interaction", f"{soil_vs_product_result['eta2_inter']:.1%}",
                          "Significatif" if soil_vs_product_result['sig_inter'] else "Non significatif")

                dominant = soil_vs_product_result['dominant']
                emoji_dominant = {"produit": "🧪", "sol": "🟤", "égalité": "⚖️", "aucun": "❓"}.get(dominant, "🎯")
                st.markdown(
                    f'<div class="verdict-dominant">{emoji_dominant} <strong>{soil_vs_product_result["verdict"]}</strong></div>',
                    unsafe_allow_html=True
                )

# ─────────────────────────────────────────────────────────────────────────
# TAB 3 — ACP Sol-Produit
# ─────────────────────────────────────────────────────────────────────────
with tab_pca:
    st.markdown("""
    <div class="vulgarisation">
    🧬 L'ACP (Analyse en Composantes Principales) résume plusieurs variables de sol en quelques
    axes synthétiques, pour identifier quel(s) facteur(s) de sol structurent le plus vos parcelles
    et vérifier si le rendement — et l'effet des bandes/produits — suit ce gradient de sol.
    </div>
    """, unsafe_allow_html=True)

    if len(soil_vars_selected) == 0:
        st.info(
            "Aucune variable de sol numérique disponible (en dehors de 'potentiel') : l'ACP nécessite "
            "au moins 2 variables numériques de sol. Ajoutez par exemple pH, argile, matière organique, "
            "CEC, profondeur, réserve utile, altitude ou NDVI à vos attributs QGIS pour l'activer."
        )
    elif len(soil_vars_selected) == 1:
        var = soil_vars_selected[0]
        st.info(f"Une seule variable de sol détectée ('{var}') : l'ACP nécessite au moins 2 variables. "
                "Voici en attendant la relation directe entre cette variable et le rendement, par bande.")
        trend = "ols" if HAS_STATSMODELS else None
        fig_simple = px.scatter(
            df_final, x=var, y="rdt", color="grp", trendline=trend,
            color_discrete_map=color_map,
            category_orders={"grp": [bande_ref] + autres_bandes},
            title=f"Rendement en fonction de '{var}', par bande"
        )
        fig_simple.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
        st.plotly_chart(fig_simple, use_container_width=True)
    else:
        pca_result = run_pca(df_final, soil_vars_selected)
        if pca_result is None:
            st.warning("Données insuffisantes pour calculer l'ACP (trop de valeurs manquantes ou pas assez de points).")
        else:
            scores = pca_result['scores']
            loadings = pca_result['loadings']
            var_exp = pca_result['explained_var']
            has_pc2 = 'PC2' in scores.columns

            c1, c2, c3 = st.columns(3)
            c1.metric("Variance expliquée — PC1", f"{var_exp[0]:.1%}")
            c2.metric("Variance expliquée — PC2", f"{var_exp[1]:.1%}" if len(var_exp) > 1 else "—")
            c3.metric("Cumul PC1+PC2", f"{sum(var_exp[:2]):.1%}" if len(var_exp) > 1 else "—")

            df_pca = df_final.loc[pca_result['index']].copy()
            df_pca['PC1'] = scores['PC1'].values
            if has_pc2:
                df_pca['PC2'] = scores['PC2'].values
            else:
                df_pca['PC2'] = 0.0

            # ── Biplot : individus + vecteurs des variables de sol ──────
            fig_biplot = px.scatter(
                df_pca, x="PC1", y="PC2",
                color="grp", size="rdt", color_discrete_map=color_map,
                category_orders={"grp": [bande_ref] + autres_bandes},
                labels={"PC1": f"PC1 ({var_exp[0]:.1%})",
                        "PC2": f"PC2 ({var_exp[1]:.1%})" if has_pc2 else "PC2"},
                title="ACP — parcelles colorées par bande, taille = rendement"
            )
            scale = float(df_pca['PC1'].abs().max()) * 0.9 if len(df_pca) else 1.0
            scale = scale if scale > 0 else 1.0
            for var in loadings.index:
                lx = loadings.loc[var, 'PC1'] * scale
                ly = (loadings.loc[var, 'PC2'] * scale) if has_pc2 else 0
                fig_biplot.add_shape(type="line", x0=0, y0=0, x1=lx, y1=ly,
                                      line=dict(color="#7b241c", width=1.5))
                fig_biplot.add_annotation(x=lx, y=ly, text=var, showarrow=False,
                                           font=dict(color="#7b241c", size=11))
            fig_biplot.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
            st.plotly_chart(fig_biplot, use_container_width=True)
            st.caption("Les flèches rouges montrent la direction et le poids de chaque variable de sol dans l'ACP.")

            with st.expander("📋 Contribution des variables de sol à chaque axe (loadings)"):
                st.dataframe(loadings.round(3), use_container_width=True)

            # ── Corrélation des axes avec le rendement ───────────────────
            corr_pc1 = df_pca['PC1'].corr(df_pca['rdt'])
            corr_pc2 = df_pca['PC2'].corr(df_pca['rdt']) if has_pc2 else np.nan
            top_var_pc1 = loadings['PC1'].abs().sort_values(ascending=False).index[0]

            verdict_pca = (
                f"L'axe PC1 (dominé par **{top_var_pc1}**) résume {var_exp[0]:.1%} de la variabilité "
                f"de sol mesurée, et sa corrélation avec le rendement est de **{corr_pc1:.2f}**"
                + (" : le rendement suit donc en bonne partie ce gradient de sol."
                   if abs(corr_pc1) > 0.3 else
                   " : la relation avec le rendement reste faible sur cet axe.")
            )
            st.markdown(f'<div class="verdict-dominant">🧬 {verdict_pca}</div>', unsafe_allow_html=True)

            # ── Position moyenne de chaque bande sur les axes de sol ─────
            cols_pos = ['PC1', 'PC2'] if has_pc2 else ['PC1']
            pos_bande = df_pca.groupby('grp')[cols_pos].mean().round(2)
            pos_bande = pos_bande.reindex([bande_ref] + [b for b in autres_bandes if b in pos_bande.index])
            st.subheader("📍 Position moyenne de chaque bande sur les axes de sol")
            st.dataframe(pos_bande, use_container_width=True)
            st.caption(
                "Si une bande se situe nettement à part sur PC1/PC2, elle a été testée sur un type de "
                "sol différent des autres — à garder en tête avant de généraliser son résultat à toute la parcelle."
            )

            pca_summary = {
                'var_exp_pc1': float(var_exp[0]),
                'corr_pc1_rdt': float(corr_pc1) if pd.notna(corr_pc1) else None,
                'top_var_pc1': top_var_pc1,
                'verdict': verdict_pca,
            }

            # ── 🎯 SOL (gradient PC1) vs PRODUIT (bandes) : qui pèse le plus ? ──
            st.divider()
            st.subheader("🎯 Ici, qui a le plus d'effet : le sol (gradient PC1) ou le produit (bandes) ?")
            st.markdown("""
            <div class="vulgarisation">
            Même question que dans l'onglet ANOVA, mais avec le sol représenté par l'axe continu PC1
            de l'ACP (au lieu de la seule zone 'potentiel') : une ANCOVA (bande × PC1) sépare la part
            de variance du rendement due au <b>traitement</b>, celle due au <b>gradient de sol</b>, et
            leur <b>interaction</b> (le produit agit-il différemment selon le sol ?).
            </div>
            """, unsafe_allow_html=True)

            if not HAS_STATSMODELS:
                st.warning("statsmodels requis pour ce calcul : ajoutez-le à votre requirements.txt")
            else:
                grp_vs_pca = compare_grp_vs_pca_axis(df_pca, alpha_v)
                if grp_vs_pca is None:
                    st.info("Impossible de calculer cette comparaison sur ce jeu de données "
                            "(pas assez de variation sur PC1 ou sur les bandes).")
                else:
                    f1, f2, f3 = st.columns(3)
                    f1.metric("Eta² — Produit (bandes)", f"{grp_vs_pca['eta2_grp']:.1%}",
                              "Significatif" if grp_vs_pca['sig_grp'] else "Non significatif")
                    f2.metric("Eta² — Sol (PC1)", f"{grp_vs_pca['eta2_pc1']:.1%}",
                              "Significatif" if grp_vs_pca['sig_pc1'] else "Non significatif")
                    f3.metric("Eta² — Interaction", f"{grp_vs_pca['eta2_inter']:.1%}",
                              "Significatif" if grp_vs_pca['sig_inter'] else "Non significatif")

                    dominant_pca = grp_vs_pca['dominant']
                    emoji_dom = {"produit": "🧪", "sol": "🟤", "égalité": "⚖️", "aucun": "❓"}.get(dominant_pca, "🎯")
                    st.markdown(
                        f'<div class="verdict-dominant">{emoji_dom} <strong>{grp_vs_pca["verdict"]}</strong></div>',
                        unsafe_allow_html=True
                    )
                    pca_summary['soil_vs_product_pca'] = grp_vs_pca

                # ── 🎯 COMPARER LES BANDES "À NIVEAU DE SOL ÉQUIVALENT" ──
                st.divider()
                st.subheader("⚖️ Les bandes comparées à niveau de sol équivalent")
                st.markdown("""
                <div class="vulgarisation">
                Ici, on ne compare plus les bandes globalement, mais <b>à sol comparable</b> : la parcelle
                est découpée en niveaux de sol homogènes à partir de l'axe PC1 (comme des zones de
                potentiel, mais calculées automatiquement à partir de vos variables de sol). Cela permet
                de répondre à « est-ce que la bande est vraiment meilleure, ou est-ce simplement qu'elle
                est tombée sur un meilleur sol ? ».
                </div>
                """, unsafe_allow_html=True)

                n_niveaux = st.radio(
                    "Nombre de niveaux de sol (découpage de PC1)",
                    [2, 3, 4], index=1, horizontal=True, key="n_niveaux_pca"
                )
                df_pca_binned, ordered_levels = bin_pca_levels(df_pca, n_levels=n_niveaux)

                fig_niveaux = px.box(
                    df_pca_binned, x="niveau_pc1", y="rdt", color="grp",
                    color_discrete_map=color_map,
                    category_orders={"grp": [bande_ref] + autres_bandes, "niveau_pc1": ordered_levels},
                    labels={"niveau_pc1": "Niveau de sol (PC1)", "rdt": "Rendement (qtx/ha)"},
                    title="Rendement par bande, à niveau de sol comparable"
                )
                fig_niveaux.update_layout(plot_bgcolor='white', paper_bgcolor='white', font_color='#33433a')
                st.plotly_chart(fig_niveaux, use_container_width=True)

                tbl_niveaux = compare_bands_within_pca_levels(df_pca_binned, ordered_levels, bande_ref, alpha_v)
                bandes_niveaux_pca_result = tbl_niveaux
                if tbl_niveaux.empty:
                    st.info("Pas assez d'observations par bande et par niveau de sol (≥ 4 requises) pour ce découpage. "
                            "Essayez avec moins de niveaux.")
                else:
                    st.dataframe(tbl_niveaux, use_container_width=True, hide_index=True)
                    nb_sig_niveaux = int((tbl_niveaux['Significatif'] == '✅ Oui').sum())
                    st.caption(
                        f"✅ {nb_sig_niveaux} comparaison(s) bande-vs-référence significative(s) à sol comparable, "
                        f"sur {len(tbl_niveaux)} comparaison(s) calculée(s)."
                    )

                # ── 🧮 GAIN AJUSTÉ PAR ANCOVA (à sol moyen équivalent) ────
                st.divider()
                st.subheader("🧮 Gain ajusté de chaque bande, à sol moyen équivalent (ANCOVA)")
                st.markdown("""
                <div class="vulgarisation">
                Plutôt que de découper le sol en niveaux, cette méthode utilise directement le gradient
                PC1 dans un modèle statistique (ANCOVA) pour calculer le <b>gain de chaque bande vs la
                référence, ramené à un niveau de sol équivalent (le sol moyen de l'essai)</b> — avec sa
                p-value et son intervalle de confiance à 95 %. Elle indique aussi si ce gain est stable
                ou s'il varie selon le sol (interaction bande × sol).
                </div>
                """, unsafe_allow_html=True)

                ancova_res = run_ancova_adjusted_gain(df_pca, bande_ref, alpha_v)
                ancova_adjusted_result = ancova_res
                if ancova_res is None or ancova_res['table'].empty:
                    st.info("Impossible de calculer le gain ajusté sur ce jeu de données "
                            "(pas assez de variation sur PC1 ou sur les bandes).")
                else:
                    tbl = ancova_res['table'].copy()
                    tbl_display = pd.DataFrame({
                        'Bande': tbl['bande'],
                        'Gain ajusté vs réf. (qtx)': tbl['gain_ajuste'].round(2),
                        'IC 95 %': tbl.apply(lambda r: f"[{r['ci_low']:.2f} ; {r['ci_high']:.2f}]", axis=1),
                        'p-value': tbl['p'].apply(format_pval),
                        'Significatif (à sol égal)': tbl['sig'].map({True: '✅ Oui', False: '❌ Non'}),
                        'Gain varie selon le sol ? (p interaction)': tbl['p_interaction'].apply(format_pval),
                        'Interaction significative': tbl['sig_interaction'].map({True: '⚠️ Oui', False: 'Non'}),
                    })
                    st.dataframe(tbl_display, use_container_width=True, hide_index=True)

                    best_adj = tbl.loc[tbl['gain_ajuste'].idxmax()] if not tbl.empty else None
                    if best_adj is not None:
                        sig_adj = bool(best_adj['sig'])
                        verdict_adj_html = f"""<div class="{'verdict-sig' if sig_adj else 'verdict-nosig'}">
                        <strong>{'✅ À sol équivalent, une bande se détache' if sig_adj else '❌ À sol équivalent, aucune bande ne se détache clairement'}</strong>
                        — La bande <b>{best_adj['bande']}</b> affiche un gain ajusté de
                        <b>{best_adj['gain_ajuste']:+.2f} qtx/ha</b> vs '{bande_ref}' au niveau de sol moyen de l'essai
                        (IC 95 % [{best_adj['ci_low']:.2f} ; {best_adj['ci_high']:.2f}], p = {format_pval(best_adj['p'])}).
                        {("Cet écart tient même une fois le niveau de sol neutralisé." if sig_adj else
                          "Une fois le niveau de sol neutralisé, l'écart observé n'est plus démontré statistiquement.")}
                        </div>"""
                        st.markdown(verdict_adj_html, unsafe_allow_html=True)
                        if bool(best_adj['sig_interaction']):
                            st.caption(
                                f"⚠️ L'effet de la bande **{best_adj['bande']}** varie significativement selon le "
                                f"niveau de sol (interaction p = {format_pval(best_adj['p_interaction'])}) : son gain "
                                f"n'est donc pas constant partout sur la parcelle — regardez le graphique par niveau ci-dessus."
                            )

# ─────────────────────────────────────────────────────────────────────────
# TAB 4 — Météo & Stress
# ─────────────────────────────────────────────────────────────────────────
with tab_meteo:
    params = {
        "t_echaudage": t_echaudage,
        "t_critique": t_critique,
        "t_gel": t_gel,
        "precip_min_jour": precip_min_jour,
    }
    st.markdown(f"""
    <div class="vulgarisation">
    🌦️ Analyse météo de la période <b>semis → récolte</b> au point GPS indiqué dans la barre latérale
    (lat {lat_input:.4f}, lon {lon_input:.4f}). Seuils actuellement réglés :
    chaleur ≥ {t_echaudage}°C, critique ≥ {t_critique}°C, gel ≤ {t_gel}°C,
    sécheresse = {jours_secheresse} jours consécutifs sans pluie utile (&lt; {precip_min_jour} mm/j).
    Ajustez-les dans la barre latérale (🌡️ Seuils de stress).
    </div>
    """, unsafe_allow_html=True)

    if d_recolt < d_semis:
        st.error("La date de récolte doit être postérieure à la date de semis.")
    else:
        with st.spinner("Récupération des données météo…"):
            df_w = fetch_weather(lat_input, lon_input, d_semis, d_recolt)

        if df_w is None or df_w.empty:
            st.warning("Données météo indisponibles pour cette période/localisation. Vérifiez vos coordonnées GPS.")
        else:
            df_w = compute_stress(df_w, params, jours_secheresse)

            nb_chaleur = int(df_w['stress_chaleur'].sum())
            nb_critique = int(df_w['stress_critique'].sum())
            nb_gel = int(df_w['stress_gel'].sum())
            nb_secheresse = int(df_w['stress_secheresse'].sum())
            total_jours = len(df_w)

            stress_total = nb_critique > 0 or nb_secheresse > 0
            meteo_summary = {
                'nb_chaleur': nb_chaleur, 'nb_critique': nb_critique, 'nb_gel': nb_gel,
                'nb_secheresse': nb_secheresse, 'total_jours': total_jours, 'stress_total': stress_total,
            }
            html_s = f"""<div class="{'stress-high' if stress_total else 'stress-low'}">
            <strong>{'⚠️ Stress détecté pendant le cycle' if stress_total else '✅ Aucun stress majeur détecté'}</strong>
            — {nb_chaleur} jour(s) ≥ seuil d'échaudage, {nb_critique} jour(s) de chaleur critique,
            {nb_gel} jour(s) de gel, {nb_secheresse} jour(s) en séquence de sécheresse (≥ {jours_secheresse}j), sur {total_jours} jours analysés.
            </div>"""
            st.markdown(html_s, unsafe_allow_html=True)
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("🔥 Jours chaleur (échaudage)", nb_chaleur)
            m2.metric("🌡️ Jours chaleur critique", nb_critique)
            m3.metric("❄️ Jours de gel", nb_gel)
            m4.metric("🏜️ Jours en séquence sèche", nb_secheresse)

            fig_meteo = go.Figure()
            fig_meteo.add_trace(go.Bar(
                x=df_w['time'], y=df_w['precipitation_sum'], name="Précipitations (mm)",
                marker_color='#3498db', opacity=0.5, yaxis='y2'
            ))
            fig_meteo.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_max'], name="T° Max",
                line=dict(color='#e74c3c', width=2), mode='lines'
            ))
            fig_meteo.add_trace(go.Scatter(
                x=df_w['time'], y=df_w['temperature_2m_min'], name="T° Min",
                line=dict(color='#5dade2', width=2), mode='lines', fill='tonexty', fillcolor='rgba(93,173,226,0.08)'
            ))
            fig_meteo.add_hline(y=params['t_echaudage'], line_dash="dash", line_color="orange",
                          annotation_text=f"Seuil échaudage ({params['t_echaudage']}°C)")
            fig_meteo.add_hline(y=params['t_critique'], line_dash="dash", line_color="red",
                          annotation_text=f"Seuil critique ({params['t_critique']}°C)")
            fig_meteo.add_hline(y=params['t_gel'], line_dash="dash", line_color="#2980b9",
                          annotation_text=f"Seuil gel ({params['t_gel']}°C)")

            for _, row in df_w[df_w['stress_critique']].iterrows():
                fig_meteo.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="red", opacity=0.08, line_width=0)
            for _, row in df_w[df_w['stress_secheresse']].iterrows():
                fig_meteo.add_vrect(x0=row['time'] - pd.Timedelta(hours=12), x1=row['time'] + pd.Timedelta(hours=12),
                              fillcolor="#d35400", opacity=0.06, line_width=0)

            appli_ts = pd.Timestamp(d_appli)
            if df_w['time'].min() <= appli_ts <= df_w['time'].max():
                fig_meteo.add_vline(x=appli_ts, line_dash="dot", line_color="green",
                              annotation_text="Application produit", annotation_position="top")

            fig_meteo.update_layout(
                title="Évolution météo et zones de stress pendant le cycle cultural",
                xaxis_title="Date", yaxis_title="Température (°C)",
                yaxis2=dict(title="Précipitations (mm)", overlaying='y', side='right', showgrid=False),
                height=520, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified", plot_bgcolor='white', paper_bgcolor='white'
            )
            st.plotly_chart(fig_meteo, use_container_width=True)

            with st.expander("📋 Données météo journalières détaillées"):
                show_cols = ['time', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum',
                             'stress_chaleur', 'stress_critique', 'stress_gel', 'stress_secheresse']
                st.dataframe(df_w[show_cols].rename(columns={
                    'time': 'Date', 'temperature_2m_max': 'T° Max', 'temperature_2m_min': 'T° Min',
                    'precipitation_sum': 'Précip. (mm)', 'stress_chaleur': 'Stress chaleur',
                    'stress_critique': 'Stress critique', 'stress_gel': 'Gel', 'stress_secheresse': 'Sécheresse'
                }), use_container_width=True)

            with st.expander("🔍 Comment interpréter ce graphique ?"):
                st.markdown(f"""
                - Courbe **rouge** = température max du jour ; courbe **bleue** = température min.
                - Zones **rouges légères** = jours où la chaleur a dépassé votre seuil critique ({t_critique}°C).
                - Zones **orange** = séquence de sécheresse (≥ {jours_secheresse} jours sans pluie utile).
                - Ligne **verte pointillée** = date d'application produit : regardez si elle tombe juste avant
                  ou pendant une période de stress, ce qui peut influencer l'efficacité du traitement.
                """)

# ══════════════════════════════════════════════════════════════════════════
# 8. EXPORT DES RÉSULTATS (rapport texte + CSV — plus de PDF)
# ══════════════════════════════════════════════════════════════════════════
st.divider()
st.subheader("📤 Export des résultats")

report_lines = [
    f"# Rapport Précis'Up — {datetime.now().strftime('%d/%m/%Y %H:%M')}",
    f"**Culture** : {culture}  |  **Bande de référence** : {bande_ref}",
    f"**Semis** : {d_semis}  |  **Application** : {d_appli}  |  **Récolte** : {d_recolt}",
    "",
    "## Comparatif des bandes vs référence",
]
for r in resultats_bandes:
    report_lines.append(
        f"- **{r['bande']}** (N={r['n']}) : gain {r['gain']:+.2f} qtx/ha ({r['gain_pct']:+.1f}%), "
        f"marge {r['marge']:.0f} €/ha — {r['test']}, p = {format_pval(r['p'])} "
        f"({'significatif' if r['sig'] else 'non significatif'}), Cohen's d = {r['d']:.2f} ({r['d_label']})"
    )
if p_global is not None:
    report_lines += ["", f"**Test global (Kruskal-Wallis)** : H = {h_stat:.2f}, p = {format_pval(p_global)}"]
if soil_vs_product_result is not None:
    report_lines += [
        "",
        "## Sol vs Traitement (bandes) — ANOVA",
        f"- Eta² Traitement : {soil_vs_product_result['eta2_grp']:.1%} (p = {format_pval(soil_vs_product_result['p_grp'])})",
        f"- Eta² Sol : {soil_vs_product_result['eta2_pot']:.1%} (p = {format_pval(soil_vs_product_result['p_pot'])})",
        f"- {soil_vs_product_result['verdict'].replace('**', '')}",
    ]
if pca_summary is not None:
    report_lines += [
        "",
        "## ACP — variables de sol",
        f"- Variable la plus structurante sur PC1 : {pca_summary['top_var_pc1']} "
        f"({pca_summary['var_exp_pc1']:.1%} de variance expliquée)",
        f"- Corrélation PC1 / rendement : {pca_summary['corr_pc1_rdt']:.2f}" if pca_summary['corr_pc1_rdt'] is not None else "",
        f"- {pca_summary['verdict'].replace('**', '')}",
    ]
    if pca_summary.get('soil_vs_product_pca') is not None:
        svp = pca_summary['soil_vs_product_pca']
        report_lines += [
            "",
            "## Sol (PC1) vs Produit (bandes) — ANCOVA",
            f"- Eta² Produit : {svp['eta2_grp']:.1%} (p = {format_pval(svp['p_grp'])})",
            f"- Eta² Sol (PC1) : {svp['eta2_pc1']:.1%} (p = {format_pval(svp['p_pc1'])})",
            f"- {svp['verdict'].replace('**', '')}",
        ]
if potentiel_zone_stats_result is not None and not potentiel_zone_stats_result.empty:
    report_lines += ["", "## Comparaison des bandes à potentiel de sol égal (zones déclarées)"]
    for _, row in potentiel_zone_stats_result.iterrows():
        report_lines.append(
            f"- {row['Zone de potentiel']} — {row['Bande']} vs {bande_ref} : "
            f"gain {row['Gain (qtx)']} qtx, p = {row['p-value']} ({row['Significatif']})"
        )
if bandes_niveaux_pca_result is not None and not bandes_niveaux_pca_result.empty:
    report_lines += ["", "## Comparaison des bandes à niveau de sol équivalent (PC1)"]
    for _, row in bandes_niveaux_pca_result.iterrows():
        report_lines.append(
            f"- {row['Niveau de sol (PC1)']} — {row['Bande']} vs {bande_ref} : "
            f"gain {row['Gain (qtx)']} qtx, p = {row['p-value']} ({row['Significatif']})"
        )
if ancova_adjusted_result is not None and not ancova_adjusted_result['table'].empty:
    report_lines += ["", "## Gain ajusté par ANCOVA (à sol moyen équivalent)"]
    for _, row in ancova_adjusted_result['table'].iterrows():
        report_lines.append(
            f"- **{row['bande']}** vs {bande_ref} : gain ajusté {row['gain_ajuste']:+.2f} qtx/ha "
            f"[{row['ci_low']:.2f} ; {row['ci_high']:.2f}], p = {format_pval(row['p'])} "
            f"({'significatif' if row['sig'] else 'non significatif'}) — "
            f"interaction avec le sol : p = {format_pval(row['p_interaction'])} "
            f"({'significative' if row['sig_interaction'] else 'non significative'})"
        )
report_text = "\n".join([l for l in report_lines if l is not None])

exp_col1, exp_col2 = st.columns(2)
with exp_col1:
    st.download_button(
        "⬇️ Rapport texte (.md)",
        report_text.encode("utf-8"),
        file_name=f"Précis'Up_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True,
    )
with exp_col2:
    csv = df_final.to_csv(index=False).encode('utf-8')
    st.download_button(
        "⬇️ Données filtrées (.csv)",
        csv,
        file_name="precisup_donnees_filtrees.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("📋 Données filtrées (après nettoyage IQR)"):
    st.dataframe(df_final.reset_index(drop=True), use_container_width=True)

