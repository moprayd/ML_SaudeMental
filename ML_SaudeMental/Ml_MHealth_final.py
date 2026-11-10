"""
Ml_MHealth_final.py
KNN (GPU/CPU via PyTorch) para prever AMI (any mental illness) no NSDUH 2021-2024.

Uso:
    python Ml_MHealth_final.py            # sem oversampling (baseline)
    python Ml_MHealth_final.py random     # oversampling aleatorio (so no treino)
    python Ml_MHealth_final.py smote      # SMOTE (so no treino)   [pip install imbalanced-learn]

Saidas:
    - Resultado no formato "hub" (acuracia, acuracia balanceada, F1 binario, relatorio)
    - matriz_confusao.png
    - tempos_execucao.csv (acumula uma linha por execucao, para calcular a media de tempo)
"""
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    balanced_accuracy_score, f1_score, roc_auc_score, average_precision_score,
)

# Corrige acentos no terminal do Windows
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ------------------------------------------------------------------ CONFIG
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "NSDUH_2021_2024_Tab.tsv"
TIMES_CSV = BASE_DIR / "tempos_execucao.csv"
CM_PNG = BASE_DIR / "matriz_confusao.png"

TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 5
K_GRID = [15, 30, 50, 70, 100, 150, 200]
METRICS = ["euclidean", "manhattan"]
WEIGHTS = ["uniform", "distance"]

OVERSAMPLING = sys.argv[1] if len(sys.argv) > 1 else "none"
assert OVERSAMPLING in ("none", "random", "smote"), "use: none, random ou smote"

CLASS_LABELS = ["Sem indicação de transtorno mental",
                "Com indicação de transtorno (AMI)"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

categorical_features = ["AGE3", "SEXAGE", "NEWRACE2", "IRMARIT", "EDUHIGHCAT",
                        "ENRLCOLLFT2", "ENRLCOLLST2", "IRWRKSTAT", "WRKSTATWK2"]
ordinal_features = ["IREDUHIGHST2", "INCOME", "POVERTY3", "IRFAMIN3", "WRKNUMJOB2"]
numeric_features = ["WRKDHRSWK2", "IRALCAGE", "IRALCYFU", "IRALCBNG30D"]
SUBSTANCE_BINARY = ["MRJYR", "MRJMON", "COCYR", "CRKYR", "HERYR", "HERMON"]
binary_features = ["ALCYR", "ALCMON"] + SUBSTANCE_BINARY

NEEDED = set(["AMIPY", "SMIPY"] + categorical_features + ordinal_features
             + numeric_features + binary_features)

# ------------------------------------------------- BARRA DE PROGRESSO / TEMPOS
STAGE_NAMES = [
    "1. Carregar dados",
    "2. Filtrar adultos e limpar",
    "3. Dividir treino/teste",
    "4. Busca em grade (CV 5 folds)",
    "5. Escolher configuracao e limiar",
    "6. Treino final e predicao no teste",
    "7. Metricas, baselines e grafico",
]
TIMES = {}
overall = tqdm(total=len(STAGE_NAMES), desc="Progresso geral", unit="etapa", position=0)


@contextmanager
def stage(name):
    overall.set_description(f"Etapa {name}")
    t0 = time.perf_counter()
    yield
    TIMES[name] = time.perf_counter() - t0
    overall.update(1)


def fmt(sec):
    m, s = divmod(int(round(sec)), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:02d}m{s:02d}s"


tqdm.write(f"[INFO] Dispositivo: {DEVICE} | Oversampling: {OVERSAMPLING}")


# ---------------------------------------------------------------- FUNCOES
def _norm(c):
    return str(c).replace("\ufeff", "").strip().upper()


def make_preprocessor():
    t = []
    if categorical_features:
        t.append(("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                   ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                  categorical_features))
    for name, cols in [("ord", ordinal_features), ("num", numeric_features)]:
        if cols:
            t.append((name, Pipeline([("i", SimpleImputer(strategy="median")),
                                      ("s", StandardScaler())]), cols))
    if binary_features:
        t.append(("bin", Pipeline([("i", SimpleImputer(strategy="most_frequent")),
                                   ("s", StandardScaler())]), binary_features))
    return ColumnTransformer(t, remainder="drop")


def resample(X_tr, y_tr):
    """Oversampling aplicado SOMENTE no treino (nunca em validacao/teste)."""
    if OVERSAMPLING == "none":
        return X_tr, y_tr
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    sampler = (RandomOverSampler(random_state=RANDOM_STATE) if OVERSAMPLING == "random"
               else SMOTE(random_state=RANDOM_STATE, k_neighbors=5))
    return sampler.fit_resample(X_tr, y_tr)


def neighbors(X_tr, y_tr_, X_q, metric, kmax, batch=1000, desc="vizinhos"):
    """Distancias e rotulos dos kmax vizinhos mais proximos (GPU se disponivel)."""
    Xt = torch.tensor(X_tr, dtype=torch.float32, device=DEVICE)
    yt = torch.tensor(np.asarray(y_tr_), dtype=torch.long, device=DEVICE)
    p = 2.0 if metric == "euclidean" else 1.0
    D, L = [], []
    with torch.no_grad():
        for i in tqdm(range(0, len(X_q), batch), desc=desc, unit="lote",
                      leave=False, position=2):
            xb = torch.tensor(X_q[i:i + batch], dtype=torch.float32, device=DEVICE)
            dk, idx = torch.topk(torch.cdist(xb, Xt, p=p), k=kmax, largest=False, sorted=True)
            D.append(dk.cpu()); L.append(yt[idx].cpu())
    return torch.cat(D).numpy().astype(np.float64), torch.cat(L).numpy().astype(np.float64)


def proba_from(D, L, k, weights):
    Lk = L[:, :k]
    if weights == "uniform":
        return Lk.mean(axis=1)
    w = 1.0 / np.maximum(D[:, :k], 1e-12)
    return (w * Lk).sum(axis=1) / w.sum(axis=1)


def best_threshold(y_true, proba):
    ths = np.round(np.arange(0.10, 0.50, 0.005), 4)
    f1s = [f1_score(y_true, proba >= t, zero_division=0) for t in ths]
    i = int(np.argmax(f1s))
    return float(ths[i]), float(f1s[i])


def print_hub(y_true, pred):
    print("-" * 50)
    print(f"Acurácia: {accuracy_score(y_true, pred):.4f}")
    print(f"Acurácia Balanceada: {balanced_accuracy_score(y_true, pred):.4f}")
    print(f"F1-Score Binário: {f1_score(y_true, pred):.4f}")
    print("-" * 50)
    print("\nRelatório de Classificação:")
    print(classification_report(y_true, pred, labels=[0, 1],
                                target_names=CLASS_LABELS, zero_division=0))


# ================================================================== ETAPAS
with stage(STAGE_NAMES[0]):
    df = pd.read_csv(DATA_PATH, sep="\t", usecols=lambda c: _norm(c) in NEEDED, dtype=str)
    df.columns = [_norm(c) for c in df.columns]
    tqdm.write(f"Shape carregado: {df.shape} | colunas nao encontradas: {sorted(NEEDED - set(df.columns))}")
    categorical_features = [c for c in categorical_features if c in df.columns]
    ordinal_features = [c for c in ordinal_features if c in df.columns]
    numeric_features = [c for c in numeric_features if c in df.columns]
    binary_features = [c for c in binary_features if c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

with stage(STAGE_NAMES[1]):
    d = df.loc[df["AGE3"].between(4, 11)].copy()
    ok = d["AMIPY"].isin([0, 1]) & d["SMIPY"].isin([0, 1])
    cons = (d["SMIPY"] == 0) | ((d["SMIPY"] == 1) & (d["AMIPY"] == 1))
    d = d.loc[ok & cons].copy()
    d["TARGET"] = d["AMIPY"].astype(int)

    special = {"IRMARIT": [99], "EDUHIGHCAT": [5], "WRKNUMJOB2": [85, 94, 97, 98, 99],
               "WRKDHRSWK2": [985, 994, 997, 998, 999], "IRALCAGE": [991],
               "IRALCYFU": [9999],
               "WRKSTATWK2": [98, 99], "IRWRKSTAT": [99]}
    for c, codes in special.items():
        if c in d.columns:
            d[c] = d[c].replace(codes, np.nan)

    if "IRALCBNG30D" in d.columns:
        d["ALC_BINGE_NEVER_USED"] = d["IRALCBNG30D"].eq(91).astype(int)
        d["ALC_BINGE_NO_USE_PAST_MONTH"] = d["IRALCBNG30D"].eq(93).astype(int)
        d["IRALCBNG30D"] = d["IRALCBNG30D"].replace({91: 0, 93: 0})
        bad = ~d["IRALCBNG30D"].between(0, 30) & d["IRALCBNG30D"].notna()
        d.loc[bad, "IRALCBNG30D"] = np.nan
        binary_features += ["ALC_BINGE_NEVER_USED", "ALC_BINGE_NO_USE_PAST_MONTH"]

    for c in SUBSTANCE_BINARY + ["ALCYR", "ALCMON"]:
        if c in d.columns:
            d.loc[~d[c].isin([0, 1]), c] = np.nan

    all_feats = categorical_features + ordinal_features + numeric_features + binary_features
    const = [c for c in all_feats if d[c].nunique(dropna=True) <= 1]
    categorical_features = [c for c in categorical_features if c not in const]
    ordinal_features = [c for c in ordinal_features if c not in const]
    numeric_features = [c for c in numeric_features if c not in const]
    binary_features = [c for c in binary_features if c not in const]
    features = categorical_features + ordinal_features + numeric_features + binary_features

    X, y = d[features].copy(), d["TARGET"].copy()
    keep = X.notna().any(axis=1)
    X, y = X.loc[keep].copy(), y.loc[keep].copy()
    tqdm.write(f"Adultos validos: {len(X)} | prevalencia AMI: {y.mean():.4f}")

with stage(STAGE_NAMES[2]):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    y_tr_arr, y_te_arr = y_train.to_numpy(), y_test.to_numpy()
    tqdm.write(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

with stage(STAGE_NAMES[3]):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    kmax = max(K_GRID)
    oof = {(m, k, w): np.zeros(len(X_train)) for m in METRICS for k in K_GRID for w in WEIGHTS}
    inner = tqdm(total=N_SPLITS * len(METRICS), desc="CV (fold x metrica)", unit="passo",
                 position=1, leave=False)
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train), 1):
        pre = make_preprocessor()
        Xtr = pre.fit_transform(X_train.iloc[tr])          # ajustado so no treino do fold
        Xva = pre.transform(X_train.iloc[va])
        Xtr_fit, ytr_fit = resample(Xtr, y_tr_arr[tr])     # oversampling so no treino do fold
        for m in METRICS:
            D, L = neighbors(Xtr_fit, ytr_fit, Xva, m, kmax, desc=f"fold {fold} {m}")
            for k in K_GRID:
                for w in WEIGHTS:
                    oof[(m, k, w)][va] = proba_from(D, L, k, w)
            inner.update(1)
    inner.close()

with stage(STAGE_NAMES[4]):
    rows = []
    for (m, k, w), p in oof.items():
        thr, f1_t = best_threshold(y_train, p)
        rows.append({"metric": m, "k": k, "weights": w, "F1_AMI_tuned": f1_t, "thr": thr})
    cv_table = pd.DataFrame(rows).sort_values("F1_AMI_tuned", ascending=False).reset_index(drop=True)
    best = cv_table.iloc[0]
    BEST_K, BEST_THR = int(best.k), float(best.thr)

with stage(STAGE_NAMES[5]):
    pre = make_preprocessor()
    Xtr = pre.fit_transform(X_train)
    Xte = pre.transform(X_test)
    Xtr_fit, ytr_fit = resample(Xtr, y_tr_arr)             # teste mantem a prevalencia original
    D, L = neighbors(Xtr_fit, ytr_fit, Xte, best.metric, BEST_K, desc="teste")
    proba = proba_from(D, L, BEST_K, best.weights)
    pred = (proba >= BEST_THR).astype(int)

with stage(STAGE_NAMES[6]):
    cm = confusion_matrix(y_te_arr, pred, labels=[0, 1])
    roc, pr = roc_auc_score(y_te_arr, proba), average_precision_score(y_te_arr, proba)
    rec_05 = ((proba >= 0.5) & (y_te_arr == 1)).sum() / (y_te_arr == 1).sum()
    base_maj = 1 - y_te_arr.mean()
    base_all = f1_score(y_te_arr, np.ones_like(y_te_arr))
    cm_path = BASE_DIR / f"matriz_confusao_{OVERSAMPLING}.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5.5))
        im = ax.imshow(cm, cmap="Reds")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sem Transtorno", "Com Transtorno"])
        ax.set_yticklabels(["Sem Transtorno", "Com Transtorno"])
        ax.set_xlabel("Classe Prevista"); ax.set_ylabel("Classe Real")
        ax.set_title("Matriz de Confusão", fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        fig.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(cm_path, dpi=150); plt.close(fig)
    except Exception as e:
        tqdm.write(f"[AVISO] grafico nao gerado ({type(e).__name__}: {e})")

overall.set_description("Concluido")
overall.close()

# ================================================================ RESULTADOS
print("\n" + "=" * 50)
print(f"RESULTADO FINAL  |  oversampling: {OVERSAMPLING}  |  dispositivo: {DEVICE}")
print("=" * 50)
print_hub(y_te_arr, pred)

print("Informações adicionais")
print(f"  Configuração (escolhida por CV no treino): metric={best.metric}, k={BEST_K}, "
      f"weights={best.weights}, limiar={BEST_THR:.3f}")
print(f"  ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} (aleatório = {y_te_arr.mean():.4f})")
print(f"  Recall AMI no limiar 0,50 (referência): {rec_05:.4f}")
print(f"  Matriz de confusão [[TN FP] [FN TP]]:\n{cm}")
print(f"  Baselines: acurácia da classe majoritária = {base_maj:.4f} | F1 de 'sempre AMI' = {base_all:.4f}")
if cm_path.exists():
    print(f"  Gráfico salvo em: {cm_path}")

print("\nTop 5 da busca (validação cruzada no treino):")
print(cv_table.head(5).round(4).to_string(index=False))

# ------------------------------------------------------------ TEMPOS
total = sum(TIMES.values())
print("\n" + "-" * 50)
print("Tempo de execução por etapa")
print("-" * 50)
for name in STAGE_NAMES:
    t = TIMES.get(name, 0.0)
    print(f"{name:<40} {fmt(t):>10}  {100 * t / total:5.1f}%")
print(f"{'TOTAL':<40} {fmt(total):>10}")

row = {"data_hora": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
       "oversampling": OVERSAMPLING, "dispositivo": str(DEVICE)}
row.update({n: round(TIMES.get(n, 0.0), 2) for n in STAGE_NAMES})
row["TOTAL_s"] = round(total, 2)
try:
    hist = pd.read_csv(TIMES_CSV) if TIMES_CSV.exists() else pd.DataFrame()
    hist = pd.concat([hist, pd.DataFrame([row])], ignore_index=True)
    hist.to_csv(TIMES_CSV, index=False)
    same = hist[(hist["oversampling"] == OVERSAMPLING) & (hist["dispositivo"] == str(DEVICE))]
    if len(same) >= 2:
        print(f"\nMédia de {len(same)} execuções (oversampling={OVERSAMPLING}, {DEVICE}): "
              f"{fmt(same['TOTAL_s'].mean())}  (mín {fmt(same['TOTAL_s'].min())}, "
              f"máx {fmt(same['TOTAL_s'].max())})")
    print(f"Tempos registrados em: {TIMES_CSV}")
except Exception as e:
    print(f"[AVISO] não foi possível atualizar {TIMES_CSV}: {e}")