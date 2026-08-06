"""SparseGuard-NIDS research pipeline.

This code is designed for Google Colab execution from Drive. It creates publication-grade
intermediate samples, metrics, plots, runtime logs, and paper-ready tables. The model is
architecture-first: semantic multi-path branches, cross-branch attention, reconstruction
consistency, uncertainty, and sparse-attack robustness hooks.
"""
from __future__ import annotations

import io, json, os, time, zipfile, tarfile, math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, matthews_corrcoef, confusion_matrix, brier_score_loss
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except Exception:
    torch = None

def _find_upward_marker(start: Path, marker: str) -> Optional[Path]:
    for parent in [start, *start.parents]:
        if (parent / marker).exists():
            return parent
    return None


def locate_project_root() -> Path:
    env = os.environ.get('SPARSEGUARD_ROOT')
    if env and (Path(env) / 'src' / 'sparseguard_pipeline.py').exists():
        return Path(env)
    here = Path(__file__).resolve()
    root = _find_upward_marker(here.parent, 'README.md')
    if root and (root / 'src' / 'sparseguard_pipeline.py').exists():
        return root
    drive = Path('/users/')
    if drive.exists():
        for base in [drive]:
            for candidate in base.rglob('IMPLEMENTATION/src/sparseguard_pipeline.py'):
                return candidate.parents[1]
    raise FileNotFoundError('Could not locate IMPLEMENTATION/src/sparseguard_pipeline.py after mounting Drive.')


def locate_dataset_root() -> Path:
    env = os.environ.get('SPARSEGUARD_DATASET_ROOT')
    if env and (Path(env) / 'X-IIoTID' / 'X-IIoTID dataset.csv').exists():
        return Path(env)
    drive = Path('/users/')
    if drive.exists():
        for base in [drive]:
            matches = list(base.rglob('X-IIoTID/X-IIoTID dataset.csv'))
            if matches:
                return matches[0].parents[1]
    raise FileNotFoundError('Could not locate dataset root containing X-IIoTID/X-IIoTID dataset.csv.')


PROJECT_ROOT = locate_project_root()
DATASET_ROOT = locate_dataset_root()
os.environ['SPARSEGUARD_ROOT'] = str(PROJECT_ROOT)
os.environ['SPARSEGUARD_DATASET_ROOT'] = str(DATASET_ROOT)

DATASETS = {
    'x_iiotid': {
        'name': 'X-IIoTID', 'file': 'X-IIoTID/X-IIoTID dataset.csv', 'expected_rows': 820834, 'expected_columns': 68,
        'labels': ['class1','class2','label','Label','attack','Attack','target'], 'benign': ['normal','benign','Normal','Benign','0',0]
    },
    'cic_iiot_2025': {
        'name': 'CIC-IIoT-2025', 'file': 'CIC-IIoT-2025/dataset/processed_files/all_attack_benign_samples.tar.xz',
        'labels': ['label1','label','Label','label_full','class','Class','attack_type','attack','category'], 'benign': ['benign','Benign','normal','Normal','0',0]
    },
    'cic_ids2017': {
        'name': 'CIC-IDS2017', 'file': 'CIC-IDS2017/CIC-IDS-2017/CSVs/MachineLearningCSV.zip', 'expected_rows': 2830743, 'expected_csv_files': 8,
        'labels': ['Label','label','class','Class'], 'benign': ['BENIGN','Benign','benign','0',0]
    },
}

SEMANTIC_PATTERNS = {
    'protocol_semantics': ['protocol','proto','service','state','flag','tcp','udp','icmp'],
    'flow_timing': ['duration','time','timestamp','idle','active','iat'],
    'packet_volume': ['packet','pkts','pkt','count','total','fwd','bwd'],
    'byte_volume': ['byte','bytes','octet','len','length','header'],
    'rate_dynamics': ['rate','second','sec','srate','drate'],
    'port_addressing': ['port','sport','dport','src','dst','ip'],
    'statistical_flags': ['mean','std','min','max','avg','variance','entropy'],
    'unknown_numeric': [],
}


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True); return path

def write_json(path: Path, payload: dict):
    ensure(path.parent); path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')

def save_head(df: pd.DataFrame, path: Path, n=25):
    ensure(path.parent); df.head(n).to_csv(path, index=False)

def find_label(df: pd.DataFrame, candidates: list[str]) -> str:
    direct = [c for c in candidates if c in df.columns]
    if direct: return direct[0]
    lower = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    raise ValueError(f'Label column not found. candidates={candidates}; columns={list(df.columns)[:30]}')

def to_binary(y: pd.Series, benign_values: list) -> pd.Series:
    benign = {str(v).strip().lower() for v in benign_values}
    return (~y.astype(str).str.strip().str.lower().isin(benign)).astype(int)

def metrics(y_true, y_prob, threshold=0.5) -> dict:
    y_true = np.asarray(y_true).astype(int); y_prob = np.asarray(y_prob).astype(float); y_pred = (y_prob >= threshold).astype(int)
    out = {'accuracy': float(accuracy_score(y_true,y_pred)), 'balanced_accuracy': float(balanced_accuracy_score(y_true,y_pred)), 'precision': float(precision_score(y_true,y_pred,zero_division=0)), 'recall': float(recall_score(y_true,y_pred,zero_division=0)), 'f1': float(f1_score(y_true,y_pred,zero_division=0)), 'mcc': float(matthews_corrcoef(y_true,y_pred)), 'brier': float(brier_score_loss(y_true,y_prob)), 'confusion_matrix': confusion_matrix(y_true,y_pred).tolist()}
    if len(np.unique(y_true)) > 1:
        out['roc_auc'] = float(roc_auc_score(y_true,y_prob)); out['average_precision'] = float(average_precision_score(y_true,y_prob))
    return out

def infer_group(col: str) -> str:
    c = str(col).lower()
    for group, pats in SEMANTIC_PATTERNS.items():
        if any(p in c for p in pats): return group
    return 'unknown_numeric'

FIXED_SEMANTIC_GROUPS = list(SEMANTIC_PATTERNS)

LEAKAGE_NAME_TOKENS = [
    'label', 'class', 'target', 'attack', 'category',
    'timestamp', 'date', 'time_start', 'time_end',
    'device_name', 'device_mac', 'mac', 'ip', 'ips_all', 'ips_dst', 'ips_src',
]

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def leakage_or_identifier(col: str) -> bool:
    c = str(col).strip().lower()
    return any(tok in c for tok in LEAKAGE_NAME_TOKENS)

def numeric_feature_frame(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    drop = [c for c in df.columns if c == label_col or leakage_or_identifier(c)]
    X = df.drop(columns=drop, errors='ignore').copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.replace([np.inf, -np.inf], np.nan)
    keep = [c for c in X.columns if X[c].notna().any()]
    if not keep:
        raise ValueError('No numeric non-leakage features remained after schema cleanup.')
    X = X[keep].fillna(X[keep].median(numeric_only=True)).fillna(0)
    return X.astype('float32')

def semantic_aggregate_from_features(X: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    groups = {g: [] for g in FIXED_SEMANTIC_GROUPS}
    for c in X.columns:
        groups.setdefault(infer_group(c), []).append(c)
    n = len(X)
    for group in FIXED_SEMANTIC_GROUPS:
        cols = groups.get(group, [])
        if cols:
            vals = X[cols].astype('float32')
            rows[f'{group}_mean'] = vals.mean(axis=1).to_numpy(dtype='float32')
            rows[f'{group}_std'] = vals.std(axis=1).fillna(0).to_numpy(dtype='float32')
            rows[f'{group}_min'] = vals.min(axis=1).to_numpy(dtype='float32')
            rows[f'{group}_max'] = vals.max(axis=1).to_numpy(dtype='float32')
            rows[f'{group}_abs_mean'] = vals.abs().mean(axis=1).to_numpy(dtype='float32')
            rows[f'{group}_nonzero_ratio'] = (vals.ne(0).sum(axis=1) / max(1, len(cols))).to_numpy(dtype='float32')
        else:
            for stat in ['mean', 'std', 'min', 'max', 'abs_mean', 'nonzero_ratio']:
                rows[f'{group}_{stat}'] = np.zeros(n, dtype='float32')
    return pd.DataFrame(rows)

def semantic_aggregate_dataset(df: pd.DataFrame, label_candidates: list[str], benign_values: list, dataset_name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = normalize_columns(df).replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='all')
    label_col = find_label(df, label_candidates)
    y = to_binary(df[label_col], benign_values)
    X = numeric_feature_frame(df, label_col)
    A = semantic_aggregate_from_features(X)
    manifest = {
        'dataset': dataset_name,
        'rows': int(len(df)),
        'features_before_semantic_aggregation': int(X.shape[1]),
        'semantic_features': int(A.shape[1]),
        'label_column': label_col,
        'target_distribution': y.value_counts().sort_index().to_dict(),
        'semantic_group_counts': {g: int(sum(infer_group(c) == g for c in X.columns)) for g in FIXED_SEMANTIC_GROUPS},
    }
    return A, y.astype(int), manifest

def eda_x_iiotid():
    out = ensure(PROJECT_ROOT/'EDA/results')
    spec = DATASETS['x_iiotid']; path = DATASET_ROOT/spec['file']
    df = pd.read_csv(path, low_memory=False)
    label = find_label(df, spec['labels'])
    save_head(df, out/'x_iiotid_raw_head25.csv')
    df.describe(include='all').transpose().to_csv(out/'x_iiotid_describe.csv')
    df[label].value_counts().to_csv(out/'x_iiotid_label_distribution.csv')
    plt.figure(figsize=(10,4)); sns.countplot(data=df, x=label, order=df[label].value_counts().index); plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(out/'x_iiotid_label_distribution.png', dpi=220); plt.close()
    numeric = df.select_dtypes(include='number').iloc[:, :50]
    plt.figure(figsize=(12,10)); sns.heatmap(numeric.corr(), cmap='vlag', center=0); plt.tight_layout(); plt.savefig(out/'x_iiotid_correlation_heatmap_top50.png', dpi=220); plt.close()
    groups = pd.Series({c: infer_group(c) for c in df.columns if c != label}, name='semantic_group')
    groups.to_csv(out/'x_iiotid_semantic_feature_groups.csv')
    write_json(out/'eda_summary.json', {'dataset': spec['name'], 'rows': int(len(df)), 'columns': int(df.shape[1]), 'label': label, 'missing_cells': int(df.isna().sum().sum()), 'duplicate_rows': int(df.duplicated().sum())})


def preprocess_x_iiotid():
    out = ensure(PROJECT_ROOT/'PREPROCESSING/results')
    spec = DATASETS['x_iiotid']; df = pd.read_csv(DATASET_ROOT/spec['file'], low_memory=False)
    label = find_label(df, spec['labels'])
    save_head(df, out/'step_0_raw_sample.csv')
    cleaned = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').drop_duplicates()
    save_head(cleaned, out/'step_1_cleaned_sample.csv')
    y = to_binary(cleaned[label], spec['benign'])
    leakage_cols = {
        'class1', 'class2', 'class3', 'label', 'target',
        'Date', 'Timestamp', 'Scr_IP', 'Des_IP',
    }
    drop_cols = [c for c in cleaned.columns if c == label or c in leakage_cols or str(c).lower() in {x.lower() for x in leakage_cols}]
    X = cleaned.drop(columns=drop_cols, errors='ignore').copy()
    for c in X.columns:
        if X[c].dtype == 'object': X[c] = X[c].astype('category').cat.codes
        X[c] = pd.to_numeric(X[c], errors='coerce')
    X = X.fillna(X.median(numeric_only=True))
    save_head(X.assign(target=y.values), out/'step_2_encoded_sample.csv')
    scaler = StandardScaler(); Xs = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    save_head(Xs.assign(target=y.values), out/'step_3_scaled_sample.csv')
    train, temp, y_train, y_temp = train_test_split(Xs, y, test_size=.30, random_state=42, stratify=y)
    val, test, y_val, y_test = train_test_split(temp, y_temp, test_size=2/3, random_state=42, stratify=y_temp)
    for name, a, b in [('train',train,y_train),('val',val,y_val),('test',test,y_test)]: a.assign(target=b.values).to_parquet(out/f'x_iiotid_{name}.parquet', index=False)
    joblib.dump(scaler, out/'x_iiotid_scaler.joblib')
    write_json(out/'preprocessing_manifest.json', {'label': label, 'dropped_leakage_columns': drop_cols, 'raw_rows': int(len(df)), 'clean_rows': int(len(cleaned)), 'features': int(Xs.shape[1]), 'target_distribution': y.value_counts().sort_index().to_dict(), 'split_rows': {'train': len(train), 'val': len(val), 'test': len(test)}})

if torch is not None:
    class GatedResidualBlock(nn.Module):
        def __init__(self, dim, dropout):
            super().__init__(); self.ff = nn.Sequential(nn.Linear(dim,dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim,dim)); self.gate = nn.Sequential(nn.Linear(dim,dim), nn.Sigmoid()); self.norm = nn.LayerNorm(dim)
        def forward(self,x): return self.norm(x + self.gate(x)*self.ff(x))
    class SemanticBranch(nn.Module):
        def __init__(self, in_dim, branch_dim=64, dropout=.2):
            super().__init__(); self.net = nn.Sequential(nn.Linear(max(in_dim,1), branch_dim), nn.LayerNorm(branch_dim), nn.GELU(), GatedResidualBlock(branch_dim, dropout))
        def forward(self,x): return self.net(x)
    class SparseGuardMultiPath(nn.Module):
        def __init__(self, group_dims: Dict[str,int], branch_dim=64, hidden_dim=128, dropout=.2, heads=4):
            super().__init__(); self.groups = list(group_dims); self.branches = nn.ModuleDict({g: SemanticBranch(d, branch_dim, dropout) for g,d in group_dims.items()}); self.attn = nn.MultiheadAttention(branch_dim, heads, dropout=dropout, batch_first=True); self.fusion = nn.Sequential(nn.Linear(branch_dim*len(self.groups), hidden_dim), nn.GELU(), nn.Dropout(dropout), GatedResidualBlock(hidden_dim, dropout)); self.cls = nn.Linear(hidden_dim,1); self.unc = nn.Linear(hidden_dim,1); self.sem = nn.Linear(hidden_dim,1); self.rec = nn.ModuleDict({g: nn.Linear(branch_dim, max(group_dims[g],1)) for g in self.groups})
        def forward(self, xg):
            zs=[]; recon={}
            for g in self.groups:
                z=self.branches[g](xg[g]); zs.append(z); recon[g]=self.rec[g](z)
            tok=torch.stack(zs, dim=1); att,_=self.attn(tok,tok,tok,need_weights=True); h=self.fusion(att.flatten(1)); logit=self.cls(h).squeeze(-1)
            return {'logit':logit, 'prob':torch.sigmoid(logit), 'uncertainty':F.softplus(self.unc(h)).squeeze(-1), 'semantic_violation':torch.sigmoid(self.sem(h)).squeeze(-1), 'reconstruction':recon}
    class FrameDataset(Dataset):
        def __init__(self, df): self.y=df['target'].astype('float32').values; self.cols=[c for c in df.columns if c!='target']; self.X=df[self.cols].astype('float32').values
        def __len__(self): return len(self.y)
        def __getitem__(self,i): return torch.from_numpy(self.X[i]), torch.tensor(self.y[i])

def group_slices(columns):
    out={}
    for i,c in enumerate(columns): out.setdefault(infer_group(c), []).append(i)
    return out

def to_groups(X, slices): return {g: X[:,idxs] if idxs else X[:,:1]*0 for g,idxs in slices.items()}

def rec_loss(out, groups):
    losses=[F.smooth_l1_loss(out['reconstruction'][g], x) for g,x in groups.items() if out['reconstruction'][g].shape == x.shape]
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=out['logit'].device)

def train_sparseguard(max_epochs=80, batch_size=4096):
    if torch is None: raise RuntimeError('PyTorch is required')
    out = ensure(PROJECT_ROOT/'EXPERIMENT/results'); pre = PROJECT_ROOT/'PREPROCESSING/results'
    train_df=pd.read_parquet(pre/'x_iiotid_train.parquet'); val_df=pd.read_parquet(pre/'x_iiotid_val.parquet')
    train_ds=FrameDataset(train_df); val_ds=FrameDataset(val_df); slices=group_slices(train_ds.cols); dims={g:len(v) for g,v in slices.items()}
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=SparseGuardMultiPath(dims).to(device); opt=torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4); bce=nn.BCEWithLogitsLoss()
    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,num_workers=0); val_loader=DataLoader(val_ds,batch_size=batch_size,shuffle=False,num_workers=0)
    hist=[]; best=-1; patience=12
    for epoch in range(1,max_epochs+1):
        t=time.perf_counter(); model.train(); losses=[]
        for X,y in train_loader:
            X=X.to(device); y=y.to(device); g=to_groups(X,slices); o=model(g); loss=bce(o['logit'], y)+0.2*rec_loss(o,g); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        probs=[]; ys=[]; model.eval()
        with torch.no_grad():
            for X,y in val_loader:
                X=X.to(device); o=model(to_groups(X,slices)); probs.append(o['prob'].cpu().numpy()); ys.append(y.numpy())
        m=metrics(np.concatenate(ys), np.concatenate(probs)); row={'epoch':epoch,'seconds':time.perf_counter()-t,'train_loss':float(np.mean(losses)),**{f'val_{k}':v for k,v in m.items() if isinstance(v,(int,float))}}; hist.append(row)
        if m['f1']>best: best=m['f1']; patience=12; torch.save({'state_dict':model.state_dict(),'columns':train_ds.cols,'slices':slices,'dims':dims}, out/'sparseguard_best.pt')
        else:
            patience-=1
            if patience<=0: break
    pd.DataFrame(hist).to_csv(out/'training_history.csv', index=False)
    plt.figure(figsize=(10,5)); h=pd.DataFrame(hist); [plt.plot(h['epoch'],h[c],label=c) for c in h.columns if c not in ['epoch','seconds'] and h[c].dtype!='O']; plt.legend(); plt.tight_layout(); plt.savefig(out/'training_curves.png', dpi=220); plt.close()
    write_json(out/'training_summary.json', {'best_val_f1': best, 'epochs': len(hist), 'device': str(device), 'semantic_groups': {k:len(v) for k,v in slices.items()}})

def _load_sparseguard_checkpoint():
    if torch is None: raise RuntimeError('PyTorch is required')
    path = PROJECT_ROOT/'EXPERIMENT/results/sparseguard_best.pt'
    ckpt = torch.load(path, map_location='cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SparseGuardMultiPath(ckpt['dims']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    return model, ckpt, device

def _predict_frame(model, frame, slices, device, batch_size=8192):
    ds = FrameDataset(frame)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs, ys = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            probs.append(model(to_groups(X, slices))['prob'].cpu().numpy())
            ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(probs)

def evaluate_sparseguard_test():
    from sklearn.metrics import roc_curve, precision_recall_curve
    model, ckpt, device = _load_sparseguard_checkpoint()
    out = ensure(PROJECT_ROOT/'EXPERIMENT/results')
    test = pd.read_parquet(PROJECT_ROOT/'PREPROCESSING/results/x_iiotid_test.parquet')
    y, prob = _predict_frame(model, test[ckpt['columns'] + ['target']], ckpt['slices'], device)
    m = metrics(y, prob)
    write_json(out/'test_metrics.json', m)
    pd.DataFrame({'y_true': y[:5000], 'p_attack': prob[:5000]}).to_csv(out/'test_prediction_sample.csv', index=False)
    cm = np.array(m['confusion_matrix'])
    plt.figure(figsize=(4.5,4)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.xlabel('Predicted'); plt.ylabel('True'); plt.tight_layout(); plt.savefig(out/'test_confusion_matrix.png', dpi=220); plt.close()
    fpr, tpr, _ = roc_curve(y, prob); prec, rec, _ = precision_recall_curve(y, prob)
    plt.figure(figsize=(10,4)); plt.subplot(1,2,1); plt.plot(fpr,tpr); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC'); plt.subplot(1,2,2); plt.plot(rec,prec); plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR'); plt.tight_layout(); plt.savefig(out/'test_roc_pr_curves.png', dpi=220); plt.close()
    return m

class MLPBaseline(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        logit = self.net(x).squeeze(-1)
        return {'logit': logit, 'prob': torch.sigmoid(logit)}

def _train_variant(name, train_df, val_df, test_df, slices, model_kind='sparseguard', rec_weight=.2, max_epochs=35, batch_size=4096):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_ds, val_ds, test_ds = FrameDataset(train_df), FrameDataset(val_df), FrameDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    if model_kind == 'mlp':
        model = MLPBaseline(len(train_ds.cols)).to(device)
    else:
        model = SparseGuardMultiPath({g: len(v) for g,v in slices.items()}).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss()
    hist, best, best_state, patience = [], -1, None, 8
    for epoch in range(1, max_epochs+1):
        t0=time.perf_counter(); model.train(); losses=[]
        for X,y in train_loader:
            X=X.to(device); y=y.to(device)
            if model_kind == 'mlp':
                o=model(X); loss=bce(o['logit'], y)
            else:
                g=to_groups(X,slices); o=model(g); loss=bce(o['logit'], y)+rec_weight*rec_loss(o,g)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval(); probs=[]; ys=[]
        with torch.no_grad():
            for X,y in val_loader:
                X=X.to(device); o=model(X) if model_kind == 'mlp' else model(to_groups(X,slices)); probs.append(o['prob'].cpu().numpy()); ys.append(y.numpy())
        vm=metrics(np.concatenate(ys), np.concatenate(probs))
        hist.append({'variant':name,'epoch':epoch,'seconds':time.perf_counter()-t0,'train_loss':float(np.mean(losses)), **{f'val_{k}':v for k,v in vm.items() if isinstance(v,(int,float))}})
        if vm['f1'] > best:
            best=vm['f1']; patience=8; best_state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0: break
    model.load_state_dict(best_state); model.to(device); model.eval()
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    probs=[]; ys=[]
    with torch.no_grad():
        for X,y in test_loader:
            X=X.to(device); o=model(X) if model_kind == 'mlp' else model(to_groups(X,slices)); probs.append(o['prob'].cpu().numpy()); ys.append(y.numpy())
    tm=metrics(np.concatenate(ys), np.concatenate(probs))
    return hist, {'variant':name, 'best_val_f1':best, **{f'test_{k}':v for k,v in tm.items() if isinstance(v,(int,float))}}

def run_ablation_experiments():
    out = ensure(PROJECT_ROOT/'ABLATION/results')
    pre = PROJECT_ROOT/'PREPROCESSING/results'
    train_df = pd.read_parquet(pre/'x_iiotid_train.parquet')
    val_df = pd.read_parquet(pre/'x_iiotid_val.parquet')
    test_df = pd.read_parquet(pre/'x_iiotid_test.parquet')
    cols = [c for c in train_df.columns if c != 'target']
    semantic = group_slices(cols)
    single = {'single_path_all_features': list(range(len(cols)))}
    variants = [
        ('full_semantic_multipath_repeat', semantic, 'sparseguard', .2),
        ('no_reconstruction_loss', semantic, 'sparseguard', 0.0),
        ('single_path_attention', single, 'sparseguard', .2),
        ('plain_mlp_baseline', semantic, 'mlp', 0.0),
    ]
    all_hist, rows = [], []
    for name, slices, kind, rec_w in variants:
        hist, row = _train_variant(name, train_df, val_df, test_df, slices, kind, rec_w)
        all_hist.extend(hist); rows.append(row)
        pd.DataFrame(all_hist).to_csv(out/'ablation_training_history.csv', index=False)
        pd.DataFrame(rows).to_csv(out/'ablation_metrics.csv', index=False)
    df = pd.DataFrame(rows)
    plt.figure(figsize=(9,4)); sns.barplot(data=df, x='variant', y='test_f1'); plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.savefig(out/'ablation_test_f1.png', dpi=220); plt.close()
    write_json(out/'ablation_summary.json', {'variants': rows})
    return rows

def run_robustness_actual(sample_n=4096):
    model, ckpt, device = _load_sparseguard_checkpoint()
    out = ensure(PROJECT_ROOT/'ROBUSTNESS/results')
    test = pd.read_parquet(PROJECT_ROOT/'PREPROCESSING/results/x_iiotid_test.parquet')[ckpt['columns'] + ['target']]
    test = test.sample(n=min(sample_n, len(test)), random_state=42)
    X = torch.tensor(test[ckpt['columns']].astype('float32').values, device=device)
    y = torch.tensor(test['target'].astype('float32').values, device=device)
    rows = []
    with torch.no_grad():
        clean = model(to_groups(X, ckpt['slices']))['prob'].detach().cpu().numpy()
    for eps in [0.01, 0.03, 0.05, 0.10]:
        X_adv = X.detach().clone().requires_grad_(True)
        o = model(to_groups(X_adv, ckpt['slices']))
        loss = F.binary_cross_entropy_with_logits(o['logit'], y)
        grad = torch.autograd.grad(loss, X_adv)[0]
        adv = torch.clamp(X + eps * grad.sign(), -6, 6)
        with torch.no_grad():
            adv_prob = model(to_groups(adv, ckpt['slices']))['prob'].detach().cpu().numpy()
        clean_pred = (clean >= .5).astype(int); adv_pred = (adv_prob >= .5).astype(int); y_np = y.detach().cpu().numpy().astype(int)
        rows.append({'attack':'FGSM','epsilon':eps,'sample_n':len(test),'prediction_flip_rate':float((clean_pred!=adv_pred).mean()),'attack_to_benign_evasion':float(((y_np==1)&(clean_pred==1)&(adv_pred==0)).sum()/max(1,((y_np==1)&(clean_pred==1)).sum())),'mean_probability_shift':float(np.mean(clean-adv_prob))})
    df = pd.DataFrame(rows); df.to_csv(out/'fgsm_robustness_metrics.csv', index=False)
    plt.figure(figsize=(7,4)); sns.lineplot(data=df, x='epsilon', y='attack_to_benign_evasion', marker='o'); plt.tight_layout(); plt.savefig(out/'fgsm_evasion_curve.png', dpi=220); plt.close()
    write_json(out/'robustness_summary.json', {'rows': rows})
    return rows

if torch is not None:
    class FlatSparseGuardWrapper(nn.Module):
        def __init__(self, model, slices):
            super().__init__()
            self.model = model
            self.slices = slices
        def forward(self, x):
            return self.model(to_groups(x, self.slices))['logit'].unsqueeze(-1)

    class TabularVAE(nn.Module):
        def __init__(self, input_dim, latent_dim=16, hidden_dim=128):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU())
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, input_dim))
        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h).clamp(-8, 8)
        def reparameterize(self, mu, logvar):
            if self.training:
                return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return mu
        def forward(self, x):
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar

def _load_preprocessed_split(name='test', columns=None, sample_n=None, seed=42):
    path = PROJECT_ROOT/'PREPROCESSING/results'/f'x_iiotid_{name}.parquet'
    df = pd.read_parquet(path)
    if columns is not None:
        df = df[list(columns) + ['target']]
    if sample_n is not None and len(df) > sample_n:
        parts = []
        for cls in [0, 1]:
            part = df[df['target'] == cls]
            if len(part):
                parts.append(part.sample(n=min(len(part), sample_n // 2), random_state=seed + cls))
        df = pd.concat(parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df

def _attack_summary(y_np, clean_prob, adv_prob):
    clean_pred = (clean_prob >= .5).astype(int)
    adv_pred = (adv_prob >= .5).astype(int)
    attack_mask = y_np.astype(int) == 1
    detected_attack = attack_mask & (clean_pred == 1)
    return {
        'prediction_flip_rate': float((clean_pred != adv_pred).mean()),
        'attack_to_benign_evasion': float(((detected_attack) & (adv_pred == 0)).sum() / max(1, detected_attack.sum())),
        'attack_success_rate': float(((attack_mask) & (adv_pred == 0)).sum() / max(1, attack_mask.sum())),
        'mean_clean_attack_probability': float(clean_prob[attack_mask].mean()) if attack_mask.any() else float('nan'),
        'mean_adv_attack_probability': float(adv_prob[attack_mask].mean()) if attack_mask.any() else float('nan'),
        'mean_probability_shift': float(np.mean(clean_prob - adv_prob)),
    }

def compute_xai_attributions(sample_n=2048, background_n=64, run_deepshap=True):
    if torch is None: raise RuntimeError('PyTorch is required')
    out = ensure(PROJECT_ROOT/'XAI/results')
    model, ckpt, device = _load_sparseguard_checkpoint()
    df = _load_preprocessed_split('test', ckpt['columns'], sample_n=sample_n, seed=101)
    X = torch.tensor(df[ckpt['columns']].astype('float32').values, device=device, requires_grad=True)
    y_np = df['target'].astype(int).values
    t0 = time.perf_counter()
    logits = model(to_groups(X, ckpt['slices']))['logit']
    score = logits.mean()
    grad = torch.autograd.grad(score, X)[0]
    grad_np = grad.detach().cpu().numpy()
    x_np = X.detach().cpu().numpy()
    grad_input = grad_np * x_np
    abs_attr = np.abs(grad_input)
    signed_attr = grad_input
    feature_rows = []
    for i, col in enumerate(ckpt['columns']):
        feature_rows.append({
            'feature': col,
            'semantic_group': infer_group(col),
            'mean_abs_gradient_input': float(abs_attr[:, i].mean()),
            'mean_signed_gradient_input': float(signed_attr[:, i].mean()),
            'std_abs_gradient_input': float(abs_attr[:, i].std()),
        })
    feature_df = pd.DataFrame(feature_rows).sort_values('mean_abs_gradient_input', ascending=False)
    feature_df.to_csv(out/'gradient_input_feature_importance.csv', index=False)
    feature_df.head(30).to_csv(out/'top30_gradient_input_features.csv', index=False)
    group_df = feature_df.groupby('semantic_group', as_index=False)['mean_abs_gradient_input'].sum().sort_values('mean_abs_gradient_input', ascending=False)
    group_df.to_csv(out/'semantic_group_importance.csv', index=False)

    top = feature_df.head(20)
    plt.figure(figsize=(9,6)); sns.barplot(data=top, y='feature', x='mean_abs_gradient_input', hue='semantic_group', dodge=False); plt.tight_layout(); plt.savefig(out/'top20_gradient_input_importance.png', dpi=220); plt.close()
    plt.figure(figsize=(8,4)); sns.barplot(data=group_df, x='semantic_group', y='mean_abs_gradient_input'); plt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.savefig(out/'semantic_group_importance.png', dpi=220); plt.close()
    top_cols = top['feature'].tolist()
    heat = pd.DataFrame(signed_attr[: min(80, len(df)), [ckpt['columns'].index(c) for c in top_cols]], columns=top_cols)
    plt.figure(figsize=(12,6)); sns.heatmap(heat, cmap='vlag', center=0); plt.tight_layout(); plt.savefig(out/'sample_attribution_heatmap_top20.png', dpi=220); plt.close()
    np.save(out/'gradient_input_attributions_sample.npy', signed_attr[: min(2048, len(signed_attr))])

    shap_status = {'attempted': bool(run_deepshap), 'available': False, 'method': 'shap.DeepExplainer', 'error': None}
    try:
        if not run_deepshap:
            raise RuntimeError('DeepSHAP skipped for this run; gradient-input XAI artifacts were generated.')
        import shap
        shap_status['available'] = True
        wrapper = FlatSparseGuardWrapper(model, ckpt['slices']).to(device).eval()
        background = torch.tensor(df[ckpt['columns']].head(background_n).astype('float32').values, device=device)
        eval_x = torch.tensor(df[ckpt['columns']].iloc[background_n:background_n + min(512, len(df)-background_n)].astype('float32').values, device=device)
        explainer = shap.DeepExplainer(wrapper, background)
        shap_values = explainer.shap_values(eval_x)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)
        np.save(out/'deep_shap_values_sample.npy', shap_values)
        shap_imp = pd.DataFrame({
            'feature': ckpt['columns'],
            'semantic_group': [infer_group(c) for c in ckpt['columns']],
            'mean_abs_deepshap': np.abs(shap_values).mean(axis=0),
            'mean_signed_deepshap': shap_values.mean(axis=0),
        }).sort_values('mean_abs_deepshap', ascending=False)
        shap_imp.to_csv(out/'deep_shap_feature_importance.csv', index=False)
        plt.figure(figsize=(9,6)); sns.barplot(data=shap_imp.head(20), y='feature', x='mean_abs_deepshap', hue='semantic_group', dodge=False); plt.tight_layout(); plt.savefig(out/'top20_deepshap_importance.png', dpi=220); plt.close()
    except Exception as exc:
        shap_status['error'] = repr(exc)

    summary = {
        'sample_n': int(len(df)),
        'background_n': int(background_n),
        'seconds': round(time.perf_counter() - t0, 3),
        'top10_gradient_input_features': feature_df.head(10).to_dict(orient='records'),
        'group_importance': group_df.to_dict(orient='records'),
        'shap_status': shap_status,
        'class_distribution': pd.Series(y_np).value_counts().sort_index().to_dict(),
    }
    write_json(out/'xai_summary.json', summary)
    return summary

def compute_kernel_shap_attributions(sample_n=128, background_n=32, nsamples=128):
    if torch is None: raise RuntimeError('PyTorch is required')
    out = ensure(PROJECT_ROOT/'XAI/results')
    model, ckpt, device = _load_sparseguard_checkpoint()
    df = _load_preprocessed_split('test', ckpt['columns'], sample_n=max(sample_n + background_n, background_n * 2), seed=505)
    background_df = df.head(background_n)
    eval_df = df.iloc[background_n:background_n + sample_n]
    columns = ckpt['columns']

    def predict_prob(arr):
        x = torch.tensor(np.asarray(arr, dtype='float32'), device=device)
        probs = []
        with torch.no_grad():
            for start in range(0, len(x), 256):
                xb = x[start:start+256]
                probs.append(model(to_groups(xb, ckpt['slices']))['prob'].detach().cpu().numpy())
        return np.concatenate(probs)

    t0 = time.perf_counter()
    try:
        import shap
        explainer = shap.KernelExplainer(predict_prob, background_df[columns].astype('float32').values)
        shap_values = explainer.shap_values(eval_df[columns].astype('float32').values, nsamples=nsamples)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.asarray(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values.squeeze(-1)
        np.save(out/'kernel_shap_values_sample.npy', shap_values)
        imp = pd.DataFrame({
            'feature': columns,
            'semantic_group': [infer_group(c) for c in columns],
            'mean_abs_kernel_shap': np.abs(shap_values).mean(axis=0),
            'mean_signed_kernel_shap': shap_values.mean(axis=0),
        }).sort_values('mean_abs_kernel_shap', ascending=False)
        imp.to_csv(out/'kernel_shap_feature_importance.csv', index=False)
        imp.head(30).to_csv(out/'top30_kernel_shap_features.csv', index=False)
        group_imp = imp.groupby('semantic_group', as_index=False)['mean_abs_kernel_shap'].sum().sort_values('mean_abs_kernel_shap', ascending=False)
        group_imp.to_csv(out/'kernel_shap_semantic_group_importance.csv', index=False)
        plt.figure(figsize=(9,6)); sns.barplot(data=imp.head(20), y='feature', x='mean_abs_kernel_shap', hue='semantic_group', dodge=False); plt.tight_layout(); plt.savefig(out/'top20_kernel_shap_importance.png', dpi=220); plt.close()
        plt.figure(figsize=(8,4)); sns.barplot(data=group_imp, x='semantic_group', y='mean_abs_kernel_shap'); plt.xticks(rotation=35, ha='right'); plt.tight_layout(); plt.savefig(out/'kernel_shap_semantic_group_importance.png', dpi=220); plt.close()
        summary = {
            'method': 'KernelSHAP',
            'sample_n': int(len(eval_df)),
            'background_n': int(len(background_df)),
            'nsamples': int(nsamples),
            'seconds': round(time.perf_counter() - t0, 3),
            'top10_kernel_shap_features': imp.head(10).to_dict(orient='records'),
            'group_importance': group_imp.to_dict(orient='records'),
        }
        write_json(out/'kernel_shap_summary.json', summary)
        return summary
    except Exception as exc:
        summary = {'method': 'KernelSHAP', 'sample_n': int(len(eval_df)), 'background_n': int(len(background_df)), 'nsamples': int(nsamples), 'seconds': round(time.perf_counter() - t0, 3), 'error': repr(exc)}
        write_json(out/'kernel_shap_summary.json', summary)
        return summary

def run_sparse_adversarial_attacks(sample_n=4096, steps=12, epsilons=None, topk_values=None):
    if torch is None: raise RuntimeError('PyTorch is required')
    epsilons = epsilons or [0.01, 0.03, 0.05, 0.10]
    topk_values = topk_values or [3, 5, 10, 20]
    out = ensure(PROJECT_ROOT/'ROBUSTNESS/results')
    model, ckpt, device = _load_sparseguard_checkpoint()
    df = _load_preprocessed_split('test', ckpt['columns'], sample_n=sample_n, seed=202)
    X0 = torch.tensor(df[ckpt['columns']].astype('float32').values, device=device)
    y = torch.tensor(df['target'].astype('float32').values, device=device)
    y_np = y.detach().cpu().numpy().astype(int)
    with torch.no_grad():
        clean_prob = model(to_groups(X0, ckpt['slices']))['prob'].detach().cpu().numpy()

    importance_path = PROJECT_ROOT/'XAI/results/gradient_input_feature_importance.csv'
    if importance_path.exists():
        imp = pd.read_csv(importance_path)
        ranked = [c for c in imp['feature'].tolist() if c in ckpt['columns']]
    else:
        ranked = list(ckpt['columns'])
    rows = []
    feature_freq = {c: 0 for c in ckpt['columns']}

    for eps in epsilons:
        alpha = eps / max(1, steps // 2)
        for attack_name, topk in [('PGD', None)] + [('TopK-Masked-PGD', k) for k in topk_values]:
            adv = X0.detach().clone()
            if topk is None:
                mask = torch.ones_like(adv)
                feature_budget = len(ckpt['columns'])
                selected_features = list(ckpt['columns'])
            else:
                selected_features = ranked[:min(topk, len(ranked))]
                feature_budget = len(selected_features)
                idx = [ckpt['columns'].index(c) for c in selected_features]
                mask = torch.zeros_like(adv)
                mask[:, idx] = 1.0
                for c in selected_features:
                    feature_freq[c] += 1
            for _ in range(steps):
                adv = adv.detach().clone().requires_grad_(True)
                o = model(to_groups(adv, ckpt['slices']))
                target = torch.zeros_like(y)
                loss = F.binary_cross_entropy_with_logits(o['logit'], target)
                grad = torch.autograd.grad(loss, adv)[0]
                adv = adv - alpha * grad.sign() * mask
                delta = torch.clamp(adv - X0, -eps, eps) * mask
                adv = torch.clamp(X0 + delta, -6, 6)
            with torch.no_grad():
                adv_prob = model(to_groups(adv, ckpt['slices']))['prob'].detach().cpu().numpy()
            summary = _attack_summary(y_np, clean_prob, adv_prob)
            rows.append({
                'attack': attack_name,
                'epsilon': eps,
                'steps': steps,
                'topk': -1 if topk is None else topk,
                'feature_budget': feature_budget,
                'sample_n': len(df),
                **summary,
            })
            if attack_name == 'TopK-Masked-PGD' and eps == max(epsilons) and topk == max(topk_values):
                pd.DataFrame({'y_true': y_np[:2000], 'clean_prob': clean_prob[:2000], 'adv_prob': adv_prob[:2000]}).to_csv(out/'topk_masked_pgd_prediction_sample.csv', index=False)
    attack_df = pd.DataFrame(rows)
    attack_df.to_csv(out/'attack_success_by_epsilon.csv', index=False)
    freq_df = pd.DataFrame([{'feature': k, 'selection_count': v, 'semantic_group': infer_group(k)} for k, v in feature_freq.items() if v > 0]).sort_values('selection_count', ascending=False)
    freq_df.to_csv(out/'topk_feature_frequency.csv', index=False)
    plt.figure(figsize=(9,5)); sns.lineplot(data=attack_df, x='epsilon', y='attack_to_benign_evasion', hue='attack', style='topk', marker='o'); plt.tight_layout(); plt.savefig(out/'attack_success_curve.png', dpi=220); plt.close()
    topk_df = attack_df[attack_df['attack'] == 'TopK-Masked-PGD']
    if len(topk_df):
        plt.figure(figsize=(8,5)); sns.lineplot(data=topk_df, x='feature_budget', y='attack_to_benign_evasion', hue='epsilon', marker='o', palette='viridis'); plt.tight_layout(); plt.savefig(out/'feature_budget_vs_evasion.png', dpi=220); plt.close()
    write_json(out/'sparse_adversarial_summary.json', {'rows': rows, 'ranked_feature_source': str(importance_path), 'top_selected_features': ranked[:20]})
    return rows

def train_vae_reconstruction_detector(max_epochs=40, batch_size=4096, sample_n_train=240000, sample_n_eval=20000):
    if torch is None: raise RuntimeError('PyTorch is required')
    out = ensure(PROJECT_ROOT/'ROBUSTNESS/results')
    model, ckpt, device = _load_sparseguard_checkpoint()
    train_df = _load_preprocessed_split('train', ckpt['columns'], sample_n=sample_n_train, seed=303)
    test_df = _load_preprocessed_split('test', ckpt['columns'], sample_n=sample_n_eval, seed=304)
    benign_train = train_df[train_df['target'] == 0]
    if len(benign_train) == 0:
        raise ValueError('No benign rows available for VAE detector training.')
    train_tensor = torch.tensor(benign_train[ckpt['columns']].astype('float32').values, device=device)
    ds = torch.utils.data.TensorDataset(train_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)
    vae = TabularVAE(len(ckpt['columns'])).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3, weight_decay=1e-4)
    hist = []
    for epoch in range(1, max_epochs + 1):
        t0 = time.perf_counter(); vae.train(); losses = []
        for (xb,) in loader:
            recon, mu, logvar = vae(xb)
            recon_loss = F.mse_loss(recon, xb, reduction='mean')
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + 0.001 * kl
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss.detach().cpu()))
        hist.append({'epoch': epoch, 'seconds': time.perf_counter() - t0, 'vae_loss': float(np.mean(losses))})
    pd.DataFrame(hist).to_csv(out/'vae_training_history.csv', index=False)
    torch.save({'state_dict': vae.state_dict(), 'columns': ckpt['columns']}, out/'vae_reconstruction_detector.pt')

    eval_X = torch.tensor(test_df[ckpt['columns']].astype('float32').values, device=device)
    y_np = test_df['target'].astype(int).values
    with torch.no_grad():
        clean_prob = model(to_groups(eval_X, ckpt['slices']))['prob'].detach().cpu().numpy()
    eps = 0.10; steps = 12; alpha = eps / 6
    adv = eval_X.detach().clone()
    for _ in range(steps):
        adv = adv.detach().clone().requires_grad_(True)
        o = model(to_groups(adv, ckpt['slices']))
        target = torch.zeros_like(torch.tensor(y_np, dtype=torch.float32, device=device))
        loss = F.binary_cross_entropy_with_logits(o['logit'], target)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = torch.clamp(eval_X + torch.clamp(adv - alpha * grad.sign() - eval_X, -eps, eps), -6, 6)

    vae.eval()
    with torch.no_grad():
        clean_recon, _, _ = vae(eval_X)
        adv_recon, _, _ = vae(adv)
        clean_err = ((clean_recon - eval_X) ** 2).mean(dim=1).detach().cpu().numpy()
        adv_err = ((adv_recon - adv) ** 2).mean(dim=1).detach().cpu().numpy()
        adv_prob = model(to_groups(adv, ckpt['slices']))['prob'].detach().cpu().numpy()
    benign_err = clean_err[y_np == 0]
    threshold = float(np.quantile(benign_err, 0.95)) if len(benign_err) else float(np.quantile(clean_err, 0.95))
    rows = []
    for name, err in [('clean', clean_err), ('pgd_eps_0.10', adv_err)]:
        detected = err > threshold
        rows.append({
            'condition': name,
            'sample_n': len(err),
            'threshold_benign_95pct': threshold,
            'mean_reconstruction_error': float(np.mean(err)),
            'median_reconstruction_error': float(np.median(err)),
            'attack_detection_rate': float(detected[y_np == 1].mean()) if (y_np == 1).any() else float('nan'),
            'benign_false_positive_rate': float(detected[y_np == 0].mean()) if (y_np == 0).any() else float('nan'),
        })
    det_df = pd.DataFrame(rows)
    det_df.to_csv(out/'vae_reconstruction_detection_metrics.csv', index=False)
    pd.DataFrame({'target': y_np, 'clean_prob': clean_prob, 'adv_prob': adv_prob, 'clean_reconstruction_error': clean_err, 'adv_reconstruction_error': adv_err}).head(10000).to_csv(out/'vae_reconstruction_error_sample.csv', index=False)
    plt.figure(figsize=(9,5)); sns.kdeplot(clean_err, label='clean', fill=True); sns.kdeplot(adv_err, label='PGD eps=0.10', fill=True); plt.axvline(threshold, color='red', linestyle='--', label='benign 95% threshold'); plt.legend(); plt.tight_layout(); plt.savefig(out/'reconstruction_error_distribution.png', dpi=220); plt.close()
    plt.figure(figsize=(9,5)); h=pd.DataFrame(hist); sns.lineplot(data=h, x='epoch', y='vae_loss'); plt.tight_layout(); plt.savefig(out/'vae_training_curve.png', dpi=220); plt.close()
    summary = {'epochs': len(hist), 'train_benign_rows': int(len(benign_train)), 'eval_rows': int(len(test_df)), 'threshold_benign_95pct': threshold, 'rows': rows, 'pgd_attack_summary': _attack_summary(y_np, clean_prob, adv_prob)}
    write_json(out/'vae_reconstruction_summary.json', summary)
    return summary

def profile_sparseguard_runtime(sample_n=8192, repeats=20):
    if torch is None: raise RuntimeError('PyTorch is required')
    out = ensure(PROJECT_ROOT/'PROFILING/results')
    model, ckpt, device = _load_sparseguard_checkpoint()
    df = _load_preprocessed_split('test', ckpt['columns'], sample_n=sample_n, seed=404)
    X = torch.tensor(df[ckpt['columns']].astype('float32').values, device=device)
    params = sum(p.numel() for p in model.parameters())
    memory_mb = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(to_groups(X, ckpt['slices']))['prob']
        if device.type == 'cuda':
            torch.cuda.synchronize()
    times = []
    energy_rows = []
    for _ in range(repeats):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(to_groups(X, ckpt['slices']))['prob']
        if device.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        estimated_watts = 70.0 if device.type == 'cuda' else 35.0
        energy_rows.append({'seconds': dt, 'estimated_watts': estimated_watts, 'estimated_joules': dt * estimated_watts})
    if device.type == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    approx_flops_per_sample = 2 * params
    rows = [{
        'device': str(device),
        'sample_n': len(df),
        'repeats': repeats,
        'parameters': int(params),
        'approx_flops_per_sample': int(approx_flops_per_sample),
        'approx_total_flops_per_forward': int(approx_flops_per_sample * len(df)),
        'mean_forward_seconds': float(np.mean(times)),
        'std_forward_seconds': float(np.std(times)),
        'samples_per_second': float(len(df) / np.mean(times)),
        'peak_memory_mb': None if memory_mb is None else float(memory_mb),
        'mean_estimated_joules': float(np.mean([r['estimated_joules'] for r in energy_rows])),
    }]
    pd.DataFrame(rows).to_csv(out/'runtime_energy_flops_profile.csv', index=False)
    pd.DataFrame(energy_rows).to_csv(out/'energy_proxy_iterations.csv', index=False)
    plt.figure(figsize=(7,4)); sns.lineplot(x=list(range(1, repeats+1)), y=times, marker='o'); plt.xlabel('repeat'); plt.ylabel('forward seconds'); plt.tight_layout(); plt.savefig(out/'inference_latency_repeats.png', dpi=220); plt.close()
    write_json(out/'profiling_summary.json', {'profile': rows[0], 'note': 'FLOPs are an analytical linear-layer proxy; energy is a transparent wattage proxy because Colab does not expose direct power telemetry.'})
    return rows[0]

def run_remaining_q1_sections():
    started = time.time()
    report = {'started_at': time.strftime('%Y-%m-%d %H:%M:%S'), 'stages': []}
    stages = [
        ('XAI', compute_xai_attributions),
        ('SPARSE_ADVERSARIAL_ROBUSTNESS', run_sparse_adversarial_attacks),
        ('VAE_RECONSTRUCTION_DETECTOR', train_vae_reconstruction_detector),
        ('PROFILING', profile_sparseguard_runtime),
    ]
    for name, fn in stages:
        t0 = time.time()
        result = fn()
        report['stages'].append({'stage': name, 'seconds': round(time.time() - t0, 3), 'result_type': type(result).__name__})
        write_json(PROJECT_ROOT/'logs'/'remaining_q1_progress.json', report)
    report['total_seconds'] = round(time.time() - started, 3)
    write_json(PROJECT_ROOT/'logs'/'remaining_q1_complete.json', report)
    return report

def _sample_by_label(df: pd.DataFrame, label_col: str, benign_values: list, max_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    y = to_binary(df[label_col], benign_values)
    per_class = max(1, max_rows // 2)
    pieces = []
    for cls in [0, 1]:
        part = df.loc[y == cls]
        if len(part):
            pieces.append(part.sample(n=min(per_class, len(part)), random_state=seed + cls))
    if not pieces:
        return df.sample(n=min(max_rows, len(df)), random_state=seed)
    return pd.concat(pieces, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

def load_x_iiotid_external_frame(max_rows: int = 240000) -> pd.DataFrame:
    spec = DATASETS['x_iiotid']
    df = pd.read_csv(DATASET_ROOT/spec['file'], low_memory=False)
    df = normalize_columns(df)
    label = find_label(df, spec['labels'])
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any').drop_duplicates()
    return _sample_by_label(df, label, spec['benign'], max_rows, seed=11)

def load_cic_ids2017_frame(max_rows: int = 240000, chunksize: int = 120000) -> pd.DataFrame:
    spec = DATASETS['cic_ids2017']
    path = DATASET_ROOT / spec['file']
    parts, counts = [], {0: 0, 1: 0}
    target_each = max_rows // 2
    with zipfile.ZipFile(path) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith('.csv')]
        for csv_name in csv_names:
            with zf.open(csv_name) as fh:
                for chunk in pd.read_csv(fh, chunksize=chunksize, low_memory=False):
                    chunk = normalize_columns(chunk)
                    label = find_label(chunk, spec['labels'])
                    y = to_binary(chunk[label], spec['benign'])
                    take = []
                    for cls in [0, 1]:
                        need = target_each - counts[cls]
                        if need <= 0:
                            continue
                        idx = y[y == cls].index
                        if len(idx):
                            chosen = pd.Index(idx).to_series().sample(n=min(need, len(idx)), random_state=42 + len(parts) + cls).values
                            take.extend(chosen)
                            counts[cls] += len(chosen)
                    if take:
                        parts.append(chunk.loc[take])
                    if counts[0] >= target_each and counts[1] >= target_each:
                        return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)
    if not parts:
        raise ValueError('No CIC-IDS2017 rows were loaded from MachineLearningCSV.zip.')
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)

def load_cic_iiot_2025_frame(max_rows: int = 240000) -> pd.DataFrame:
    spec = DATASETS['cic_iiot_2025']
    path = DATASET_ROOT / spec['file']
    rows_per_member = max(500, max_rows // 20)
    parts = []
    with tarfile.open(path, mode='r:xz') as outer:
        members = [m for m in outer.getmembers() if m.isfile() and m.name.endswith('.csv.tar.xz')]
        for member in sorted(members, key=lambda m: m.name):
            outer_fh = outer.extractfile(member)
            if outer_fh is None:
                continue
            nested_bytes = outer_fh.read()
            with tarfile.open(fileobj=io.BytesIO(nested_bytes), mode='r:xz') as inner:
                csv_members = [m for m in inner.getmembers() if m.isfile() and m.name.lower().endswith('.csv')]
                if not csv_members:
                    continue
                csv_member = csv_members[0]
                inner_fh = inner.extractfile(csv_member)
                if inner_fh is None:
                    continue
                df = pd.read_csv(inner_fh, nrows=rows_per_member, low_memory=False)
                df = normalize_columns(df)
                if 'label1' not in df.columns:
                    df['label1'] = 'attack' if 'attack_data' in member.name else 'benign'
                df['source_window'] = Path(member.name).stem.replace('.csv', '')
                parts.append(df)
    if not parts:
        raise ValueError('No CIC-IIoT-2025 nested CSV rows were loaded.')
    frame = pd.concat(parts, axis=0, ignore_index=True)
    label = find_label(frame, spec['labels'])
    return _sample_by_label(frame, label, spec['benign'], max_rows, seed=23)

def _fit_eval_semantic_classifier(train_X, train_y, test_X, test_y, random_state=42):
    t0 = time.perf_counter()
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_X)
    Xte = scaler.transform(test_X)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=0.02,
        early_stopping=True,
        random_state=random_state,
    )
    model.fit(Xtr, train_y)
    train_seconds = time.perf_counter() - t0
    t1 = time.perf_counter()
    prob = model.predict_proba(Xte)[:, 1]
    infer_seconds = time.perf_counter() - t1
    m = metrics(test_y, prob)
    return m, {'train_seconds': train_seconds, 'inference_seconds': infer_seconds, 'iterations': int(getattr(model, 'n_iter_', 0))}

def run_cross_dataset_validation(max_rows_per_dataset: int = 240000):
    out = ensure(PROJECT_ROOT/'EVALUATION/results')
    loaders = {
        'X-IIoTID': (load_x_iiotid_external_frame, DATASETS['x_iiotid']),
        'CIC-IIoT-2025': (load_cic_iiot_2025_frame, DATASETS['cic_iiot_2025']),
        'CIC-IDS2017': (load_cic_ids2017_frame, DATASETS['cic_ids2017']),
    }
    frames, manifests = {}, []
    for name, (loader, spec) in loaders.items():
        t0 = time.perf_counter()
        raw = loader(max_rows_per_dataset)
        X, y, manifest = semantic_aggregate_dataset(raw, spec['labels'], spec['benign'], name)
        manifest['load_and_aggregate_seconds'] = round(time.perf_counter() - t0, 3)
        manifests.append(manifest)
        frames[name] = X.assign(target=y.values, dataset=name)
    pd.DataFrame(manifests).to_csv(out/'external_dataset_manifest.csv', index=False)
    pd.concat([df.head(30) for df in frames.values()], axis=0).to_csv(out/'external_semantic_feature_sample.csv', index=False)

    rows, cms = [], {}
    for name, df in frames.items():
        X = df.drop(columns=['target', 'dataset'])
        y = df['target'].astype(int)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.30, random_state=42, stratify=y)
        m, timing = _fit_eval_semantic_classifier(Xtr, ytr, Xte, yte, random_state=42)
        rows.append({'protocol': 'within_dataset', 'train_dataset': name, 'test_dataset': name, **{k:v for k,v in m.items() if isinstance(v, (int, float))}, **timing, 'train_rows': len(Xtr), 'test_rows': len(Xte)})
        cms[('within_dataset', name, name)] = m['confusion_matrix']

    source = frames['X-IIoTID']
    src_X = source.drop(columns=['target', 'dataset'])
    src_y = source['target'].astype(int)
    src_Xtr, _, src_ytr, _ = train_test_split(src_X, src_y, test_size=.30, random_state=42, stratify=src_y)
    for target_name in ['CIC-IIoT-2025', 'CIC-IDS2017']:
        target = frames[target_name]
        m, timing = _fit_eval_semantic_classifier(src_Xtr, src_ytr, target.drop(columns=['target', 'dataset']), target['target'].astype(int), random_state=61)
        rows.append({'protocol': 'x_iiotid_to_external', 'train_dataset': 'X-IIoTID', 'test_dataset': target_name, **{k:v for k,v in m.items() if isinstance(v, (int, float))}, **timing, 'train_rows': len(src_Xtr), 'test_rows': len(target)})
        cms[('x_iiotid_to_external', 'X-IIoTID', target_name)] = m['confusion_matrix']

    combined = pd.concat(frames.values(), axis=0, ignore_index=True)
    strat = combined['dataset'].astype(str) + '_' + combined['target'].astype(str)
    combined_train, combined_test = train_test_split(combined, test_size=.30, random_state=42, stratify=strat)
    m, timing = _fit_eval_semantic_classifier(
        combined_train.drop(columns=['target', 'dataset']),
        combined_train['target'].astype(int),
        combined_test.drop(columns=['target', 'dataset']),
        combined_test['target'].astype(int),
        random_state=67,
    )
    rows.append({'protocol': 'mixed_multi_dataset_holdout', 'train_dataset': 'X-IIoTID+CIC-IIoT-2025+CIC-IDS2017', 'test_dataset': 'stratified_all_datasets', **{k:v for k,v in m.items() if isinstance(v, (int, float))}, **timing, 'train_rows': len(combined_train), 'test_rows': len(combined_test)})
    cms[('mixed_multi_dataset_holdout', 'all', 'all')] = m['confusion_matrix']

    for target_name in frames:
        target = frames[target_name]
        target_y = target['target'].astype(int)
        target_cal, target_test = train_test_split(target, train_size=.05, random_state=88, stratify=target_y)
        source_train = pd.concat([df for name, df in frames.items() if name != target_name], axis=0, ignore_index=True)
        adapted_train = pd.concat([source_train, target_cal], axis=0, ignore_index=True)
        m, timing = _fit_eval_semantic_classifier(
            adapted_train.drop(columns=['target', 'dataset']),
            adapted_train['target'].astype(int),
            target_test.drop(columns=['target', 'dataset']),
            target_test['target'].astype(int),
            random_state=89,
        )
        rows.append({'protocol': 'few_shot_target_adaptation_5pct', 'train_dataset': f'other_datasets+5pct_{target_name}', 'test_dataset': target_name, **{k:v for k,v in m.items() if isinstance(v, (int, float))}, **timing, 'train_rows': len(adapted_train), 'test_rows': len(target_test)})
        cms[('few_shot_target_adaptation_5pct', 'adapted', target_name)] = m['confusion_matrix']

    for heldout_name in frames:
        train_df = pd.concat([df for name, df in frames.items() if name != heldout_name], axis=0, ignore_index=True)
        test_df = frames[heldout_name]
        m, timing = _fit_eval_semantic_classifier(
            train_df.drop(columns=['target', 'dataset']),
            train_df['target'].astype(int),
            test_df.drop(columns=['target', 'dataset']),
            test_df['target'].astype(int),
            random_state=71,
        )
        rows.append({'protocol': 'leave_one_dataset_out', 'train_dataset': '+'.join([n for n in frames if n != heldout_name]), 'test_dataset': heldout_name, **{k:v for k,v in m.items() if isinstance(v, (int, float))}, **timing, 'train_rows': len(train_df), 'test_rows': len(test_df)})
        cms[('leave_one_dataset_out', 'others', heldout_name)] = m['confusion_matrix']

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(out/'external_validation_metrics.csv', index=False)
    runtime_cols = ['protocol', 'train_dataset', 'test_dataset', 'train_rows', 'test_rows', 'train_seconds', 'inference_seconds', 'iterations']
    metrics_df[runtime_cols].to_csv(out/'external_runtime_profile.csv', index=False)

    heat = metrics_df.pivot_table(index='protocol', columns='test_dataset', values='f1', aggfunc='max')
    plt.figure(figsize=(9,4.5)); sns.heatmap(heat, annot=True, fmt='.4f', cmap='viridis', vmin=0, vmax=1); plt.tight_layout(); plt.savefig(out/'cross_dataset_f1_heatmap.png', dpi=220); plt.close()

    fig, axes = plt.subplots(1, min(4, len(cms)), figsize=(14,3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for ax, (key, cm) in zip(axes, list(cms.items())[:4]):
        sns.heatmap(np.array(cm), annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_title('\\n'.join([key[0], f'{key[1]} -> {key[2]}']), fontsize=8)
        ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    plt.tight_layout(); plt.savefig(out/'cross_dataset_confusion_matrices.png', dpi=220); plt.close()

    write_json(out/'external_validation_summary.json', {
        'method': 'semantic aggregate schema adapter over X-IIoTID, CIC-IIoT-2025, and CIC-IDS2017',
        'datasets': manifests,
        'protocols': ['within_dataset', 'x_iiotid_to_external', 'mixed_multi_dataset_holdout', 'few_shot_target_adaptation_5pct', 'leave_one_dataset_out'],
        'rows': rows,
    })
    return rows

def write_ablation_plan():
    out=ensure(PROJECT_ROOT/'ABLATION/results')
    plan=[{'name':'full_sparseguard','semantic_branches':1,'attention':1,'reconstruction':1,'semantic_validation':1},{'name':'no_attention','semantic_branches':1,'attention':0,'reconstruction':1,'semantic_validation':1},{'name':'no_reconstruction','semantic_branches':1,'attention':1,'reconstruction':0,'semantic_validation':1},{'name':'no_semantic_validation','semantic_branches':1,'attention':1,'reconstruction':1,'semantic_validation':0},{'name':'single_path_mlp','semantic_branches':0,'attention':0,'reconstruction':0,'semantic_validation':0}]
    write_json(out/'ablation_plan.json', {'variants': plan})

def write_robustness_plan():
    out=ensure(PROJECT_ROOT/'ROBUSTNESS/results')
    write_json(out/'robustness_plan.json', {'attacks':['FGSM','PGD','TopK-Masked-PGD'], 'epsilons':[.01,.03,.05,.1], 'topk':[3,5,10,20], 'outputs':['attack_success_by_epsilon.csv','feature_budget_vs_evasion.png','reconstruction_error_distribution.png','topk_feature_frequency.csv']})

def write_external_validation_plan():
    out=ensure(PROJECT_ROOT/'EVALUATION/results')
    write_json(out/'external_validation_plan.json', {'train':'X-IIoTID','external':['CIC-IIoT-2025','CIC-IDS2017'], 'method':'semantic_group_adapter_binary_validation', 'metrics':['accuracy','balanced_accuracy','precision','recall','f1','roc_auc','mcc','brier']})

def write_paper_asset_plan():
    out=ensure(PROJECT_ROOT/'PAPER_ASSETS/results')
    write_json(out/'paper_asset_plan.json', {'tables':['dataset_statistics_table.csv','main_performance_table.csv','cross_dataset_table.csv','ablation_table.csv','robustness_table.csv','runtime_resource_table.csv'], 'figures':['architecture_diagram.png','training_curves.png','feature_importance.png','attack_success_curve.png','cross_dataset_curves.png']})

def run_full_project():
    """Run the full project sequentially.

    This is intentionally conservative: expensive model training starts only after
    EDA and preprocessing have created verifiable outputs.
    """
    started = time.time()
    report = {'project_root': str(PROJECT_ROOT), 'dataset_root': str(DATASET_ROOT), 'stages': []}
    stages = [
        ('EDA', eda_x_iiotid),
        ('PREPROCESSING', preprocess_x_iiotid),
        ('EXPERIMENT', train_sparseguard),
        ('ABLATION', write_ablation_plan),
        ('ROBUSTNESS', write_robustness_plan),
        ('EVALUATION', run_cross_dataset_validation),
        ('PAPER_ASSETS', write_paper_asset_plan),
    ]
    for name, fn in stages:
        t0 = time.time()
        fn()
        report['stages'].append({'stage': name, 'seconds': round(time.time() - t0, 3)})
        write_json(PROJECT_ROOT / 'logs' / 'full_project_progress.json', report)
    report['total_seconds'] = round(time.time() - started, 3)
    write_json(PROJECT_ROOT / 'logs' / 'full_project_complete.json', report)
    return report
