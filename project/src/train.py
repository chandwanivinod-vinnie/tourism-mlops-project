
from pathlib import Path
import json
import os
import pickle
import warnings
from datetime import datetime, timezone
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings(
    'ignore',
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
 )

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
root = Path(__file__).resolve().parents[1]
models_dir = root / 'models'; models_dir.mkdir(parents=True, exist_ok=True)
exp_dir = root / 'experiments'; exp_dir.mkdir(parents=True, exist_ok=True)
target = 'ProdTaken'

hf_repo = os.getenv('HF_DATASET_REPO', '')
hf_token = os.getenv('HF_TOKEN', '')
hf_model = os.getenv('HF_MODEL_REPO', '')

train_df = None
test_df = None
if hf_repo:
    try:
        ds = load_dataset('csv', data_files={'train': f'hf://datasets/{hf_repo}/processed/train.csv', 'test': f'hf://datasets/{hf_repo}/processed/test.csv'})
        train_df = ds['train'].to_pandas(); test_df = ds['test'].to_pandas()
    except Exception:
        pass
if train_df is None:
    train_df = pd.read_csv(root / 'data' / 'processed' / 'train.csv')
    test_df = pd.read_csv(root / 'data' / 'processed' / 'test.csv')

X_train = train_df.drop(columns=[target]); y_train = train_df[target].astype(int)
X_test = test_df.drop(columns=[target]); y_test = test_df[target].astype(int)

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
pre = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

candidates = {
    'RandomForest': {
        'estimator': RandomForestClassifier(random_state=42, n_jobs=1),
        'params': {
            'classifier__n_estimators': [200, 300, 500],
            'classifier__max_depth': [None, 8, 12, 16],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4],
            'classifier__max_features': ['sqrt', 'log2', None],
        },
        'n_iter': 15,
    },
    'GradientBoosting': {
        'estimator': GradientBoostingClassifier(random_state=42),
        'params': {
            'classifier__n_estimators': [100, 150, 200],
            'classifier__learning_rate': [0.03, 0.05, 0.1],
            'classifier__max_depth': [2, 3, 4],
            'classifier__subsample': [0.7, 0.85, 1.0],
        },
        'n_iter': 12,
    },
}

results = {}
for name, cfg in candidates.items():
    pipe = Pipeline([('preprocessor', pre), ('classifier', cfg['estimator'])])
    search = RandomizedSearchCV(pipe, cfg['params'], n_iter=cfg['n_iter'], scoring='roc_auc', cv=cv, n_jobs=1, random_state=42)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    pred = best.predict(X_test)
    proba = best.predict_proba(X_test)[:, 1]
    results[name] = {
        'best_estimator': best,
        'best_params': search.best_params_,
        'best_cv_score_roc_auc': float(search.best_score_),
        'test_metrics': {
            'accuracy': float(accuracy_score(y_test, pred)),
            'precision': float(precision_score(y_test, pred)),
            'recall': float(recall_score(y_test, pred)),
            'f1': float(f1_score(y_test, pred)),
            'roc_auc': float(roc_auc_score(y_test, proba)),
        },
    }

best_name = max(results, key=lambda n: results[n]['test_metrics']['roc_auc'])
best_result = results[best_name]
model_path = models_dir / 'best_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(best_result['best_estimator'], f)

schema = {'feature_columns': X_train.columns.tolist(), 'selected_model': best_name}
with open(models_dir / 'inference_schema.json', 'w', encoding='utf-8') as f:
    json.dump(schema, f, indent=2)

with open(exp_dir / 'metrics.json', 'w', encoding='utf-8') as f:
    json.dump({
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'selected_model': best_name,
        'all_model_results': {
            k: {
                'best_params': v['best_params'],
                'best_cv_score_roc_auc': v['best_cv_score_roc_auc'],
                'test_metrics': v['test_metrics'],
            } for k, v in results.items()
        },
    }, f, indent=2)

if hf_model and hf_token:
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_model, repo_type='model', exist_ok=True, private=False)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo='best_model.pkl', repo_id=hf_model, repo_type='model', token=hf_token)
    api.upload_file(path_or_fileobj=str(models_dir / 'inference_schema.json'), path_in_repo='inference_schema.json', repo_id=hf_model, repo_type='model', token=hf_token)

