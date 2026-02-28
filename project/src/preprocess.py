
from pathlib import Path
import os
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from sklearn.model_selection import train_test_split

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
root = Path(__file__).resolve().parents[1]
raw_path = root / 'data' / 'raw' / 'tourism.csv'
processed = root / 'data' / 'processed'
processed.mkdir(parents=True, exist_ok=True)

hf_repo = os.getenv('HF_DATASET_REPO', '')
hf_token = os.getenv('HF_TOKEN', '')
target = 'ProdTaken'

df = None
if hf_repo:
    try:
        df = load_dataset(hf_repo, split='train').to_pandas()
    except Exception:
        pass
if df is None:
    df = pd.read_csv(raw_path)

df.columns = [c.strip() for c in df.columns]
if 'Gender' in df.columns:
    df['Gender'] = df['Gender'].astype(str).str.replace('Fe Male', 'Female', regex=False)
if 'TypeofContact' in df.columns:
    df['TypeofContact'] = df['TypeofContact'].astype(str).str.replace('Self Enquiry', 'Self Inquiry', regex=False)

for c in df.columns:
    if df[c].dtype == 'object':
        mode_v = df[c].mode(dropna=True)
        df[c] = df[c].fillna(mode_v.iloc[0] if not mode_v.empty else 'Unknown')
    else:
        df[c] = df[c].fillna(df[c].median())

df = df.drop_duplicates().drop(columns=[x for x in ['Unnamed: 0', 'CustomerID'] if x in df.columns], errors='ignore')
if {'NumberOfPersonVisiting', 'NumberOfChildrenVisiting'}.issubset(df.columns):
    df['FamilySize'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']

X = df.drop(columns=[target])
y = df[target].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
train_df = X_train.copy(); train_df[target] = y_train.values
test_df = X_test.copy(); test_df[target] = y_test.values

train_path = processed / 'train.csv'
test_path = processed / 'test.csv'
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

if hf_repo and hf_token:
    api = HfApi(token=hf_token)
    api.upload_file(path_or_fileobj=str(train_path), path_in_repo='processed/train.csv', repo_id=hf_repo, repo_type='dataset', token=hf_token)
    api.upload_file(path_or_fileobj=str(test_path), path_in_repo='processed/test.csv', repo_id=hf_repo, repo_type='dataset', token=hf_token)

