
from pathlib import Path
import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv(Path(__file__).resolve().parents[2] / '.env')
root = Path(__file__).resolve().parents[1]
data_dir = root / 'data'
raw_dir = data_dir / 'raw'
raw_dir.mkdir(parents=True, exist_ok=True)

csv_path = Path(__file__).resolve().parents[2] / 'tourism.csv'
df = pd.read_csv(csv_path)
raw_path = raw_dir / 'tourism.csv'
df.to_csv(raw_path, index=False)

hf_token = os.getenv('HF_TOKEN', '')
hf_repo = os.getenv('HF_DATASET_REPO', '')

if hf_token and hf_repo:
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=hf_repo, repo_type='dataset', exist_ok=True, private=False)
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(hf_repo, token=hf_token)
    print(f'Registered dataset: {hf_repo}')
else:
    print('HF credentials missing; dataset upload skipped.')

