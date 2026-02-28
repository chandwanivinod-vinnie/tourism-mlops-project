"""Push deployment files to a Hugging Face Space."""

import os
from huggingface_hub import HfApi

# Non-secret space configuration is defined in code (not in .env).
HF_SPACE_REPO = "tourism-mlops-app"
HF_TOKEN = os.getenv("HF_TOKEN")
HF_USERNAME = os.getenv("HF_USERNAME")

if not HF_TOKEN or not HF_USERNAME:
    raise ValueError("Set HF_TOKEN and HF_USERNAME before running this script.")

repo_id = f"{HF_USERNAME}/{HF_SPACE_REPO}"
api = HfApi(token=HF_TOKEN)

# HF API currently accepts space SDK values: gradio, docker, static.
# We use docker because deployment includes a Dockerfile that runs Streamlit.
api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", private=False, exist_ok=True)

api.upload_folder(
    folder_path="tourism_project/deployment",
    repo_id=repo_id,
    repo_type="space",
    commit_message="Deploy Streamlit app files via Docker Space"
 )

print(f"Deployment files pushed successfully to HF Space: {repo_id}")
