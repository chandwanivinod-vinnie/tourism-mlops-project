# Push deployment files to a Hugging Face Space for hosting
import os
from pathlib import Path
from huggingface_hub import HfApi

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_SPACE_REPO = os.getenv("HF_SPACE_REPO", "")

if not HF_TOKEN or not HF_SPACE_REPO:
    raise ValueError("Set HF_TOKEN and HF_SPACE_REPO in environment/.env before pushing.")

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=HF_SPACE_REPO, repo_type="space", exist_ok=True, private=False, space_sdk="docker")

# Ensure README has valid Space metadata so runtime does not enter CONFIG_ERROR
readme_path = Path("project/deployment/README.md")
readme_path.write_text(
    """---
title: Tourism MLOps App
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Tourism MLOps App
Dockerized Streamlit app for tourism package purchase prediction.
""",
    encoding="utf-8",
)

api.upload_file(
    path_or_fileobj="project/deployment/README.md",
    path_in_repo="README.md",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj="project/deployment/app.py",
    path_in_repo="app.py",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj="project/deployment/requirements.txt",
    path_in_repo="requirements.txt",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)
api.upload_file(
    path_or_fileobj="project/deployment/Dockerfile",
    path_in_repo="Dockerfile",
    repo_id=HF_SPACE_REPO,
    repo_type="space",
    token=HF_TOKEN,
)

print(f"Deployment files pushed to HF Space: https://huggingface.co/spaces/{HF_SPACE_REPO}")
