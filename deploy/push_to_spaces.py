"""Deploy to Hugging Face Spaces.

Usage:
    python deploy/push_to_spaces.py

Uploads the project (excluding local README.md, .env, heavy data dirs)
then uploads the Space-specific README with correct YAML frontmatter.

The Space README lives at deploy/spaces_readme.md so it is never
accidentally overwritten by the folder upload.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REPO_ID = "fikri0o0/indo-financial-agent"
COMMIT_MSG = sys.argv[1] if len(sys.argv) > 1 else "chore: sync from local"

try:
    from huggingface_hub import HfApi
except ImportError:
    print("huggingface_hub not installed — pip install huggingface-hub")
    sys.exit(1)

api = HfApi()

print(f"Uploading project to {REPO_ID} ...")
api.upload_folder(
    repo_id=REPO_ID,
    repo_type="space",
    folder_path=str(ROOT),
    ignore_patterns=[
        # version control / tooling
        ".git*", "__pycache__", "*.pyc", ".venv", "venv",
        ".pytest_cache", "*.egg-info", "node_modules",
        # data (too large / private)
        "data/vectorstore/*", "data/pdfs/*", "reports/*", "notebooks",
        # secrets
        ".env",
        # project README — Space README uploaded separately below
        "README.md",
    ],
    commit_message=COMMIT_MSG,
)

print("Uploading Space README (with YAML frontmatter) ...")
spaces_readme = ROOT / "deploy" / "spaces_readme.md"
api.upload_file(
    repo_id=REPO_ID,
    repo_type="space",
    path_or_fileobj=spaces_readme.read_bytes(),
    path_in_repo="README.md",
    commit_message=f"{COMMIT_MSG} [README]",
)

print(f"\n✅ Done — https://huggingface.co/spaces/{REPO_ID}")
