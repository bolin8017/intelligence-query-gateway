#!/usr/bin/env python3
"""Upload trained model to Hugging Face Hub.

This script simplifies uploading your trained SemanticRouter model to
Hugging Face Hub for easy distribution and auto-download.

Usage:
    python scripts/upload_model_to_hub.py

Requirements:
    - You must be logged in: huggingface-cli login
    - The model must exist at ./models/router
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def main() -> None:
    """Upload model to Hugging Face Hub."""
    # Configuration
    model_path = Path("./models/router")
    repo_id = "bolin8017/query-gateway-router"  # Your HF repo ID

    print("=" * 60)
    print("Hugging Face Model Upload Tool")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print(f"Target repo: {repo_id}")
    print()

    # Validate model exists
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print()
        print("Please train the model first:")
        print("  python scripts/train_router.py --output-dir ./models/router")
        sys.exit(1)

    # Check required files (support both safetensors and pytorch formats)
    required_files = ["config.json", "tokenizer_config.json"]
    model_file_variants = ["pytorch_model.bin", "model.safetensors"]

    missing_files = [f for f in required_files if not (model_path / f).exists()]
    has_model_weights = any((model_path / f).exists() for f in model_file_variants)

    if missing_files:
        print(f"❌ Error: Missing required files: {missing_files}")
        print(f"   Found files: {list(model_path.glob('*'))}")
        sys.exit(1)

    if not has_model_weights:
        print(f"❌ Error: No model weights found (need one of: {model_file_variants})")
        print(f"   Found files: {list(model_path.glob('*'))}")
        sys.exit(1)

    print("✓ Model files validated")

    # Show what will be uploaded
    model_files = list(model_path.glob('*'))
    model_files = [f.name for f in model_files if f.is_file()]
    print(f"  Files to upload: {', '.join(sorted(model_files))}")
    print()

    # Confirm upload
    response = input(f"Upload model to {repo_id}? [y/N]: ")
    if response.lower() not in ["y", "yes"]:
        print("Upload cancelled.")
        sys.exit(0)

    try:
        # Initialize HF API
        api = HfApi()

        # Create repo if it doesn't exist
        print(f"Creating/verifying repository: {repo_id}")
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print("✓ Repository ready")
        print()

        # Upload model folder
        print(f"Uploading model from {model_path}...")
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload SemanticRouter model for query classification",
        )

        print()
        print("=" * 60)
        print("✓ Upload successful!")
        print("=" * 60)
        print()
        print(f"Model is now available at:")
        print(f"  https://huggingface.co/{repo_id}")
        print()
        print("Users can now auto-download this model by setting:")
        print(f"  HF_MODEL_ID={repo_id}")
        print()

    except Exception as e:
        print()
        print(f"❌ Upload failed: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Check you're logged in: huggingface-cli whoami")
        print("  2. If not logged in: huggingface-cli login")
        print("  3. Ensure you have write access to the repository")
        sys.exit(1)


if __name__ == "__main__":
    main()
