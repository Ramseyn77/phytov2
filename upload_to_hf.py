"""
upload_to_hf.py
================
Script d'upload automatique du dataset vers Hugging Face.
- Utilise l'upload LFS standard (évite l'erreur CAS/XetHub 401)
- Transmet explicitement le Token d'authentification pour chaque shard
"""

import os
import sys
from pathlib import Path

# Désactiver hf_transfer / XetHub expérimentaux qui causent des erreurs 401 sur le serveur CAS
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import HfApi, login

REPO_ID = "Ramseyn-77/plant-disease-crop-health-15classes"

def main():
    print("🔐 Connexion à Hugging Face...")
    token = input("👉 Colle ton NOUVEAU Token Hugging Face (Write) : ").strip()
    if not token:
        print("❌ Erreur: Le token est obligatoire.")
        sys.exit(1)
        
    login(token=token)

    api = HfApi(token=token)

    print(f"📁 Création / Vérification du dépôt '{REPO_ID}' sur Hugging Face...")
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        exist_ok=True,
        private=False
    )
    print("✅ Dépôt prêt !")

    # 1. Upload de README.md
    readme_path = Path("README.md")
    if readme_path.exists():
        print("📄 Téléversement de README.md (Fiche descriptive)...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token
        )
        print("✅ README.md mis en ligne avec succès.")

    # 2. Upload de dataset.rar
    rar_path = Path("dataset.rar")
    if rar_path.exists():
        size_gb = rar_path.stat().st_size / (1024**3)
        print(f"🚀 Téléversement de dataset.rar ({size_gb:.2f} GB) en cours...")
        api.upload_file(
            path_or_fileobj=str(rar_path),
            path_in_repo="dataset.rar",
            repo_id=REPO_ID,
            repo_type="dataset",
            token=token
        )
        print("🎉 dataset.rar mis en ligne avec succès !")
    else:
        print("⚠️ Le fichier dataset.rar n'a pas été trouvé.")

if __name__ == "__main__":
    main()
