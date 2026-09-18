from huggingface_hub import login, upload_folder

# 1. Connexion (un prompt interactif s'ouvrira dans le terminal pour coller votre token)
login()

# 2. Upload des fichiers du dataset vers Hugging Face
print("🚀 Début de l'envoi vers Hugging Face...")
upload_folder(
    folder_path=".",
    repo_id="Ramseyn-77/plant-disease-crop-health-15classes",
    repo_type="dataset",
    ignore_patterns=["*.py", "*.pt", "*.ipynb", "__pycache__/*", "dataset_seg/*"]  # Exclut les scripts de code locaux
)

print("🎉 Upload terminé avec succès !")
