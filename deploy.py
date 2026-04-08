from huggingface_hub import HfApi

print("🚀 Starting optimized upload to Hugging Face...")
api = HfApi()

api.upload_folder(
    folder_path=".",
    repo_id="bharath1675/sql-repair-env",
    repo_type="space",
    ignore_patterns=[
        "venv/*", 
        "venv/**", 
        ".venv/*", 
        "outputs/*", 
        "__pycache__/*", 
        "**/__pycache__/*", 
        ".pytest_cache/*", 
        ".git/*", 
        ".env", 
        "deploy.py"
    ]
)
print("✅ Upload complete! Your space is live at: https://huggingface.co/spaces/bharath1675/sql-repair-env")
