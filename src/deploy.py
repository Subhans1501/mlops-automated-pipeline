import os
from huggingface_hub import HfApi
def deploy_to_huggingface():

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("No Hugging Face token found. Skipping deployment.")
        return
    api = HfApi()
    repo_id = "subhan1501/Phishing-Detector"
    print(f"Deploying model to {repo_id}...")

    try:
        api.create_repo(repo_id=repo_id,token=hf_token,exist_ok=True)
    except Exception as e:
        print(f"Repo creation note: {e}")
    api.upload_folder(
        folder_path="src",
        repo_id=repo_id,
        repo_type="model",
        token=hf_token
    )
    print("Deployment successful!")
if __name__ == "__main__":
    deploy_to_huggingface()