from huggingface_hub import snapshot_download
import logging
import sys

# Configure logging to provide clear feedback
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def download_model(repo_id: str):
    """
    Downloads a model from the Hugging Face Hub to the local cache.
    Includes patterns to ignore legacy/duplicate weight files to save disk space.
    """
    logging.info(f"Starting download for {repo_id}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            # These patterns help ignore redundant or legacy file formats, saving disk space.
            ignore_patterns=["*.bin", "*.h5", "*.msgpack"],
        )
        logging.info(f"Successfully downloaded {repo_id}.")
    except Exception as e:
        logging.error(f"Failed to download {repo_id}. Error: {e}")
        logging.error(
            "Please ensure you have accepted the license on the Hugging Face Hub "
            "and are logged in via `huggingface-cli login`."
        )


def main():
    """
    Main function to download all required models for the project.
    """
    # A list of all model repository IDs used in your project (from probe/utils.py)
    model_repos = [
        "google/gemma-3-4b-it",
        # "Qwen/Qwen3-4B-Instruct-2507",
        # "meta-llama/Llama-3.2-3B-Instruct",
    ]

    print("\n--- Starting Model Download Process ---")
    print("This will download models to your local Hugging Face cache.")
    print(
        "Make sure you have run 'huggingface-cli login' and accepted model licenses on the Hub.\n"
    )

    for repo in model_repos:
        download_model(repo)

    print("\n--- All model downloads attempted. You can now run your scripts. ---")


if __name__ == "__main__":
    main()
