
import os
import sys
import zipfile
import argparse

# Huggingface dataset (public): 10k maps, 100 trajectories each
HF_DATASET = "jianwei-liu93/DiPPeR-Diffusion_2D_planner_training_dataset_example"
HF_FILE = "data_20230908-113959_10k_maps_100_traj.zip"


def download_dataset(output_dir: str = "./data") -> str:
    """Download DiPPeR training dataset from Huggingface. Returns path to data folder."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Install: pip install huggingface_hub")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, HF_FILE)

    if os.path.exists(zip_path):
        print(f"Dataset zip already at {zip_path}")
    else:
        print(f"Downloading {HF_DATASET} from Huggingface...")
        zip_path = hf_hub_download(
            repo_id=HF_DATASET,
            filename=HF_FILE,
            repo_type="dataset",
            local_dir=output_dir,
        )
        print(f"Downloaded to {zip_path}")

    # Unzip
    data_folder = os.path.join(output_dir, "data_20230908-113959_10k_maps_100_traj")
    if not os.path.exists(data_folder):
        print(f"Extracting to {data_folder}...")
        with zipfile.ZipFile(zip_path if os.path.isfile(zip_path) else os.path.join(output_dir, HF_FILE), "r") as z:
            z.extractall(output_dir)
        print("Extraction complete.")
    else:
        print(f"Data already extracted at {data_folder}")

    return data_folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./data", help="Where to save dataset")
    parser.add_argument("--train", action="store_true", help="Run training after download")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (full: 100)")
    args = parser.parse_args()

    data_path = download_dataset(args.output_dir)

    if args.train:
        print("\n--- Training ---")
        print("Run the training notebook with dataset_path =", repr(data_path))
        print("Or use: jupyter notebook ../diffusion_policy_2D_planner_training_example_FiLM_goal_condition.ipynb")
        print("\nIn the notebook, set:")
        print(f"  dataset_path = '{os.path.abspath(data_path)}/'")
        print("  num_epochs =", args.epochs)
        print("\nCheckpoints save to ./checkpoints/ - use the .pth file with run_showcase.py --ckpt")


if __name__ == "__main__":
    main()
