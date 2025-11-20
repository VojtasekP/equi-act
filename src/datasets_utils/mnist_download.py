from pathlib import Path
import os
import requests
import zipfile


URL = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
DATA_DIR = Path(__file__).resolve().parent / "mnist_rotation_new"


def download_mnist_rotation(out_dir: str | Path | None = None) -> Path:
    """
    Download and extract the rotated MNIST dataset if it's not already present.
    Returns the path to the extracted directory.
    """
    target_dir = Path(out_dir) if out_dir else DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    train_file = target_dir / "mnist_all_rotation_normalized_float_train_valid.amat"
    test_file = target_dir / "mnist_all_rotation_normalized_float_test.amat"
    if train_file.exists() and test_file.exists():
        print(f"MNIST rotation already present in {target_dir}, skipping download.")
        return target_dir

    zip_path = target_dir.parent / "mnist_rotation_new.zip"
    if not zip_path.exists():
        print("Downloading rotated MNIST...")
        resp = requests.get(URL, stream=True, timeout=60)
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
    else:
        print(f"Using existing zip at {zip_path}")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    # Clean up the archive to save space
    try:
        zip_path.unlink()
    except OSError:
        pass

    print("Done! Files in:", target_dir)
    return target_dir


if __name__ == "__main__":
    download_mnist_rotation()
