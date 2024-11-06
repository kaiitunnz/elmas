"""
Adapted from https://github.com/metauto-ai/GPTSwarm.
"""
import requests
import tarfile
from pathlib import Path


def download_mmlu():

    this_file_path = Path(__file__).resolve().parent
    tar_path = this_file_path.with_name("data.tar")
    if not tar_path.exists():
        url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"
        print(f"Downloading {url}")
        r = requests.get(url, allow_redirects=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {tar_path}")

    data_path = this_file_path / "data"
    if not data_path.exists():
        tar = tarfile.open(tar_path)
        tar.extractall(this_file_path)
        tar.close()
        print(f"Saved to {data_path}")


if __name__ == "__main__":
    download_mmlu()