import requests, zipfile, os

url = "http://www.iro.umontreal.ca/~lisa/icml2007data/mnist_rotation_new.zip"
zip_path = "mnist_rotation_new.zip"
out_dir = "src/datasets_utils/mnist_rotation_new"

os.makedirs(out_dir, exist_ok=True)

# Download if not already present
if not os.path.exists(zip_path):
    print("Downloading dataset...")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
else:
    print("Zip file already exists, skipping download.")

# Extract with Python
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(out_dir)

print("Done! Files in:", out_dir)
