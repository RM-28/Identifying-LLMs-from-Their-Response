import os
import time
import zipfile
import urllib.request

URL = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
FOLDER = "fastText"
ZIP_PATH = os.path.join(FOLDER, "crawl-300d-2M.vec.zip")
VEC_PATH = os.path.join(FOLDER, "crawl-300d-2M.vec")

start_time = time.time()

# 1. Create target folder if it doesn't exist
os.makedirs(FOLDER, exist_ok=True)

# 2. Check if the unzipped file already exists
if os.path.exists(VEC_PATH):
    print("fastText vectors already exist.")
else:
    # Download the zip file if it's not already downloaded
    if not os.path.exists(ZIP_PATH):
        print("Downloading fastText vectors (~4.5 GB)... This may take several minutes.")
        urllib.request.urlretrieve(URL, ZIP_PATH)
        print("Download complete.")
    else:
        print("Found existing zip file. Skipping download.")

    # Unzip the .vec file
    print("Extracting vectors...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(FOLDER)
    print("Extraction complete.")

    # Optional: Delete the zip file to save ~4.5 GB of disk space
    # os.remove(ZIP_PATH)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")