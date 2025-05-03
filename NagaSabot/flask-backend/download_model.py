import gdown
import os

# Google Drive file ID and destination path
FILE_ID = "18SQXPqR-WK1xJPSQtPEXW08KIy-eacnJ"
DEST_PATH = "nagsabot_full_model_morecleaner4.keras"

if not os.path.exists(DEST_PATH):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    print(f"Downloading model from {url} ...")
    gdown.download(url, DEST_PATH, quiet=False)
else:
    print("Model file already exists, skipping download.") 