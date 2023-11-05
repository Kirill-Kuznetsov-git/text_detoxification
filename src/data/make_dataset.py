import requests, zipfile, io
from pathlib import Path

ROW_DATA_ZIP_URL = "https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip"
PROJECT_PATH = Path(__file__).parent.parent.parent.resolve().__str__()
RAW_DATA_DIRECTORY = PROJECT_PATH + "/data/raw"

row_data = requests.get(ROW_DATA_ZIP_URL)
zip_row_data = zipfile.ZipFile(io.BytesIO(row_data.content))
zip_row_data.extractall(f"{RAW_DATA_DIRECTORY}")
