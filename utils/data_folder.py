from datetime import datetime
from pathlib import Path


def get_data_folder(folder: str) -> Path:
    folder = Path(folder).expanduser().joinpath(str(datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    folder.mkdir(parents=True, exist_ok=True)
    return folder
