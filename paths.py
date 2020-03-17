from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProjectPaths:
    BASE_PATH: Path = Path(__file__).parents[0]  # project base folder, where everything is stored
    DATA_PATH: Path = BASE_PATH / 'data'  # where the datasets and data relevant information is stored
    LOG_PATH: Path = DATA_PATH / 'log.pkl'  # store relevant info for logging: train loss, test loss, test acc is stored
    WEIGHTS_PATH = DATA_PATH / 'checkpoints'  # store the trained parameters of the model
    IMAGES_PATH = BASE_PATH / 'images'  # store all relevant images

    def __post_init__(self):
        self.DATA_PATH.mkdir(exist_ok=True)


