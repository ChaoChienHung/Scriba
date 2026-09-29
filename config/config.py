import os
from dataclasses import dataclass

@dataclass
class Settings:
    checkpoint_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments", "")
    cache_model_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "model")
    cache_data_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "dataset")