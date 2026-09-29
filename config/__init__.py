import os
from config import Settings

settings = Settings()

if not os.path.exists(settings.checkpoint_path):
    os.makedirs(settings.checkpoint_path)

if not os.path.exists(settings.cache_data_path):
    os.makedirs(settings.cache_data_path)
    
if not os.path.exists(settings.cache_model_path):
    os.makedirs(settings.cache_model_path)