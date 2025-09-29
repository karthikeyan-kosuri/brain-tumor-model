import os

def setup_directories():
    directories=['outputs/checkpoints','outputs/results']
    for directory in directories:
        os.makedirs(directory,exist_ok=True)