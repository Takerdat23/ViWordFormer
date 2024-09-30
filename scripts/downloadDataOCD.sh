#!/bin/bash

# Install gdown if it's not already installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Installing gdown..."
    pip install gdown
fi

# Download the files using gdown
echo "Downloading dev.json..."
gdown "https://drive.google.com/file/d/1AxJne-d_ZWTliHar7htCR77dJ5BIQdlv/view?usp=sharing" -O ./data/UIT-ViOCD/dev.json

echo "Downloading test.json..."
gdown "https://drive.google.com/file/d/1trjIdfVdU0wKAOt1NRq9fB4KlZK-1qak/view?usp=drive_link" -O ./data/UIT-ViOCD/test.json

echo "Downloading train.json..."
gdown "https://drive.google.com/file/d/1NoTCgaG-FUoW4Gxdqt2WcVWFTzqkMNjC/view?usp=drive_link" -O ./data/UIT-ViOCD/train.json

echo "All files downloaded successfully."
