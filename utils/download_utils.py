import requests
import zipfile
import os
import io
from tqdm.notebook import tqdm
import json

def download_and_extract_zip(keyword, config_file='/utils/config.json'):
    """
    Download a zip file based on a keyword by looking up the URL and path in a config file,
    then extract it to the specified directory.
    
    Parameters:
    - keyword (str): The keyword to look up the download link and extraction path.
    - config_file (str): The path to the configuration file.
    """
    # Load the configuration file
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Get the download link and extraction path for the given keyword
    if keyword in config['datasets']:
        download_link = config['datasets'][keyword]['link']
        extract_to = config['datasets'][keyword]['path']
    else:
        raise ValueError(f"Keyword '{keyword}' not found in the configuration file.")
    
    # Create the directory if it doesn't exist
    os.makedirs(extract_to, exist_ok=True)

    # Send a GET request to the link with stream=True to download in chunks
    response = requests.get(download_link, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the total file size from the headers
        total_size = int(response.headers.get('content-length', 0))

        # Use BytesIO to handle the downloaded content as a file-like object
        file_like_object = io.BytesIO()

        # Download the file in chunks and update the progress bar
        chunk_size = 1024  # 1 KB chunks
        for chunk in tqdm(response.iter_content(chunk_size), total=total_size // chunk_size, unit='KB', colour = "blue"):
            file_like_object.write(chunk)
        
        # Set the file_like_object position to the beginning
        file_like_object.seek(0)
        
        # Open the zip file and extract its contents
        with zipfile.ZipFile(file_like_object, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Folder downloaded and extracted successfully!")
    else:
        print(f"Failed to download file: {response.status_code}")



# Example usage
#download_and_extract_zip('dataset1')

