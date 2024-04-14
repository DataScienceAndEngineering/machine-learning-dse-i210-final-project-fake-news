import dropbox
import pandas as pd
import io
import os 
from config import dropbox_access_token

# Initialize Dropbox client
dbx = dropbox.Dropbox(dropbox_access_token)

# Dropbox file path
dropbox_file_path = '/fake_real_ML_project_dataset.csv'

# Download the file
metadata, response = dbx.files_download(dropbox_file_path)

# Read the CSV file from the response content using io.BytesIO
df = pd.read_csv(io.BytesIO(response.content))

# Show the results
print("Dataframe loaded from Dropbox:")
print(df.head())
