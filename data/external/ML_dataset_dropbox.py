import dropbox
import pandas as pd
import io 

# Dropbox access token
DROPBOX_ACCESS_TOKEN = 'sl.ByU4ITWZF9fhmTE_RBP5Ungczy859C4t9k16Yd029DLY4LGwAZgVLEd4K7Po3GugXFJx0JL6FIwgJzzgFAVwf9683rEQDbD1ao2_tmJ36vscnYi3HARCcibXBWYBxDOG5z-JfgyARwO6'

# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Dropbox file path
dropbox_file_path = '/fake_real_ML_project_dataset.csv'

# Download the file
metadata, response = dbx.files_download(dropbox_file_path)

# Read the CSV file from the response content using io.BytesIO
df = pd.read_csv(io.BytesIO(response.content))

# Show the results
print("Dataframe loaded from Dropbox:")
print(df)
