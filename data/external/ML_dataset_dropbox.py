import dropbox
import pandas as pd

# Dropbox access token
DROPBOX_ACCESS_TOKEN = 'sl.ByVy_h6napjpde37Rn35XPaodYF7p0lI_n13l0ompdhuYgS81UwWFSrpRpwuVNh_ktxBMNbueFtUk-YPxD-K8C7siblHTsY09-qa5WybZfFE_eF7zjh4uQz4koilUYzNukfornsb_y1L'
# Initialize Dropbox client
dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)

# Dropbox file path
dropbox_file_path = '/fake_real_ML_project_dataset.csv'

# Download the file
metadata, response = dbx.files_download(dropbox_file_path)

# Save the downloaded file
with open('data.csv', 'wb') as f:
    f.write(response.content)

# Read the CSV file
df = pd.read_csv('data.csv')

# Show the results
print("Dataframe loaded from Dropbox:")
print(df)

