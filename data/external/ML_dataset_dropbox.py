import dropbox
import pandas as pd

# Dropbox access token
DROPBOX_ACCESS_TOKEN = 'sl.ByVh8odFyt4CaBJ2qq_bOUjewfR3X3Cbhx-2-QooBJaX9c3hTRCiR0Kw7G4MC35Hm82Xkb1-QdsmYqMukcHE7vJ2n00w4Rg-TcsVrjBLuAPGleU2L1TijWHKTwM9BhVflnlEvaxEAas3'
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

