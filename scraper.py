import requests
import pandas as pd


url = "https://www.uvek-gis.admin.ch/BFE/ogd/104/ogd104_stromproduktion_swissgrid.csv"
response = requests.get(url)

if response.status_code == 200:
    csv_file_path = 'stromproduktion_swissgrid.csv'
    with open(csv_file_path, 'wb') as file:
        file.write(response.content)
    
  
    df = pd.read_csv(csv_file_path)
    

    print(df.head())
    
    # saving the DataFrame to another file or manipulate it as needed
    # df.to_csv('processed_stromproduktion_swissgrid.csv', index=False)

    print("Failed to download the file. Status code:", response.status_code)