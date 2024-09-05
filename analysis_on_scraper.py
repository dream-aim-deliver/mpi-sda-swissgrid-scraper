
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

unique_energietraeger = df['Energietraeger'].unique()
print("Unique values in Energietraeger column:", unique_energietraeger)

# Step 2: Create a dictionary to map German values to English
translation_dict = {
    'Flusskraft': 'Hydropower (Run-of-river)',
    'Kernkraft': 'Nuclear power',
    'Speicherkraft': 'Hydropower (Storage)',
    'Thermische': 'Thermal power',
    'Photovoltaik':'Photovoltaics',
    'Wind': 'Wind power',

}

# Step 3: Convert the 'Energietraeger' column to English using the translation dictionary
df['Energietraeger_English'] = df['Energietraeger'].map(translation_dict)

# Display the updated DataFrame with the English column
print(df.head())

# Convert the 'Datum' column to datetime format
df['Datum'] = pd.to_datetime(df['Datum'])

# Find the latest date in the 'Datum' column
latest_date = df['Datum'].max()

print("The latest date in the dataset is:", latest_date)

#visualizing the data

import matplotlib.pyplot as plt

start_date = '2023-01-01'
end_date = '2024-12-31'


df_filtered = df[(df['Datum'] >= start_date) & (df['Datum'] <= end_date)]
#aggregrated on day
daily_aggregated = df_filtered.groupby(['Datum', 'Energietraeger_English']).sum().unstack().fillna(0)
# print(daily_aggregated.head())

# Daily Aggregated Data Plot
daily_aggregated.plot(kind='line', stacked=True, figsize=(14, 8))
plt.title('Daily Energy Production by Source')
plt.ylabel('Total Production (GWh)')
plt.xlabel('Date')
plt.legend(title='Energietraeger')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Alternatively, aggregate by week (sum of Produktion_GWh for each Energietraeger by week)
weekly_aggregated = df_filtered.groupby([pd.Grouper(key='Datum', freq='W'), 'Energietraeger_English']).sum().unstack().fillna(0)
# print(weekly_aggregated.head())


# Weekly Aggregated Data Plot
weekly_aggregated.plot(kind='line', stacked=True, figsize=(14, 8))
plt.title('Weekly Energy Production by Source')
plt.ylabel('Total Production (GWh)')
plt.xlabel('Week')
plt.legend(title='Energietraeger')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


monthly_table = df_filtered.pivot_table(index=df['Datum'].dt.to_period('M'), columns='Energietraeger_English', values='Produktion_GWh', aggfunc='sum').fillna(0)

print(monthly_table)

#monthly aggregrate

# Alternatively, aggregate by week (sum of Produktion_GWh for each Energietraeger by week)
monthly_aggregated = df_filtered.groupby([pd.Grouper(key='Datum', freq='M'), 'Energietraeger_English']).sum().unstack().fillna(0)

monthly_aggregated.plot(kind='bar', stacked=True, figsize=(14, 8))
plt.title('Monthly Energy Production by Source (2023-2024)')
plt.ylabel('Total Production (GWh)')
plt.xlabel('Month')
plt.legend(title='Energietraeger')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()