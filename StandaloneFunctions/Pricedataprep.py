import pandas as pd
from matplotlib import pyplot as plt


freq_prices = pd.read_csv(r"/Data/RawFCRNPrices.csv")
en_prices = pd.read_csv(r"/Data/RawSpotPrices.csv", delimiter=";")


data = pd.concat([en_prices[en_prices["PriceArea"]=="DK2"][["HourUTC","SpotPriceEUR"]].reset_index(drop=True), freq_prices[["Taajuusohjattu käyttöreservi, tuntimarkkinahinnat"]].reset_index(drop=True)], axis=1)

data = data.rename(columns={"Taajuusohjattu käyttöreservi, tuntimarkkinahinnat": "FCRNPriceEUR"})

data["SpotPriceEUR"] = data["SpotPriceEUR"].str.replace(',', '.').astype(float)
data = data.iloc[1:8761]

print(data)
data.to_csv(r"Data/CombinedPrices.csv", index=False)

plt.plot(data["HourUTC"].astype("datetime64[ns]"), data["SpotPriceEUR"])
plt.plot(data["HourUTC"].astype("datetime64[ns]"), data["FCRNPriceEUR"])
#plt.show()



### Create FCRN Price data file with a column for each day ###

Data = pd.read_csv(r"../InputData/Prices/Old/CombinedPrices.csv")
Data["HourUTC"] = pd.to_datetime(Data["HourUTC"])

Data['hour'] = Data["HourUTC"].dt.hour
Data['day'] = Data["HourUTC"].dt.date

pivot_df = Data.pivot(index='hour', columns='day', values=['FCRNPriceEUR'])
pivot_df = pivot_df.reset_index()

# Displaying the resulting DataFrame

daily_data = pivot_df["FCRNPriceEUR"].copy()
daily_data = daily_data.reset_index()

daily_data.to_csv(r"Data/DailyFCRNPrices.csv", index=False)


### Create Spot Price data file with a column for each day ###
pivot_df = Data.pivot(index='hour', columns='day', values=['SpotPriceEUR'])
pivot_df = pivot_df.reset_index()

# Displaying the resulting DataFrame

daily_data = pivot_df["SpotPriceEUR"].copy()
daily_data = daily_data.reset_index()

daily_data.to_csv(r"Data/DailySpotPrices.csv", index=False)