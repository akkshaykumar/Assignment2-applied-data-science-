#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import wbdata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

sns.set()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the data into dataframes
df_co2 = pd.read_csv('API_19_DS2_en_csv_v2_5312862.csv', skiprows=4)
df_countries = pd.read_csv('Metadata_Country_API_19_DS2_en_csv_v2_5312862.csv')
df_indicators = pd.read_csv('Metadata_Indicator_API_19_DS2_en_csv_v2_5312862.csv')


# In[3]:


def clean_data(filename):
    df = pd.read_csv(filename, skiprows=4)
    # Drop unnecessary columns
    df = df.drop(['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'], axis=1)
    # Transpose dataframe
    df_t = df.set_index('Country Name').T
    # Reset index
    df_t = df_t.reset_index()
    # Rename index column
    df_t = df_t.rename(columns={'index': 'Year'})
    # Convert year column to datetime type
    df_t['Year'] = pd.to_datetime(df_t['Year'], format='%Y')
    # Set year column as index
    df_t = df_t.set_index('Year')
    # Drop rows with all NaN values
    df_t = df_t.dropna(how='all')
    # Drop columns with any NaN values
    df_t = df_t.dropna(axis=1, how='any')
    # Fill remaining NaN values with 0
    df_t = df_t.fillna(0)
    # Create dataframe with countries as columns
    df_countries = df_t.transpose()
    # Reset index
    df_countries = df_countries.reset_index()
    # Rename index column
    df_countries = df_countries.rename(columns={'index': 'Country'})
    # Create dataframe with years as columns
    df_years = df_t.reset_index()
    return df_years, df_countries


# In[4]:


df_years, df_countries = clean_data('API_19_DS2_en_csv_v2_5312862.csv')


# In[5]:


df_countries.head()


# In[6]:


df_years.head()


# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def read_and_process(filename):
    # Read the CSV file
    df = pd.read_csv(filename, skiprows=4)
    
    # Drop unnecessary columns
    df = df.drop(columns=['Country Code', 'Indicator Name', 'Indicator Code', 'Unnamed: 66'])

    # Pivot the dataframe
    df_years = df.melt(id_vars=['Country Name'], var_name='Year', value_name='Value')
    df_countries = df.set_index('Country Name').transpose()

    return df_years, df_countries

# Read and process the data
co2_data_years, co2_data_countries = read_and_process('API_19_DS2_en_csv_v2_5312862.csv')

# Visualization 1: CO2 Emissions by Country
top10_emissions = co2_data_countries.loc['2019'].sort_values(ascending=False).head(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=top10_emissions.index, y=top10_emissions.values)
plt.title('Top 7 Countries by CO2 Emissions in 2019')
plt.ylabel('CO2 Emissions (kt)')
plt.xticks(rotation=45)
plt.show()


# In[43]:


# Visualization 3: CO2 Emissions Over Time
selected_countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom']
co2_data_selected = co2_data_years[co2_data_years['Country Name'].isin(selected_countries)]
co2_data_selected['Year'] = pd.to_numeric(co2_data_selected['Year'])
plt.figure(figsize=(12, 6))
sns.lineplot(data=co2_data_selected, x='Year', y='Value', hue='Country Name')
plt.title('CO2 Emissions Over Time')
plt.ylabel('CO2 Emissions (kt)')
plt.show()



# # https://data.worldbank.org/indicator/EG.FEC.RNEW.ZS?downloadformat=csv
# # https://data.worldbank.org/indicator/NY.GDP.MKTP.CD?downloadformat=csv

# In[45]:


# Read and process the data for renewable energy consumption
renewable_energy_years, renewable_energy_countries = read_and_process('API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5343795.csv')

# Read and process the data for GDP
gdp_years, gdp_countries = read_and_process('API_NY.GDP.MKTP.CD_DS2_en_csv_v2_5345463.csv')

# Visualization 4: Renewable Energy Consumption Over Time
selected_countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom']
renewable_energy_selected = renewable_energy_years[renewable_energy_years['Country Name'].isin(selected_countries)]
renewable_energy_selected['Year'] = pd.to_numeric(renewable_energy_selected['Year'])
plt.figure(figsize=(12, 6))
sns.lineplot(data=renewable_energy_selected, x='Year', y='Value', hue='Country Name')
plt.title('Renewable Energy Consumption Over Time')
plt.ylabel('Renewable Energy Consumption (% of total final energy consumption)')
plt.show()



# In[47]:


# Visualization 5: CO2 Emissions vs GDP (2019)
selected_countries = ['United States', 'China', 'India', 'Germany', 'United Kingdom']
co2_data_2019 = co2_data_countries.loc['2019', selected_countries].reset_index()
gdp_data_2019 = gdp_countries.loc['2019', selected_countries].reset_index()

co2_gdp_data = pd.DataFrame({
    'Country Name': co2_data_2019['Country Name'],
    'CO2 Emissions (kt)': co2_data_2019['2019'],
    'GDP (current US$)': gdp_data_2019['2019']
})

plt.figure(figsize=(12, 6))
sns.scatterplot(data=co2_gdp_data, x='GDP (current US$)', y='CO2 Emissions (kt)', hue='Country Name')
plt.title('CO2 Emissions vs GDP in 2019')
plt.xlabel('GDP (current US$)')
plt.ylabel('CO2 Emissions (kt)')
plt.show()


# In[4]:


def fetch_data(indicators, countries, start_year, end_year):
    data_date = datetime.datetime(start_year, 1, 1), datetime.datetime(end_year, 12, 31)
    data = wbdata.get_dataframe(indicators, country=countries, data_date=data_date)
    data.reset_index(inplace=True)
    return data


# In[5]:


indicators = {
    'EN.ATM.CO2E.KT': 'CO2_emissions',
    'EG.USE.PCAP.KG.OE': 'Energy_use',
    'SP.POP.TOTL': 'Total_population',
    'AG.LND.AGRI.K2': 'Agricultural_land',
    'SP.URB.TOTL': 'Urban_population',
    'EG.ELC.ACCS.ZS': 'Access_to_electricity'
}

countries = ['USA', 'CHN', 'IND', 'DEU', 'BRA']
start_year, end_year = 2000, 2019

data = fetch_data(indicators, countries, start_year, end_year)


# In[6]:


data.head()


# In[7]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='CO2_emissions', hue='country')
plt.title('CO2 Emissions (2000-2019)')
plt.xlabel('Year')
plt.ylabel('CO2 Emissions (kt)')
plt.show()


# In[8]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='Energy_use', hue='country')
plt.title('Energy Use per Capita (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Energy Use (kg of oil equivalent per capita)')
plt.show()


# In[9]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='Total_population', hue='country')
plt.title('Total Population (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Total Population')
plt.show()


# In[10]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='Agricultural_land', hue='country')
plt.title('Agricultural Land Area (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Agricultural Land Area (sq. km)')
plt.show()


# In[11]:


plt.figure(figsize=(12, 6))
sns.lineplot(data=data, x='date', y='Urban_population', hue='country')
plt.title('Urban Population (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.show()


# In[12]:


plt.figure(figsize=(12, 6))
sns.violinplot(data=data, x='country', y='Agricultural_land')
plt.title('Agricultural Land Area by Country (2000-2019)')
plt.xlabel('Country')
plt.ylabel('Agricultural Land Area (sq. km)')
plt.show()


# In[13]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=data, x='date', y='Urban_population', hue='country')
plt.title('Urban Population by Country (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Urban Population')
plt.show()


# In[15]:


access_electricity_pivot = data.pivot_table(values='Access_to_electricity', index='country', columns='date')

plt.figure(figsize=(12, 6))
sns.heatmap(access_electricity_pivot, annot=True, cmap='coolwarm', fmt='.1f')
plt.title('Access to Electricity by Country (2000-2019)')
plt.xlabel('Year')
plt.ylabel('Country')
plt.show()


# In[ ]:





# In[ ]:




