#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


# In[16]:


rental_all_homes_df = pd.read_csv("/kaggle/input/dataset/rental_long.csv")
mortgage_df = pd.read_csv("/kaggle/input/dataset/mortgage_long.csv")


# In[17]:


rental_all_homes_df


# In[18]:


def forecast_all_metros(df, target_col, periods=24):
    all_forecasts = []
    for region_id, group in df.groupby("RegionID"):
        try:
            metro = group[["Date", target_col]].rename(columns={"Date": "ds", target_col: "y"})
            model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            model.fit(metro)
            future = model.make_future_dataframe(periods=periods, freq="M")
            forecast = model.predict(future)
            forecast["RegionID"] = region_id
            forecast["RegionName"] = group["RegionName"].iloc[0]
            forecast["StateName"] = group["StateName"].iloc[0]
            forecast["SizeRank"] = group["SizeRank"].iloc[0]
            forecast["Target"] = target_col
            all_forecasts.append(forecast[["ds", "RegionID", "RegionName", "StateName", "SizeRank", "yhat", "yhat_lower", "yhat_upper", "Target"]])
        except Exception as e:
            print(f"Skipping RegionID {region_id} due to error: {e}")
    return [model, pd.concat(all_forecasts, ignore_index=True)]


# In[19]:


rental_model, forecast_rentals = forecast_all_metros(rental_all_homes_df, "Rent_Index")
forecast_rentals


# In[20]:


mortgage_model, forecast_mortgage = forecast_all_metros(mortgage_df, "Monthly_Payment_20Down")
forecast_mortgage


# In[21]:


rent_forecast = forecast_rentals.rename(columns={"yhat": "Rent_Forecast"})
mortgage_forecast = forecast_mortgage.rename(columns={"yhat": "Mortgage_Forecast"})


# In[22]:


rent_forecast


# In[23]:


forecast_merged = (
    rent_forecast
    .merge(mortgage_forecast, on=["ds", "RegionID", "RegionName", "StateName", "SizeRank"], how="inner")
)


# In[24]:


forecast_merged


# In[25]:


forecast_merged = forecast_merged.drop(["Target_x", "Target_y"], axis=1)


# In[26]:


forecast_merged["Rent_Mortgage_Ratio_Forecast"] = (
    forecast_merged["Rent_Forecast"] / forecast_merged["Mortgage_Forecast"]
)

forecast_merged["Affordability_Gap_Forecast"] = (
    forecast_merged["Mortgage_Forecast"] - forecast_merged["Rent_Forecast"]
)


# In[ ]:


# forecast_merged = forecast_merged[['ds', 'RegionID', 'RegionName', 'StateName',
#        'SizeRank', 'Rent_Forecast', 'Mortgage_Forecast', 'Rent_Mortgage_Ratio_Forecast',
#        'Affordability_Gap_Forecast']]


# In[27]:


forecast_merged.columns


# In[ ]:


national_forecast_trend = forecast_merged.groupby('ds')[["Rent_Forecast", 
"Mortgage_Forecast", "Rent_Mortgage_Ratio_Forecast", "Affordability_Gap_Forecast"]].mean().reset_index()


# In[29]:


national_forecast_trend


# In[30]:


state_2027 = (
    forecast_merged[forecast_merged["ds"] == "2027-06-30"]
    .groupby("StateName")[["Rent_Forecast", "Mortgage_Forecast",
                           "Rent_Mortgage_Ratio_Forecast", "Affordability_Gap_Forecast"]]
    .mean()
    .reset_index()
    .sort_values("Affordability_Gap_Forecast", ascending=False)
)
state_2027


# In[31]:


metro_2027 = forecast_merged[forecast_merged["ds"] == "2027-06-30"]

top_expensive_to_own = metro_2027.sort_values("Affordability_Gap_Forecast", ascending=False).head(10)
top_expensive_to_rent = metro_2027.sort_values("Affordability_Gap_Forecast", ascending=True).head(10)


# In[32]:


top_expensive_to_own


# In[33]:


top_expensive_to_rent


# In[36]:


plt.figure(figsize=(12,6))
plt.plot(national_forecast_trend["ds"], national_forecast_trend["Affordability_Gap_Forecast"], label="Affordability Gap")
plt.axhline(0, color="red", linestyle="--")
plt.title("National Forecast: Affordability Gap (2024–2027)")
plt.xlabel("Date")
plt.ylabel("Gap (Mortgage - Rent)")
plt.legend()
plt.grid(True)
plt.show()


# In[39]:


plt.figure(figsize=(12,12))
sns.barplot(x="Affordability_Gap_Forecast", y="StateName", data=state_2027)
plt.title("State-Level Affordability Gap (2027)")
plt.xlabel("Affordability Gap (Mortgage - Rent)")
plt.ylabel("State")
plt.show()


# In[40]:


plt.figure(figsize=(12,6))
sns.barplot(x="Affordability_Gap_Forecast", y="RegionName", data=top_expensive_to_own)
plt.title("Top 10 Metros: Owning More Expensive than Renting (2027)")
plt.xlabel("Affordability Gap (Mortgage - Rent)")
plt.ylabel("Metro Area")
plt.show()


# In[41]:


plt.figure(figsize=(12,6))
sns.barplot(x="Affordability_Gap_Forecast", y="RegionName", data=top_expensive_to_rent)
plt.title("Top 10 Metros: Renting More Expensive than Owning (2030)")
plt.xlabel("Affordability Gap (Mortgage - Rent)")
plt.ylabel("Metro Area")
plt.show()


# In[ ]:




