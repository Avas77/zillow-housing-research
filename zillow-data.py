#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet


# In[2]:


all_homes_df = pd.read_csv("dataset/zhvi-all-homes.csv")
rental_all_homes_df = pd.read_csv("dataset/zori-rental-all-homes.csv")
total_monthly_payments_df = pd.read_csv("dataset/zhvi-total-monthly-payment.csv")
affordability_df = pd.read_csv("dataset/affordability-home-owner.csv")


# In[3]:


all_homes_df = all_homes_df[all_homes_df["RegionType"] == "msa"]
rental_all_homes_df = rental_all_homes_df[rental_all_homes_df["RegionType"] == "msa"]
total_monthly_payments_df = total_monthly_payments_df[total_monthly_payments_df["RegionType"] == "msa"]
affordability_df = affordability_df[affordability_df["RegionType"] == "msa"]


# In[4]:


all_homes_df = all_homes_df.drop(columns=["RegionType"])
rental_all_homes_df = rental_all_homes_df.drop(columns=["RegionType"])
total_monthly_payments_df = total_monthly_payments_df.drop(columns=["RegionType"])
affordability_df = affordability_df.drop(columns=["RegionType"])


# In[5]:


def melt_df(df, value_name):
    id_vars = ['RegionID', 'RegionName', 'StateName', 'SizeRank'] 
    df_melted = df.melt(id_vars=id_vars, var_name="Date", value_name=value_name)
    df_melted["Date"] = pd.to_datetime(df_melted["Date"]) 
    return df_melted


# In[6]:


all_homes_long = melt_df(all_homes_df, "Home_Value_Index")
rental_all_homes_long = melt_df(rental_all_homes_df, "Rent_Index")
total_monthly_payments_long = melt_df(total_monthly_payments_df, "Monthly_Payment_20Down")
affordability_long = melt_df(affordability_df, "Median_Income")


# In[7]:


rental_all_homes_df.head()


# In[8]:


merged_rent_monthly_payment = pd.merge(rental_all_homes_long, total_monthly_payments_long, on=["RegionID", "RegionName", "StateName", "SizeRank", "Date"])


# In[9]:


merged_rent_monthly_payment.head(10)


# In[10]:


# It shows how much renting costs compared to owning a home in a given area. For example:
# A ratio of 1 means renting and owning cost about the same per month.
# A ratio greater than 1 means renting is more expensive than owning.
# A ratio less than 1 means owning is more expensive than renting.

merged_rent_monthly_payment["Rent_Mortgage_ratio"] = merged_rent_monthly_payment["Rent_Index"] / merged_rent_monthly_payment["Monthly_Payment_20Down"]


# In[11]:


# It shows the difference in monthly cost between owning a home and renting in a given area:
# A positive Affordability_gap means owning a home (mortgage payment) is more expensive than renting.
# A negative Affordability_gap means renting is more expensive than owning.
# A gap close to zero means renting and owning cost about the same.

merged_rent_monthly_payment["Affordability_gap"] = merged_rent_monthly_payment["Monthly_Payment_20Down"] - merged_rent_monthly_payment["Rent_Index"]


# In[12]:


merged_rent_monthly_payment.head(10)


# In[13]:


national_trend = merged_rent_monthly_payment.groupby("Date")[["Rent_Index", "Monthly_Payment_20Down"]].mean()
national_trend


# In[14]:


plt.figure(figsize=(12,6))
plt.plot(national_trend.index, national_trend["Rent_Index"], label="Rent Index", linewidth=2)
plt.plot(national_trend.index, national_trend["Monthly_Payment_20Down"], label="Mortgage (20% Down)", linewidth=2)
plt.title("National Rent vs Mortgage Trend Over Time", fontsize=14)
plt.xlabel("Date")
plt.ylabel("USD")
plt.legend()
plt.grid(True)
plt.show()


# What the Plot Shows
# 
# Early Years (2016–2019):
# 
# - Both the Rent Index and Mortgage payment started at a similar level, around $1,000–$1,200 per month.
# - The Mortgage line fluctuated more, with a noticeable peak around 2018–2019, while the Rent Index grew steadily but at a slower pace.
# 
# 
# Mid-Period (2020–2022):
# 
# - Around 2020, the Mortgage payment dropped slightly, possibly due to lower interest rates or economic changes (e.g., pandemic-related mortgage relief).
# - After 2020, the Mortgage payment began a sharp upward trend, rising significantly by 2022, reaching around $1,800–$2,000.
# - The Rent Index continued to increase steadily, crossing the $1,400 mark by 2022, but it grew more gradually compared to the Mortgage payment.
# 
# 
# Later Years (2023–2026):
# 
# - The Mortgage payment continued to climb steeply, peaking above $2,200 by 2024 and stabilizing or slightly declining toward $2,400 by 2026.
# - The Rent Index kept rising steadily, reaching around $1,600–$1,700 by 2026, but it remained well below the Mortgage payment throughout this period.
# - The gap between the two lines widened significantly after 2022, showing that the cost of owning a home grew much faster than the cost of renting.

# Key Insight:
# 
# The plot indicates that, nationally, the cost of owning a home (mortgage payment with 20% down) has become much higher than the cost of renting over time. By 2026, the Mortgage payment is about $700–$800 more per month than the Rent Index.

# In[15]:


latest_date = merged_rent_monthly_payment["Date"].max()
latest_data = merged_rent_monthly_payment[merged_rent_monthly_payment["Date"] == latest_date]
latest_data


# In[16]:


top_10_cheap_rents = latest_data.sort_values("Rent_Mortgage_ratio", ascending=False).head(10)
top_10_cheap_rents


# In[17]:


top_10_buys_cheap = latest_data.sort_values("Rent_Mortgage_ratio", ascending=True).head(10)
top_10_buys_cheap


# In[18]:


print("Top metros where renting is more expensive than owning:")
print(top_10_cheap_rents[["RegionName", "StateName", "Rent_Mortgage_ratio"]])


# In[19]:


print("\nTop metros where owning is more expensive than renting:")
print(top_10_buys_cheap[["RegionName", "StateName", "Rent_Mortgage_ratio"]])


# In[20]:


state_ratio = latest_data.groupby("StateName")["Rent_Mortgage_ratio"].mean().reset_index()

plt.figure(figsize=(12,12))
sns.barplot(data=state_ratio.sort_values(["Rent_Mortgage_ratio"], ascending=False), x="Rent_Mortgage_ratio", y="StateName", palette="coolwarm")
plt.title("Rent-to-Mortgage Ratio by State (Latest Month)")
plt.xlabel("Ratio (Rent / Mortgage)")
plt.ylabel("State")
plt.show()


# Highest Ratios (Top States):
# 
# States like West Virginia (WV), Mississippi (MS), Oklahoma (OK), Louisiana (LA), and Alabama (AL) have the highest ratios, approaching or slightly exceeding 1.0. This suggests that in these states, the cost of renting is nearly as high as (or very close to) the cost of owning a home.
# 
# Lowest Ratios (Bottom States):
# 
# States like South Dakota (SD), Hawaii (HI), Montana (MT), Wyoming (WY), and California (CA) have the lowest ratios, around 0.2 to 0.4. This indicates that renting is significantly cheaper than owning in these states.

# In[21]:


gap_trend = merged_rent_monthly_payment.groupby("Date")["Affordability_gap"].mean()

plt.figure(figsize=(12, 6))
plt.plot(gap_trend.index, gap_trend, label="Avg Affordability Gap", color="purple", linewidth=2)
plt.axhline(0, color="red", linestyle="--", label="Break-even Line")
plt.title("National Affordability Gap: Rent vs Mortgage")
plt.xlabel("Date")
plt.ylabel("USD (Mortgage - Rent)")
plt.legend()
plt.grid(True)
plt.show()


# Lines:
# 
# - Purple Line (Avg Affordability Gap): Shows the average national affordability gap, calculated as the difference between the monthly mortgage payment (e.g., with 20% down) and the monthly rent (e.g., Zillow’s Observed Rent Index, ZORI).
# 
# - Red Dashed Line (Break-even Line): Represents a gap of $0, where the cost of renting equals the cost of owning. Values above this line indicate owning is more expensive; values below would indicate renting is more expensive.

# In[22]:


metro_id = 394913
metro_home = all_homes_long[all_homes_long["RegionID"] == metro_id]
metro_home = metro_home[["Date", "Home_Value_Index"]].copy()
metro_home = metro_home.rename(columns={"Date": "ds", "Home_Value_Index": "y"})


# In[23]:


metro_rent = rental_all_homes_long[rental_all_homes_long["RegionID"] == metro_id]
metro_rent = metro_rent[["Date", "Rent_Index"]].copy()
metro_rent = metro_rent.rename(columns={"Date": "ds", "Rent_Index": "y"})


# In[24]:


metro_mortgage = total_monthly_payments_long[total_monthly_payments_long["RegionID"] == metro_id]
metro_mortgage = metro_mortgage[["Date", "Monthly_Payment_20Down"]].copy()
metro_mortgage = metro_mortgage.rename(columns={"Date": "ds", "Monthly_Payment_20Down": "y"})


# In[25]:


mortgage_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
mortgage_model.fit(metro_mortgage)


# In[26]:


home_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
home_model.fit(metro_home)


# In[27]:


rent_model = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
rent_model.fit(metro_rent)


# In[28]:


future_rent = rent_model.make_future_dataframe(periods=24, freq='M')
forecast_rent = rent_model.predict(future_rent)


# In[29]:


future_mortgage = mortgage_model.make_future_dataframe(periods=24, freq='M')
forecast_mortgage = mortgage_model.predict(future_mortgage)


# In[30]:


future_home = home_model.make_future_dataframe(periods=24, freq="M")
forecast_home = home_model.predict(future_home)


# In[31]:


forecast_home


# In[32]:


home_model.plot(forecast_home)
plt.title("Forecast: Home Value Index")
plt.show()


# In[33]:


rent_model.plot(forecast_rent)
plt.title("Forecast: Rent Index")
plt.show()


# In[34]:


forecast_combined = pd.DataFrame({
    "Date": forecast_home['ds'],
    "Home_Value_Forecast": forecast_home['yhat'],
    "Rent_Forecast": forecast_rent['yhat'],
    "Mortgage_Value_Forecase": forecast_mortgage['yhat']
})


# In[35]:


forecast_combined = forecast_combined.dropna()
forecast_combined


# In[ ]:




