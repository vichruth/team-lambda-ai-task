import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 

# to load data
df=pd.read_csv("/home/vichruth/ml_projects/team-lambda-ai-task/Dataset.csv")

# to clean data
# fill null values
df['Discount Applied']=df['Discount Applied'].fillna(False)
df['Item']= df['Item'].fillna("Unknown Item")

# fill total spent
mask_total = df['Total Spent'].isnull() & df['Price Per Unit'].notnull() & df['Quantity'].notnull()
df.loc[mask_total, 'Total Spent'] = df.loc[mask_total, 'Price Per Unit'] * df.loc[mask_total, 'Quantity']

# fill price per unit
mask_price = df['Price Per Unit'].isnull() & df['Total Spent'].notnull() & df['Quantity'].notnull()
df.loc[mask_price, 'Price Per Unit'] = df.loc[mask_price, 'Total Spent'] / df.loc[mask_price, 'Quantity']

# fill quantity
mask_qty = df['Quantity'].isnull() & df['Total Spent'].notnull() & df['Price Per Unit'].notnull()
df.loc[mask_qty, 'Quantity'] = df.loc[mask_qty, 'Total Spent'] / df.loc[mask_qty, 'Price Per Unit']
df_clean = df.dropna(subset=['Total Spent', 'Price Per Unit', 'Quantity']).copy()

# fill date
df_clean['Transaction Date'] = pd.to_datetime(df_clean['Transaction Date'])
df_clean['Month_Year'] = df_clean['Transaction Date'].dt.to_period('M')

# save cleaned data
df_clean.to_csv('cleaned_retail_data.csv', index=False)
print("Data cleaning complete. Saved to 'cleaned_retail_data.csv'.")

# Data Visualization

# monthly sales trends
plt.figure(figsize=(12, 6))
monthly_sales = df_clean.groupby('Month_Year')['Total Spent'].sum()
monthly_sales.plot(marker='o', color='royalblue')
plt.title('Sales Trends Over Time (Monthly Revenue)')
plt.ylabel('Total Revenue')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# top product categories
plt.figure(figsize=(10, 6))
cat_sales = df_clean.groupby('Category')['Total Spent'].sum().sort_values(ascending=False)
sns.barplot(x=cat_sales.values, y=cat_sales.index, palette='viridis')
plt.title('Top Product Categories by Total Sales')
plt.xlabel('Total Revenue')
plt.show()

# payment method distribution
plt.figure(figsize=(8, 8))
df_clean['Payment Method'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=sns.color_palette('pastel'))
plt.title('Payment Method Distribution')
plt.ylabel('')
plt.show()

#Location-wise Sales Performance
plt.figure(figsize=(8, 6))
sns.boxplot(x='Location', y='Total Spent', data=df_clean, palette='Set2')
plt.title('Sales Distribution by Location')
plt.show()

#Correlation Analysis
plt.figure(figsize=(8, 6))
corr = df_clean[['Price Per Unit', 'Quantity', 'Total Spent']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation: Price, Quantity, vs Total Spent')
plt.show()

# Summary Statistics
item_stats = df_clean[df_clean['Item'] != 'Unknown'].groupby('Item')['Total Spent'].mean()
print(f"Item with Highest Avg Sales: {item_stats.idxmax()} ({item_stats.max():.2f})")
print(f"Item with Lowest Avg Sales: {item_stats.idxmin()} ({item_stats.min():.2f})")
