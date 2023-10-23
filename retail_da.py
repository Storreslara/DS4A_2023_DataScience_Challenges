import pandas as pd
import numpy as np
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


class Retail:
    def __init__(self):
        self.retail = pd.read_parquet('retail.parquet')
        self.retail['InvoiceDate'] = pd.to_datetime(self.retail['InvoiceDate'])

        self.find_country = lambda i: self.retail.query(f'CustomerID=="{i}"')['Country'].unique()[0]
        self.prod_desc = dict(zip(self.retail['StockCode'].unique(), self.retail['Description'].unique()))

        self.popular_products = self.retail.groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
        self.returns_products = self.retail.groupby('StockCode')['Quantity'].sum().sort_values(ascending=True)
        self.unpopular_products = self.retail.query('Quantity>0').groupby('StockCode')['Quantity'].sum().sort_values(ascending=True)

        # Create a new dataframe  with only the relevant columns
        retail_df_relevant = self.retail[['Customer ID', 'InvoiceDate', 'Quantity', 'Price']].copy(deep=False)
        retail_df_relevant = retail_df_relevant.query('Quantity > 0')
        retail_df_relevant['TotalPrice'] = retail_df_relevant['Quantity'] * retail_df_relevant['Price']
        retail_df_relevant = retail_df_relevant.groupby(
            ['Customer ID', pd.Grouper(key='InvoiceDate', freq='D')]).agg({'TotalPrice': 'sum'}).sort_values(
            by='InvoiceDate').reset_index()
        # Create Recency, Frequency and Monetary columns
        # Recency
        recency_df = retail_df_relevant.groupby(['Customer ID'])['InvoiceDate'].max().reset_index()
        recency_df.columns = ['Customer ID', 'InvoiceDate']
        recency_df['Recency'] = recency_df['InvoiceDate'].max() - recency_df['InvoiceDate']
        recency_df['Recency'] = recency_df['Recency'].dt.days
        # Frequency
        frequency_df = retail_df_relevant.groupby('Customer ID')['InvoiceDate'].count().reset_index()
        frequency_df.columns = ['Customer ID', 'Frequency']
        # Monetary
        monetary_df = retail_df_relevant.groupby('Customer ID')['TotalPrice'].sum().reset_index()
        monetary_df.columns = ['Customer ID', 'Monetary']
        # AvgMonetary
        avg_monetary = retail_df_relevant.groupby('Customer ID')['TotalPrice'].mean().reset_index()
        avg_monetary.columns = ['Customer ID', 'AvgMonetary']
        # Merge the three dfs into a new df, self.self.rfm_df
        self.self.rfm_df = recency_df.merge(frequency_df, on='Customer ID')
        self.self.rfm_df = self.self.rfm_df.merge(monetary_df, on='Customer ID')
        self.self.rfm_df = self.self.rfm_df.merge(avg_monetary, on='Customer ID')
        self.self.rfm_df.drop('InvoiceDate', axis=1, inplace=True)
        # Calculate Churn Rate and CLTV
        self.repeat_rate = self.self.rfm_df[self.self.rfm_df['Frequency'] > 1].shape[0] / self.self.rfm_df.shape[0]
        print(f'Repeat rate: {self.repeat_rate}')
        self.churn_rate = 1 - repeat_rate
        print(f'Churn rate: {churn_rate}')
        self.self.rfm_df['CLTV'] = self.self.rfm_df['AvgMonetary'] * self.self.rfm_df['Frequency'] / self.churn_rate

        # Normalize the data
        self.rfm_df_norm = self.rfm_df.copy(deep=False)
        self.rfm_df_norm['Recency'] = (self.rfm_df_norm['Recency'] - self.rfm_df_norm['Recency'].min()) / (self.rfm_df_norm['Recency'].max() - self.rfm_df_norm['Recency'].min())
        self.rfm_df_norm['Frequency'] = (self.rfm_df_norm['Frequency'] - self.rfm_df_norm['Frequency'].min()) / (
                    self.rfm_df_norm['Frequency'].max() - self.rfm_df_norm['Frequency'].min())
        self.rfm_df_norm['Monetary'] = (self.rfm_df_norm['Monetary'] - self.rfm_df_norm['Monetary'].min()) / (self.rfm_df_norm['Monetary'].max() - self.rfm_df_norm['Monetary'].min())
        self.rfm_df_norm['AvgMonetary'] = (self.rfm_df_norm['AvgMonetary'] - self.rfm_df_norm['AvgMonetary'].min()) / (
                    self.rfm_df_norm['AvgMonetary'].max() - self.rfm_df_norm['AvgMonetary'].min())
        self.rfm_df_norm['CLTV'] = (self.rfm_df_norm['CLTV'] - self.rfm_df_norm['CLTV'].min()) / (self.rfm_df_norm['CLTV'].max() - self.rfm_df_norm['CLTV'].min())
        
        
