import pandas as pd
import numpy as np
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


def cluster_data(df,cluster_type ,n_clusters=4):
    if cluster_type=='kmeans':
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(df[['Recency', 'Frequency', 'AvgMonetary']])
        df['Cluster'] = kmeans.labels_
        return df
    if cluster_type=='hierarchical':
        labels = shc.fcluster(shc.linkage(df[['Recency', 'Frequency', 'AvgMonetary']], method='ward'), n_clusters, criterion='maxclust')
        df['Cluster'] = labels
        return df
    if cluster_type=='agglomerative':
        agg = AgglomerativeClustering(n_clusters=num_clusters, affinity='euclidean', linkage='ward')
        agg.fit_predict(df[['Recency', 'Frequency', 'AvgMonetary']])
        df['Cluster'] = agg.labels_
        return df
def train_eval_model_cluster(df, use_model='linear'):
    clusters = df['Cluster'].unique()
    cluster_models = {}
    model_eval = {}
    for cluster in clusters:
        # Subset the data for the current segment
        segment_data = df.query(f"Cluster == {cluster}")

        # Define your predictor feature (e.g., Recency) and target variable (e.g., Sales)
        X = segment_data[['Recency', 'Frequency', 'AvgMonetary']]
        y = segment_data['CLTV']

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create a Linear Regression model
        if use_model == 'linear':
            model = LinearRegression()
        if use_model == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        if use_model == 'decision_tree':
            model = DecisionTreeRegressor(random_state=42)

        # Train the model on the training data
        model.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred = model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        model_eval[cluster] = {
            'mse': mse,
            'model_score': model.score(X_test, y_test)
        }
        cluster_models[cluster] = model

    print(
        f"Average metrics for {use_model}:)")
    print(
        f"\n\tMSE={np.mean([model_eval[cluster]['mse'] for cluster in model_eval.keys()])})"
        print(f" \n\tModel Score={np.mean([model_eval[cluster]['model_score'] for cluster in model_eval.keys()])})


class Retail:
    def __init__(self):
        self.retail = pd.read_parquet('retail.parquet')
        self.retail['InvoiceDate'] = pd.to_datetime(self.retail['InvoiceDate'])
        self.retail['PurchasePrice'] = self.retail['Quantity'] * self.retail['Price']
        self.find_country = lambda i: self.retail.query(f'CustomerID=="{i}"')['Country'].unique()[0]
        self.prod_desc = dict(zip(self.retail['StockCode'].unique(), self.retail['Description'].unique()))

        self.popular_products = lambda x: self.retail.query(f'Country=="{x}"').groupby('StockCode')['Quantity'].sum().sort_values(ascending=False)
        self.returns_products = lambda x: self.retail.query(f'Country=="{x}"').groupby('StockCode')['Quantity'].sum().sort_values(ascending=True)
        self.unpopular_products = lambda x: self.retail.query(f'Country=="{x}"').query('Quantity>0').groupby('StockCode')['Quantity'].sum().sort_values(
            ascending=True)

        # Create a new dataframe  with only the relevant columns
        self.retail_relevant = self.retail[['Customer ID', 'InvoiceDate', 'Quantity', 'Price']].copy(deep=False)
        self.retail_relevant = self.retail_relevant.query('Quantity > 0')
        self.retail_relevant['TotalPrice'] = self.retail_relevant['Quantity'] * self.retail_relevant['Price']
        self.retail_relevant = self.retail_relevant.groupby(
            ['Customer ID', pd.Grouper(key='InvoiceDate', freq='D')]).agg({'TotalPrice': 'sum'}).sort_values(
            by='InvoiceDate').reset_index()
        # Create Recency, Frequency and Monetary columns
        # Recency
        recency_df = self.retail_relevant.groupby(['Customer ID'])['InvoiceDate'].max().reset_index()
        recency_df.columns = ['Customer ID', 'InvoiceDate']
        recency_df['Recency'] = recency_df['InvoiceDate'].max() - recency_df['InvoiceDate']
        recency_df['Recency'] = recency_df['Recency'].dt.days
        # Frequency
        frequency_df = self.retail_relevant.groupby('Customer ID')['InvoiceDate'].count().reset_index()
        frequency_df.columns = ['Customer ID', 'Frequency']
        # Monetary
        monetary_df = self.retail_relevant.groupby('Customer ID')['TotalPrice'].sum().reset_index()
        monetary_df.columns = ['Customer ID', 'Monetary']
        # AvgMonetary
        avg_monetary = self.retail_relevant.groupby('Customer ID')['TotalPrice'].mean().reset_index()
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

        # Normalize the RFM df
        self.rfm_df_norm = self.rfm_df.copy(deep=False)
        self.rfm_df_norm['Recency'] = (self.rfm_df_norm['Recency'] - self.rfm_df_norm['Recency'].min()) / (
                    self.rfm_df_norm['Recency'].max() - self.rfm_df_norm['Recency'].min())
        self.rfm_df_norm['Frequency'] = (self.rfm_df_norm['Frequency'] - self.rfm_df_norm['Frequency'].min()) / (
                self.rfm_df_norm['Frequency'].max() - self.rfm_df_norm['Frequency'].min())
        self.rfm_df_norm['Monetary'] = (self.rfm_df_norm['Monetary'] - self.rfm_df_norm['Monetary'].min()) / (
                    self.rfm_df_norm['Monetary'].max() - self.rfm_df_norm['Monetary'].min())
        self.rfm_df_norm['AvgMonetary'] = (self.rfm_df_norm['AvgMonetary'] - self.rfm_df_norm['AvgMonetary'].min()) / (
                self.rfm_df_norm['AvgMonetary'].max() - self.rfm_df_norm['AvgMonetary'].min())
        self.rfm_df_norm['CLTV'] = (self.rfm_df_norm['CLTV'] - self.rfm_df_norm['CLTV'].min()) / (
                    self.rfm_df_norm['CLTV'].max() - self.rfm_df_norm['CLTV'].min())

        self.rfm_kmeans = cluster_data(self.rfm_df_norm, 'kmeans')
        self.rfm_hierarchical = cluster_data(self.rfm_df_norm, 'hierarchical')
        self.rfm_agglomerative = cluster_data(self.rfm_df_norm, 'agglomerative')
    def plot_monthly_purchase_top5(self):
        plt.figure(figsize=(12, 6))
        top_5_countries = list(self.retail.groupby('Country')['CustomerID'].nunique().sort_values(ascending=False).head(6).index)
        top_5_countries_df = self.retail.query(f'Country in {top_5_countries}')
        monthly_data = top_5_countries_df.groupby([pd.Grouper(key='InvoiceDate', freq='M'), 'Country'])['PurchasePrice'].sum().reset_index()
        ax = sns.barplot(data=monthly_data, x='InvoiceDate', y='PurchasePrice', hue='Country')

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.title('Total PurchasePrice by Month for Top 5 Countries')
        plt.show()

    def plot_monthly_purchase_per_country(self, country):
        plt.figure(figsize=(12, 6))
        monthly_data = self.retail.query(f'Country=="{country}"').groupby([pd.Grouper(key='InvoiceDate', freq='M'), 'Country'])[
            'PurchasePrice'].sum().reset_index()
        ax = sns.barplot(data=monthly_data, x='InvoiceDate', y='PurchasePrice')

        # Rotate x-axis labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        plt.title(f'Total PurchasePrice by Month for {country}')
        plt.show()

    def pretty_print_country_stats(self, country='United Kingdom'):
        most_pop_pord = self.popular_products(country).head(1).index[0]
        most_pop_pord_desc = self.prod_desc[most_pop_pord]
        least_pop_pord = self.popular_products(country).tail(1).index[0]
        least_pop_pord_desc = self.prod_desc[least_pop_pord]
        most_ret_pord = self.returns_products(country).head(1).index[0]
        most_ret_pord_desc = self.prod_desc[most_ret_pord]

        print(f'Country: {country}')
        print(f'Number of customers: {self.retail.query(f"Country=='{country}'")["CustomerID"].nunique()}')
        print(f'Number of products sold: {self.retail.query(f"Country=='{country}'")["StockCode"].nunique()}')
        print(f'Total revenue: {self.retail.query(f"Country=='{country}'")["PurchasePrice"].sum()}')
        print(f'Most popular product: {most_pop_pord_desc}')
        print(f'Least popular product: {least_pop_pord_desc}')
        print(f'Most returned product: {most_ret_pord_desc}')
