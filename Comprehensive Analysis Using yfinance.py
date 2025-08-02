import requests
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from mlxtend.frequent_patterns import apriori, association_rules

# Setup
headers = {'User-Agent': 'hbirring@seattleu.edu'}
cik_map = {
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "COST": "0000909832",
    "FSS": "0000277509",
    "JNJ": "0000200406"
}

# XBRL tags to retrieve
tags = {
    'Revenue': 'RevenueFromContractWithCustomerExcludingAssessedTax',
    'NetIncome': 'NetIncomeLoss',
    'TotalAssets': 'Assets',
    'TotalLiabilities': 'Liabilities',
    'CurrentAssets': 'AssetsCurrent',
    'CurrentLiabilities': 'LiabilitiesCurrent',
    'Inventory': 'InventoryNet'
}

# Function to fetch latest 10-K values per year
def get_tag_data(cik, tag):
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json"
    r = requests.get(url, headers=headers)
    if r.status_code != 200:
        return {}
    data = r.json()
    values = data.get('units', {}).get('USD', [])
    filtered = [f for f in values if f.get('form') == '10-K' and f.get('fy') and f['fy'] >= 2019]
    return {f['fy']: f['val'] for f in filtered}

stock_dfs = {}

# Pull data for each stock
ticker_features = []
for ticker, cik in cik_map.items():
    all_data = {}
    for display_name, tag in tags.items():
        tag_data = get_tag_data(cik, tag)
        for year, val in tag_data.items():
            if year not in all_data:
                all_data[year] = {}
            all_data[year][display_name] = val

    df = pd.DataFrame.from_dict(all_data, orient='index').sort_index(ascending=False)
    df.index.name = "Fiscal Year"
    df.reset_index(inplace=True)

    # Fill missing required columns with NaN if not present
    for col in ['Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities', 'CurrentAssets', 'CurrentLiabilities', 'Inventory']:
        if col not in df.columns:
            df[col] = pd.NA

    # Drop rows with missing values required for ratio calculation
    df = df.dropna(subset=['Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities', 'CurrentAssets', 'CurrentLiabilities', 'Inventory'])

    if df.empty:
        print(f" Not enough valid data to analyze {ticker}.")
        continue

    # Compute financial ratios
    df['net_profit_margin'] = df['NetIncome'] / df['Revenue']
    df['current_ratio'] = df['CurrentAssets'] / df['CurrentLiabilities']
    df['quick_ratio'] = (df['CurrentAssets'] - df['Inventory']) / df['CurrentLiabilities']
    df['debt_to_equity'] = df['TotalLiabilities'] / (df['TotalAssets'] - df['TotalLiabilities'])
    df['roe'] = df['NetIncome'] / (df['TotalAssets'] - df['TotalLiabilities'])

    # Normalize features for Apriori
    scaler = MinMaxScaler()
    features_to_use = ['net_profit_margin', 'current_ratio', 'quick_ratio', 'debt_to_equity', 'roe']
    scaled = scaler.fit_transform(df[features_to_use])
    df_scaled = pd.DataFrame(scaled, columns=features_to_use)
    df_scaled['Fiscal Year'] = df['Fiscal Year']

    # Create binary features for Apriori
    df_bin = df_scaled.copy()
    df_bin['net_profit_margin_High'] = (df_bin['net_profit_margin'] > 0.5).astype(int)
    df_bin['current_ratio_High'] = (df_bin['current_ratio'] > 0.5).astype(int)
    df_bin['quick_ratio_High'] = (df_bin['quick_ratio'] > 0.5).astype(int)
    df_bin['debt_to_equity_Low'] = (df_bin['debt_to_equity'] < 0.5).astype(int)
    df_bin['roe_High'] = (df_bin['roe'] > 0.5).astype(int)
    df_bin = df_bin[['net_profit_margin_High', 'current_ratio_High', 'quick_ratio_High', 'debt_to_equity_Low', 'roe_High']]

    # Apriori
    frequent_itemsets = apriori(df_bin, min_support=0.3, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

    # Store
    stock_dfs[ticker] = {
        'raw': df,
        'scaled': df_scaled,
        'bin': df_bin,
        'rules': rules
    }

    print(f"\n Full Financial Data for {ticker}:")
    print(df[['Fiscal Year', 'Revenue', 'NetIncome', 'TotalAssets', 'TotalLiabilities']])
    print(f"\n Financial Ratios for {ticker}:")
    print(df[['Fiscal Year'] + features_to_use])
    print(f"\n Top Association Rules for {ticker}:")
    if not rules.empty:
        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        sorted_rules = rules[display_cols].sort_values(by='confidence', ascending=False).head(10)
        print(sorted_rules)

        # Plot bar chart of top association rules
        plt.figure(figsize=(10, 6))
        labels = [f"{', '.join(map(str, a))} → {', '.join(map(str, c))}" for a, c in zip(sorted_rules['antecedents'], sorted_rules['consequents'])]
        plt.bar(labels, sorted_rules['confidence'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Confidence")
        plt.title("Top Association Rules by Confidence")
        plt.tight_layout()
        plt.show()
    else:
        print(" No strong association rules found.")

    # Trend Graphs
    df.set_index('Fiscal Year')[features_to_use].plot(title=f"{ticker} Financial Ratios Over Time")
    plt.tight_layout()
    plt.show()