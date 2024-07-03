def get_returns(df):
    # Create a copy of the dataframe to avoid modifying the original one
    returns_df = df.copy()
    
    # Calculate returns for each column except the first one ('ds')
    for col in returns_df.columns[1:]:
        returns_df[col] = returns_df[col].pct_change()
    
    # Drop the first row because pct_change will result in NaN for the first entry
    returns_df.dropna(inplace=True)
    
    return returns_df