import pandas as pd
import os

try:
    file_prod = r"data/instances/test-data/Input/Elhedhli_dataset/products.pkl"
    file_new = r"data/instances/test-data/Input/Elhedhli_dataset/newproducts.pkl"

    if os.path.exists(file_prod):
        print(f"--- {file_prod} ---")
        df = pd.read_pickle(file_prod)
        print(f"Type: {type(df)}")
        if isinstance(df, pd.DataFrame):
            print(f"Shape: {df.shape}")
            print("Columns:", df.columns.tolist())
            print(df.head())
        else:
            print(df)
    
    if os.path.exists(file_new):
        print(f"\n--- {file_new} ---")
        df_new = pd.read_pickle(file_new)
        print(f"Type: {type(df_new)}")
        if isinstance(df_new, pd.DataFrame):
            print(f"Shape: {df_new.shape}")
            print("Columns:", df_new.columns.tolist())
            print(df_new.head())
        else:
            print(df_new)

except Exception as e:
    print(e)
