import pickle
import pandas as pd
import sys

def inspect(path):
    print(f"--- Inspecting {path} ---")
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            print(f"Type: {type(data)}")
            if isinstance(data, list):
                print(f"Length: {len(data)}")
                if len(data) > 0:
                    print(f"Sample item 0: {data[0]}")
            elif isinstance(data, dict):
                print(f"Keys: {list(data.keys())}")
                for k, v in list(data.items())[:2]:
                    print(f"Key {k}: Type {type(v)}, Value (repr): {repr(v)[:200]}...")
            elif isinstance(data, pd.DataFrame):
                print("DataFrame Info:")
                print(data.info())
                print(data.head())
            else:
                print(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect("data/instances/test-data/Input/Elhedhli_dataset/products.pkl")
    inspect("data/instances/test-data/Input/Elhedhli_dataset/newproducts.pkl")
