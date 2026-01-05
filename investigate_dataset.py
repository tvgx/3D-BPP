import pandas as pd
import numpy as np
import pickle
import os

def investigate_pickle_dataset(file_path, limit=None):
    """
    Investigate a pickle dataset file to understand its structure and statistics.
    """
    print("="*70)
    print(f"INVESTIGATING DATASET: {file_path}")
    print("="*70)
    
    if not os.path.exists(file_path):
        print(f"ERROR: File not found: {file_path}")
        return
    
    try:
        # Load the data
        print("\n[1] Loading data...")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        
        print(f"   Data type: {type(data)}")
        
        # Handle different data formats
        if isinstance(data, dict):
            print("\n[2] Dictionary format detected")
            print(f"   Keys: {list(data.keys())}")
            
            if "items" in data:
                items = data["items"]
                print(f"   Number of items: {len(items)}")
                if len(items) > 0:
                    print(f"   First item: {items[0]}")
                    print(f"   Item type: {type(items[0])}")
            
            if "bin_dims" in data:
                print(f"   Bin dimensions: {data['bin_dims']}")
            if "max_weight" in data:
                print(f"   Max weight: {data['max_weight']}")
        
        elif isinstance(data, pd.DataFrame):
            print("\n[2] DataFrame format detected")
            print(f"   Shape: {data.shape} (rows, columns)")
            print(f"   Columns: {list(data.columns)}")
            print(f"   Memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            # Show first few rows
            print("\n[3] First 10 rows:")
            print(data.head(10).to_string())
            
            # Basic statistics
            print("\n[4] Statistical Summary:")
            print(data.describe().to_string())
            
            # Check for missing values
            print("\n[5] Missing Values:")
            missing = data.isnull().sum()
            if missing.sum() > 0:
                print(missing[missing > 0].to_string())
            else:
                print("   No missing values!")
            
            # Analyze dimensions
            if 'width' in data.columns and 'depth' in data.columns and 'height' in data.columns:
                print("\n[6] Dimension Analysis:")
                print(f"   Width range:  {data['width'].min()} - {data['width'].max()} (mean: {data['width'].mean():.2f})")
                print(f"   Depth range:  {data['depth'].min()} - {data['depth'].max()} (mean: {data['depth'].mean():.2f})")
                print(f"   Height range: {data['height'].min()} - {data['height'].max()} (mean: {data['height'].mean():.2f})")
                
                # Volume analysis
                if 'volume' in data.columns:
                    print(f"   Volume range: {data['volume'].min()} - {data['volume'].max()} (mean: {data['volume'].mean():.2f})")
                else:
                    data['calculated_volume'] = data['width'] * data['depth'] * data['height']
                    print(f"   Calculated volume range: {data['calculated_volume'].min()} - {data['calculated_volume'].max()} (mean: {data['calculated_volume'].mean():.2f})")
            
            # Weight analysis
            if 'weight' in data.columns:
                print("\n[7] Weight Analysis:")
                print(f"   Weight range: {data['weight'].min()} - {data['weight'].max()} (mean: {data['weight'].mean():.2f})")
                print(f"   Total weight: {data['weight'].sum():.2f}")
            
            # Sample analysis
            if limit:
                print(f"\n[8] Sampling Analysis (first {limit} items):")
                sample = data.head(limit)
                print(f"   Sample size: {len(sample)}")
                if 'width' in sample.columns and 'depth' in sample.columns and 'height' in sample.columns:
                    total_vol = (sample['width'] * sample['depth'] * sample['height']).sum()
                    print(f"   Total volume of sample: {total_vol:,.0f}")
                    if 'weight' in sample.columns:
                        total_weight = sample['weight'].sum()
                        print(f"   Total weight of sample: {total_weight:,.2f}")
            
            # Bin capacity estimation
            print("\n[9] Bin Capacity Estimation (assuming default bin: 1200x1200x1200):")
            bin_volume = 1200 * 1200 * 1200
            bin_max_weight = 10000
            
            if 'width' in data.columns and 'depth' in data.columns and 'height' in data.columns:
                if limit:
                    sample_data = data.head(limit)
                else:
                    sample_data = data
                
                total_item_volume = (sample_data['width'] * sample_data['depth'] * sample_data['height']).sum()
                min_bins_by_volume = np.ceil(total_item_volume / bin_volume)
                
                if 'weight' in sample_data.columns:
                    total_weight = sample_data['weight'].sum()
                    min_bins_by_weight = np.ceil(total_weight / bin_max_weight)
                    print(f"   For {'first ' + str(limit) + ' items' if limit else 'all items'}:")
                    print(f"   - Minimum bins by volume: {int(min_bins_by_volume)}")
                    print(f"   - Minimum bins by weight: {int(min_bins_by_weight)}")
                    print(f"   - Total item volume: {total_item_volume:,.0f}")
                    print(f"   - Total weight: {total_weight:,.2f}")
                    print(f"   - Theoretical minimum bins: {max(int(min_bins_by_volume), int(min_bins_by_weight))}")
        
        else:
            print(f"\n[2] Unknown format: {type(data)}")
            print(f"   Data preview: {str(data)[:200]}...")
        
        print("\n" + "="*70)
        print("INVESTIGATION COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Investigate dataset structure and statistics")
    parser.add_argument("--file", type=str, 
                       default="data/instances/test-data/Input/Elhedhli_dataset/products.pkl",
                       help="Path to dataset file")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit analysis to first N items (for large datasets)")
    
    args = parser.parse_args()
    
    investigate_pickle_dataset(args.file, limit=args.limit)

