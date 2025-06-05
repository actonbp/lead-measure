#!/usr/bin/env python3
"""
Quick test script to verify data formats are handled correctly
"""

import pandas as pd
import sys

def test_data_loading():
    """Test that all datasets load correctly with proper column handling."""
    
    print("Testing data format handling...")
    print("="*50)
    
    # Test IPIP data
    print("\n1. Testing IPIP data:")
    try:
        ipip_df = pd.read_csv("data/IPIP.csv", encoding="latin-1").dropna()
        print(f"✓ Loaded {len(ipip_df)} IPIP items")
        print(f"  Columns: {list(ipip_df.columns)}")
        print(f"  Text column: 'text' {'✓' if 'text' in ipip_df.columns else '✗'}")
        print(f"  Label column: 'label' {'✓' if 'label' in ipip_df.columns else '✗'}")
    except Exception as e:
        print(f"✗ Error loading IPIP data: {e}")
        
    # Test holdout items
    print("\n2. Testing holdout items:")
    try:
        holdout_df = pd.read_csv("data/processed/ipip_holdout_items.csv")
        print(f"✓ Loaded {len(holdout_df)} holdout items")
        print(f"  Columns: {list(holdout_df.columns)}")
        print(f"  Unique constructs: {holdout_df['label'].nunique()}")
    except Exception as e:
        print(f"✗ Error loading holdout items: {e}")
        
    # Test leadership data
    print("\n3. Testing leadership data:")
    try:
        leader_df = pd.read_csv("data/processed/leadership_focused_clean.csv")
        print(f"✓ Loaded {len(leader_df)} leadership items")
        print(f"  Original columns: {list(leader_df.columns)}")
        
        # Test column renaming
        if 'ProcessedText' in leader_df.columns:
            leader_df = leader_df.rename(columns={'ProcessedText': 'text', 'StandardConstruct': 'label'})
            print(f"✓ Renamed columns successfully")
            print(f"  New columns: text={'text' in leader_df.columns}, label={'label' in leader_df.columns}")
        
        print(f"  Unique constructs: {sorted(leader_df['label'].unique())}")
        print(f"  Items per construct:")
        print(leader_df['label'].value_counts().sort_index())
        
    except Exception as e:
        print(f"✗ Error loading leadership data: {e}")
    
    print("\n" + "="*50)
    print("All data format tests complete!")

if __name__ == "__main__":
    test_data_loading()