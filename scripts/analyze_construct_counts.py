import csv
from collections import Counter

def analyze_csv_constructs(file_path, construct_column_index):
    """
    Analyzes a CSV file to count items per construct.

    Args:
        file_path (str): The path to the CSV file.
        construct_column_index (int): The 0-based index of the column 
                                      containing the construct names.
    """
    construct_counts = Counter()
    
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)  # Skip header row
            
            print(f"CSV Header: {header}")
            if construct_column_index >= len(header):
                print(f"Error: construct_column_index {construct_column_index} is out of bounds for header length {len(header)}.")
                return

            print(f"Counting items based on column: '{header[construct_column_index]}'")
            
            for row_number, row in enumerate(reader, 1): # Start row_number from 1 for data rows
                if len(row) > construct_column_index:
                    construct = row[construct_column_index].strip()
                    if construct: # Ensure construct is not empty
                        construct_counts[construct] += 1
                    else:
                        print(f"Warning: Empty construct found in row {row_number + 1}") # +1 to account for header
                else:
                    print(f"Warning: Row {row_number + 1} is shorter than expected. Length: {len(row)}, Expected at least: {construct_column_index + 1}")

        print("\n--- Construct Counts ---")
        total_items = 0
        for construct, count in construct_counts.items():
            print(f"{construct}: {count} items")
            total_items += count
        print("------------------------")
        print(f"Total items counted: {total_items}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Configuration
    FILE_TO_ANALYZE = "data/processed/leadership_focused_clean.csv"
    # Based on header: MeasureID,Behavior,Dimensions,Item,Xnum,Text,StandardConstruct,ProcessedText
    # 'StandardConstruct' is the 7th column, so index is 6.
    CONSTRUCT_COLUMN_IDX = 6 
    
    analyze_csv_constructs(FILE_TO_ANALYZE, CONSTRUCT_COLUMN_IDX) 