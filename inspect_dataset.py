from datasets import load_dataset

def inspect():
    print("Loading dataset lmms-lab/flickr30k in streaming mode...")
    try:
        # Load the 'test' split in streaming mode
        dataset = load_dataset("lmms-lab/flickr30k", split="test", streaming=True)
        
        print("\nDataset loaded successfully.")
        
        # Get the first example
        first_example = next(iter(dataset))
        
        print("\n--- First Example Inspection ---")
        print(f"Keys (Column Names): {list(first_example.keys())}")
        
        print("\n--- Data Types and Values ---")
        for key, value in first_example.items():
            print(f"Column: {key}")
            print(f"  Type: {type(value)}")
            print(f"  Value Preview: {str(value)[:100]}...")  # Truncate long values
            print("-" * 30)
            
        # Check for filename or id
        potential_id_cols = [col for col in first_example.keys() if 'id' in col.lower() or 'name' in col.lower() or 'file' in col.lower()]
        print(f"\nPotential ID/Filename columns found: {potential_id_cols}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    inspect()
