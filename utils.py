import os

def save_output(content, file_path):
    """
    Save content to file with proper error handling.
    """
    try:
        # Ensure directory exists
        directory = os.path.dirname(file_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Write file
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        print(f"✅ Output saved: {file_path}")
        
    except IOError as e:
        print(f"❌ Failed to save output to {file_path}: {e}")
        raise


def ensure_directories():
    """
    Create necessary output directories.
    """
    directories = [
        "outputs/summaries",
        "outputs/gap_reports",
        "data/raw_papers",  # Added: needed for CLI
        "data/vector_store"  # Fixed: matches config.py path
    ]

    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Directory ready: {directory}")
        except Exception as e:
            print(f"⚠️  Could not create {directory}: {e}")
