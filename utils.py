import os

def save_output(content, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)



def ensure_directories():
    directories = [
        "outputs/summaries",
        "outputs/gap_reports",
        "vector_store"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)