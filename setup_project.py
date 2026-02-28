import os

# All folders to create
folders = [
    "src/data",
    "src/features",
    "src/models",
    "src/explainability",
    "src/api",
    "dashboard",
    "tests",
    "notebooks",
    "mlruns",
    "data/raw",
    "data/processed",
]

# All empty _init_.py files needed
init_files = [
    "src/_init_.py",
    "src/data/_init_.py",
    "src/features/_init_.py",
    "src/models/_init_.py",
    "src/explainability/_init_.py",
    "src/api/_init_.py",
    "tests/_init_.py",
]

def create_structure():
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"✅ Created folder: {folder}")

    for filepath in init_files:
        with open(filepath, "w") as f:
            f.write("")
        print(f"✅ Created file: {filepath}")

    print("\n🎉 Project structure ready!")

if __name__ == "__main__":
    create_structure()