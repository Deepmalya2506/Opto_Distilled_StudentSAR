from huggingface_hub import HfApi

api = HfApi()
repo = "paulhoehn/Sen12Landslides"

for folder in ["data_harmonized/s1asc", "data_harmonized/s2"]:
    print(f"\n===== {folder} =====")
    total = 0
    count = 0

    for f in api.list_repo_tree(
        repo_id=repo,
        repo_type="dataset",
        path_in_repo=folder,
        recursive=True
    ):
        path = getattr(f, "path", "")
        if path.endswith(".tar.gz"):
            size = getattr(f, "size", None)
            count += 1
            if size is not None:
                total += size
                print(f"{size / 1e9:.3f} GB | {path}")
            else:
                print(f"Unknown size | {path}")

    print(f"Archive count: {count}")
    print(f"Known total: {total / 1e9:.3f} GB")
