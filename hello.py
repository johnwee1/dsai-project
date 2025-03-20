import kagglehub

# Download latest version
path = kagglehub.dataset_download("ricgomes/global-fashion-retail-stores-dataset")

# this downloaded to my local folder here at
#
# /Users/john/.cache/kagglehub/datasets/ricgomes/global-fashion-retail-stores-dataset/versions/24

print("Path to dataset files:", path)
