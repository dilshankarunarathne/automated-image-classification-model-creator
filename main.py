import image_miner

query = input("Enter the search query: ")
num_images = int(input("Enter the number of images: "))

image_miner.downloader.download(
    query,
    limit=num_images,
    output_dir='dataset',
    adult_filter_off=False,
    force_replace=False,
    timeout=60,
    verbose=True
)
