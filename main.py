from image_miner import downloader

query = input("Enter the search query: ")
num_images = int(input("Enter the number of images: "))

downloader.download(
    query,
    limit=num_images,
    output_dir='dataset',
    adult_filter_off=False,
    force_replace=False,
    timeout=60,
    verbose=True
)
