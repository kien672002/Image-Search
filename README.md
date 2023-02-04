# Image-Search

The Image Search Program is a tool for quickly and efficiently locating specific images from a large set of provided images. With its advanced search algorithms, it allows users to easily find the image they need with just a few clicks.

The program's core features include:

- Advanced search algorithms for quick results
- Fast and efficient retrieval of images even from large datasets

## How to use this program:

First install essential libraries:
```
pip3 install -r requirements.txt
```

Place the provided images folder in an easily accessible location and ensure that the images are formatted as `img_*.jpg` (you can use a simple script for this).

Run `indexing.py` to index images, which will be saved in the `data_index.bin` file using the command below:: 

```
python3 indexing.py --image_root YOUR_IMAGES_FOLDER_PATH
```

Then run the following command for your image searching : 

```
python3 main.py --image_root YOUR_IMAGES_FOLDER_PATH --test_image_path PATH_TO_IMAGE_YOU_WANT_TO_SEARCH --top_k HOW_MANY_IMAGES_WILL_BE_SHOWN
```






