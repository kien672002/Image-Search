import pathlib
from argparse import ArgumentParser


from PIL import Image

import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler

import faiss


from backbone.resnet import feature_extractor
from utils.dataloader import get_transformation


ACCEPTED_IMAGE_EXTS = ['.jpg']


def get_image_list(image_root):
    image_root = pathlib.Path(image_root)
    image_list = list()
    for image_path in image_root.iterdir():
        if image_path.exists() and image_path.suffix.lower() in ACCEPTED_IMAGE_EXTS:
            image_list.append(image_path)
    image_list = sorted(image_list, key=lambda x: int(
        x.name.split('.')[0].split('_')[1]))
    return image_list


def visualize_result(image_paths):
    for image_path in image_paths:
        pil_image = Image.open(pathlib.Path(image_path))
        pil_image.show()


def main():
    parser = ArgumentParser()
    parser.add_argument("--image_root", required=True, type=str)
    parser.add_argument("--faiss_bin_path", required=True, type=str)
    parser.add_argument("--test_image_path", required=True, type=str)
    parser.add_argument("--top_k", required=False, type=int, default=11)
    parser.add_argument("--visual", required=False, type=bool, default=False)
    args = parser.parse_args()

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = feature_extractor()
    model = model.to(device)

    img_list = get_image_list(args.image_root)

    transform = get_transformation()

    test_image_path = pathlib.Path(args.test_image_path)
    pil_image = Image.open(test_image_path)
    image_tensor = transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        feat = model(image_tensor)
        feat = feat.view((image_tensor.size(0), -1))
        
    indexer = faiss.read_index(args.faiss_bin_path)

    distances, indices = indexer.search(
        feat.cpu().detach().numpy(), k=args.top_k)
    print(distances, indices)
    indices = indices[0]
    distances = distances[0]
    for ii, index in enumerate(indices):
        print(img_list[index], distances[ii])

    if args.visual == True:
        visualize_result([img_list[index] for index in indices])


if __name__ == '__main__':
    main()
