from argparse import ArgumentParser
import tqdm
import time

import faiss
import torch
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler


from backbone.resnet import feature_extractor
from utils.dataloader import MyDataLoader


from indexer.faiss_indexer import get_faiss_indexer


def main():

    def batch_drive(images):
        images = images.to(device)

        with torch.no_grad():
            features = model(images)
            features = features.view((images.size(0), -1))
        # print(features.size())

        indexer.add(features.cpu().detach().numpy())


    parser = ArgumentParser()
    parser.add_argument("--image_root", type=str, default='images')
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    batch_size = args.batch_size

    model = feature_extractor()
    model = model.to(device)
    model.eval()

    dataset = MyDataLoader(args.image_root)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sampler, num_workers=8)

    indexer = get_faiss_indexer(dimension_size=1024)
    for indices, (images, image_paths) in tqdm.tqdm(enumerate(dataloader)):
        batch_drive(images=images)

    faiss.write_index(indexer, 'data_index.bin')


if __name__ == "__main__":
    main()
