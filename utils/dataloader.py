# Import system package
import pathlib


# Import dependencies
import PIL
from PIL import Image


from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T

ACCEPTED_IMAGE_EXTS = ['.jpg']

def get_transformation():
    '''
    torchvision transforms that take a list of `n` PIL image(s) and output
    tensor of shape `N x W X H X 3` \n
    Then following with normalize 
    and further transform will be applied.
    Return
    ------
    return torch image transforms
    '''
    transform = T.Compose(
        (
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),)
    )

    return transform

class MyDataLoader(Dataset):

    def __init__(self, image_root):
        self.image_root = pathlib.Path(image_root)
        self.image_list = list()
        for image_path in self.image_root.iterdir():
            if image_path.exists() and image_path.suffix.lower() in ACCEPTED_IMAGE_EXTS:
                self.image_list.append(image_path)
        self.image_list = sorted(self.image_list, key = lambda x: int(x.name.split('.')[0].split('_')[1]))
        self.transform = get_transformation()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        _img = self.image_list[index]
        _img = Image.open(_img)
        return self.transform(_img), str(self.image_list[index])
