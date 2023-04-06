# credits: https://github.com/SHI-Labs/StyleNAT

from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T


class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=256):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img_bytes = txn.get(key)

        buffer = BytesIO(img_bytes)
        img = Image.open(buffer)
        img = self.transform(img)

        return img


def unnormalize(image):
    if image.dim() == 4:
        image[:, 0, :, :] = image[:, 0, :, :] * 0.229 + 0.485
        image[:, 1, :, :] = image[:, 1, :, :] * 0.224 + 0.456
        image[:, 2, :, :] = image[:, 2, :, :] * 0.225 + 0.406
    elif image.dim() == 3:
        image[0, :, :] = image[0, :, :] * 0.229 + 0.485
        image[1, :, :] = image[1, :, :] * 0.224 + 0.456
        image[2, :, :] = image[2, :, :] * 0.225 + 0.406
    else:
        raise NotImplemented(f"Can't handle image of dimension {image.dim()}, please use a 3 or 4 dimensional image")
    return image


def get_ffhq(evaluation=True, minus_one_one_norm=False, norm=True):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406] if not minus_one_one_norm else [0.5, 0.5, 0.5],
                            std=[0.229, 0.224, 0.225] if not minus_one_one_norm else [0.5, 0.5,
                                                                                      0.5]) if norm else T.Normalize(
        mean=0, std=1)
    transforms = [T.Resize((256, 256))]
    if not evaluation:
        transforms.append(T.RandomHorizontalFlip())
    transforms.append(T.ToTensor())
    transforms.append(normalize)
    transforms = T.Compose(transforms)

    dataset = MultiResolutionDataset(path="./ffhq.lmdb/",
                                     transform=transforms,
                                     resolution=256)

    return dataset
