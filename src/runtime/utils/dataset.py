import os

import torch
from PIL import Image


# loader class for images
# list_path: path to text-file containing relative image paths
# data_root: root directory for image paths
class LaneDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, list_path, img_transform):
        super(LaneDataset, self).__init__()
        self.data_root = data_root
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = [line for line in f.readlines() if line != '\n']

        # exclude the incorrect path prefix '/' of CULane
        # os.path.join('/media', '/subdir/1.jpg') would return '/subdir/1.jpg'
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]

        # self.pretransform = None

    def __getitem__(self, index):
        # while this code work it doesnt seem like it is improving performance. leaving it here as reference

        # image = None
        # name = self.list[index].split()[0]
        #
        # if self.pretransform and self.pretransform[0] == name:
        #     image = self.pretransform[1].res
        #     print('using prepared image', flush=True)
        #
        # # prepare next img
        # if (index + 1) < len(self.list):
        #     next_name = self.list[index+1].split()[0]
        #     next_img_path = os.path.join(self.data_root, next_name)
        #     self.pretransform = (next_name, ThreadedImgTransform(next_img_path, self.img_transform))
        #
        # if image is not None:
        #     return image, name
        # else:
        #     img_path = os.path.join(self.data_root, name)
        #     img = Image.open(img_path)
        #
        #     # i would prefer doing the resize stuff once in process_frame() but sadly dataloader requires tensors
        #     # so for the image input it has tbd before the images are loaded into the data loader
        #     # otherwise it will throw an exception when trying to iterate through the dataset
        #     if self.img_transform is not None:
        #         img = self.img_transform(img)
        #
        #     return img, name
        #
        # class ThreadedImgTransform():
        #     def transform(self):
        #         img = Image.open(self.img_path)
        #         self.res = self.img_transform(img)
        #
        #     def __init__(self, img_path, img_transform):
        #         self.res = None
        #         self.img_path = img_path
        #         self.img_transform = img_transform
        #         Thread(target=self.transform).start()
        #
        # return

        name = self.list[index].split()[0]
        img_path = os.path.join(self.data_root, name)
        img = Image.open(img_path)

        # i would prefer doing the resize stuff once in process_frame() but sadly dataloader requires tensors
        # so for the image input it has tbd before the images are loaded into the data loader
        # otherwise it will throw an exception when trying to iterate through the dataset
        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)
