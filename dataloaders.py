import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from PIL import Image 
import albumentations as A
from pathlib import Path
import pandas as pd
import numpy as np 
from config import cfg



LOW  = np.exp(-15 / 10)
HIGH = np.exp(5 / 10)


class AporeeDataset(Dataset):
    def __init__(self, root, filter_fn=None, augment=False, max_samples=None,is_vit_for_audio=True):
        super().__init__()
        self.root = Path(root)
        self.meta = pd.read_csv(self.root / 'metadata.csv')
        self.is_vit_for_audio = is_vit_for_audio
        

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.unnormalize = Unnormalize(mean=mean, std=std)
        self.maxlen = max_samples

        if augment and 'image' in cfg.AugmentationMode:
            self.imgtransform = A.Compose([
                A.CenterCrop(512, 512),
                A.RandomResizedCrop(224, 224, scale=[0.5, 1.0]),
                A.Rotate(limit=180, p=1.0),
                A.Blur(blur_limit=3),
                A.GridDistortion(),
                A.HueSaturationValue(),
                A.Normalize(mean=0.0,std=1.0),
            ])
        else:
            self.imgtransform = A.Compose([
                A.CenterCrop(384, 384),
                A.Resize(224, 224),
                A.Normalize(mean=0.0,std=1.0)
            ])
        if cfg.AudioAugmentationMode and self.is_vit_for_audio:
          self.audiotransform = A.Compose([
            A.Resize(224,224),
            A.Normalize(mean=0.0,std=1.0)
          ])


        # join and merge
        img_present = set(int(f.stem) for f in (self.root).glob('images/*.jpg'))
        snd_present = set(int(f.stem) for f in (self.root).glob('spectrograms/*.jpg'))
        keys_present = img_present.intersection(snd_present)
        self.meta = self.meta[self.meta.short_key.isin(keys_present)]
        if filter_fn:
            self.meta = self.meta[self.meta.short_key.apply(filter_fn)]
        self.meta = self.meta.reset_index(drop=True)
        
        self.key2idx = {v: i for i, v in enumerate(self.meta.key)}
        self.augment = augment
        print('Number of Samples:', len(self.meta))

    def get_asymmetric_sampler(self, batch_size, asymmetry):
        lon = np.radians(self.meta.longitude.values)
        lat = np.radians(self.meta.latitude.values)

        coords = np.stack([
            np.cos(lon) * np.cos(lat),
            np.sin(lon) * np.cos(lat),
            np.sin(lat),
        ], axis=1)

        return AsymmetricSampler(coords, asymmetry, batch_size)

    def get_batch(self, keys):
        true_indices = map(self.key2idx.get, keys)
        return self.collate([self[i] for i in true_indices])

    def collate(self, batch):
        key, img, audio, audio_split, v = zip(*batch)

        key = torch.tensor(key)
        img = torch.stack(img, dim=0)
        
        audio = torch.cat(audio, dim=0).unsqueeze(1)
        
        if self.is_vit_for_audio:
          audio = audio.repeat(1,3,1,1)
          audio_split = None
        else:
          audio_split = audio_split

        v = torch.stack(v, dim=0)

        return key, img, audio, audio_split, v

    def __getitem__(self, idx):
        sample = self.meta.iloc[idx]
        key = sample.short_key

        # img = cv2.imread(str(self.root / 'images' / f'{key}.jpg'))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.array(Image.open(self.root / 'images' / f'{key}.jpg'))
        img = self.imgtransform(image=img)['image']
        img = torch.from_numpy(img).permute(2, 0, 1)

        # audio = cv2.imread(str(self.root / 'spectrograms' / f'{key}.jpg')).astype(np.float32)
        audio = np.array(Image.open(self.root / 'spectrograms' / f'{key}.jpg')).astype(np.float32)
        audio = audio * ((HIGH - LOW) / 255) + LOW

        if audio.shape[1] > 128 * self.maxlen:
            start = int(torch.randint(0, audio.shape[1] - 128*self.maxlen, []))
            audio = audio[:, start:start+128*self.maxlen]
        
        if self.is_vit_for_audio:
          audio = self.audiotransform(image=audio)['image']
          audio = audio.reshape(224,-1,224).transpose(1,0,2)
        else:
          audio = audio.reshape(128, -1, 128).transpose(1, 0, 2)
        
        audio = torch.from_numpy(audio)

        lon = np.radians(sample.longitude)
        lat = np.radians(sample.latitude)
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        v = torch.from_numpy(np.stack([x, y, z])).float()

        return [key, img, audio, audio.shape[0], v]

    def __len__(self):
        return len(self.meta)

class AsymmetricSampler(torch.utils.data.Sampler):
    def __init__(self, coords, asymmetry, batch_size):
        self.coords = coords
        self.asymmetry = asymmetry
        self.batch_size = batch_size
        self.knn = NearestNeighbors(n_neighbors=batch_size)
        self.knn.fit(self.coords)

    def sample_around(self, start):
        batch_idx = set([start])
        offset = self.asymmetry * self.coords[start]
        while len(batch_idx) < self.batch_size:
            X = torch.randn([1, 3]).numpy() + offset
            X = X / np.linalg.norm(X, ord='fro')
            _, candidates = self.knn.kneighbors(X)
            indices = (int(c) for c in candidates[0])
            batch_idx.add(next(i for i in indices if i not in batch_idx))

        return list(batch_idx)

    def rand(self):
        return int(torch.randint(0, self.coords.shape[0], []))

    def __iter__(self, ):
        for i in range(len(self)):
            if i % 2 == 0:
                start = self.rand()
                yield self.sample_around(start)
            else:
                yield [self.rand() for _ in range(self.batch_size)]

    def __len__(self, ):
        return self.coords.shape[0] // self.batch_size

class Unnormalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean).reshape(1, -1, 1, 1)
        self.std = torch.tensor(std).reshape(1, -1, 1, 1)

    def __call__(self, tensor):
        return tensor.mul(self.std).add(self.mean)


def get_loader(batch_size, mode, num_workers=2, asymmetry=0, max_samples=100,data_root='./datasets/SoundingEarth/data',is_vit_for_audio=False):
    FACTOR = 10
    filter_fn = {
        'train': lambda x: (x%FACTOR) not in (7, 5, 2),
        'val':   lambda x: (x%FACTOR) == 7,
        'test':  lambda x: (x%FACTOR) in (2, 5),
        'toy':  lambda x: (x%1000) in (2, 5),
        'all':   lambda x: True
    }.get(mode)
    is_train = (mode == 'train')
    dataset = AporeeDataset(root=data_root, filter_fn=filter_fn, augment=is_train, max_samples=max_samples,is_vit_for_audio=is_vit_for_audio)
    loader_args = dict(
        batch_size = batch_size,
        pin_memory = False,
        num_workers = num_workers,
        shuffle = is_train,
        collate_fn = dataset.collate,
        drop_last = True,
        prefetch_factor=2
    )
    if asymmetry != 0:
        loader_args['batch_sampler'] = dataset.get_asymmetric_sampler(batch_size, asymmetry)
        del loader_args['batch_size']
        del loader_args['shuffle']
        del loader_args['drop_last']
    return DataLoader(dataset, **loader_args)

if __name__=="__main__":
    # imDataset = ImageDataset()
    # imDataloader = DataLoader(dataset=imDataset.x,batch_size=10,shuffle=False)
    # sndDataset = SoundDataset()
    # sndDataloader = DataLoader(dataset=sndDataset.x,batch_size=10,shuffle=False)

    #get dataloader
    loader = get_loader(batch_size=32,mode="val")

    # print("metadata length: ",len(loader))

    #get dataset 
    ds = AporeeDataset(root='./datasets/SoundingEarthData/data',max_samples=50)
    #loader = DataLoader(ds)

    #key0,img0,audio0,audioshape0,v0 = ds[0]
    for i in range(4):
        k,i,a,asp,v = ds[i]
        print(f'key:{k},img:{i.shape},audio:{a.shape},audioshape:{asp},v:{v}')


    # for i,inputs in enumerate(loader):
    #     if i>1:
    #         break
    #     print(f"{i} \
    #     input {inputs[0].shape}: img_shape:{inputs[1]},snd_shape:{inputs[2]}\
    #         audio_splits:{inputs[3]}, dis: {inputs[4]}")


   
