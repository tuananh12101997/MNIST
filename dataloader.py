import cv2
import torch 
import torchvision
import os
from PIL import Image
from config import get_arguments


class MNIST(torch.utils.data.Dataset):
    def __init__(self, opt, train=True, transforms=None):
        super(MNIST, self).__init__()
        if(train):
            mode = 'train'
        else:
            mode = 'test'
        path_annotation = os.path.join(opt.dataroot, mode, 'annotation_{}.txt'.format(mode))
        
        with open(path_annotation, 'r+') as f:
            self.list_data = f.read().split('\n')
        self.list_data.remove('')
    
        self.transforms = transforms
        
    def __len__(self):
        return len(self.list_data)
    
    def __getitem__(self, idx):
        path, target = self.list_data[idx].split(',')
        target = torch.tensor(int(target))
        image = Image.open(path)
        if(self.transforms):
            image = self.transforms(image)
        return image, target
    

def get_dataloader(opt, train=True, shuffle=True):
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    ds_mnist = MNIST(opt, train, transforms)
    dataloader = torch.utils.data.DataLoader(ds_mnist, batch_size=opt.bs, num_workers=opt.num_workers, shuffle=shuffle)
    return dataloader
    
    
def main():
    opt = get_arguments().parse_args()
    transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    dl = get_dataloader(opt, False)
    inputs, targets = next(iter(dl))
    print(inputs.shape)
    
    
if(__name__ == '__main__'):
    main()