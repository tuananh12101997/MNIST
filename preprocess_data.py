import torchvision
import PIL.Image as Image
import os
import numpy as np
import cv2
from config import get_arguments


def main():
    opt = get_arguments().parse_args()
    ds_train = torchvision.datasets.MNIST(opt.dataroot, train=True, download=True)
    ds_test = torchvision.datasets.MNIST(opt.dataroot, train=False, download=False)
    dir_train = os.path.join(opt.dataroot, 'train')
    dir_test = os.path.join(opt.dataroot, 'test')
    
    try:
        os.mkdir(dir_train)
        os.mkdir(dir_test)
        for i in range(10):
            os.mkdir(os.path.join(dir_train, str(i)))
    except:
        pass
    
    # Process train data
    with open(os.path.join(dir_train, 'annotation_train.txt'), 'w+') as f:
        for idx, (image, target) in enumerate(ds_train):
            image = np.asarray(image)
            image_path = os.path.join(dir_train, str(target), 'image_{}.png'.format(idx))
            cv2.imwrite(image_path, image)
            f.write(image_path + ',' + str(target) + '\n')
            
    # Process test data
    with open(os.path.join(dir_test, 'annotation_test.txt'), 'w+') as f:
        for idx, (image, target) in enumerate(ds_test):
            image = np.asarray(image)
            image_path = os.path.join(dir_test, 'image_{}.png'.format(idx))
            cv2.imwrite(image_path, image)
            f.write(image_path + ',' + str(target) + '\n')
            
            
if(__name__ == '__main__'):
    main()