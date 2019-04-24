import torch
import torch.utils.data as data
from PIL import Image
import os
import os.path
import random
import numpy as np

def _pil_loader(path, cropArea=None, resizeDim=None, frameFlip=0):
    with open(path, 'rb') as f:
        img = Image.open(f)
        # Resize image if specified.
        resized_img = img.resize(resizeDim, Image.ANTIALIAS) if (resizeDim != None) else img
        # Crop image if crop area specified.
        cropped_img = img.crop(cropArea) if (cropArea != None) else resized_img
        # Flip image horizontally if specified.
        flipped_img = cropped_img.transpose(Image.FLIP_LEFT_RIGHT) if frameFlip else cropped_img
        #return flipped_img.convert('RGB')
        tensorImg = torch.FloatTensor(np.array(flipped_img)[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0))
        return tensorImg

    
class SepConvTrain(data.Dataset):
    def __init__(self, root, transform=None, dim=(640, 360), randomCropSize=(256, 256), train=True):
        framesPath = []
        with open(root, 'r') as document:
            index = 0
            for line in document:
                line = line.split()
                if not line:
                    continue
                if not (os.path.isdir(line[0])):
                    continue
                framesPath.append([])
                record = [int(el) for el in line[2:]]
                seqlen = record[1] - record[0] + 1
                for idx in range(seqlen):
                    picname = "%05d.jpg"%(idx + record[0])
                    imagename = os.path.join(line[0],picname)
                    framesPath[index].append(imagename)
                index += 1
        if len(framesPath) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"))
                
        self.randomCropSize = randomCropSize
        self.cropX0         = dim[0] - randomCropSize[0]
        self.cropY0         = dim[1] - randomCropSize[1]
        self.root           = root
        self.transform      = transform
        self.train          = train
        self.framesPath     = framesPath

    def __getitem__(self, index):
        sample = []
        # 减少帧的选取区间
        istep = 3
        if (self.train):
            ### Data Augmentation ###
            firstFrame = 0
            # Apply random crop
            cropX = random.randint(0, self.cropX0)
            cropY = random.randint(0, self.cropY0)
            cropArea = (cropX, cropY, cropX + self.randomCropSize[0], cropY + self.randomCropSize[1])
            # Random reverse frame
            IFrameIndex = random.randint(firstFrame + 1, firstFrame + istep - 1)
            if (random.randint(0, 1)):
                frameRange = [firstFrame, IFrameIndex, firstFrame + istep]
                returnIndex = IFrameIndex - firstFrame - 1
            else:
                frameRange = [firstFrame + istep, IFrameIndex, firstFrame]
                returnIndex = firstFrame - IFrameIndex + istep - 1
            # Random flip frame
            randomFrameFlip = random.randint(0, 1)
        else:
            # Fixed settings to return same samples every epoch.
            # For validation/test sets.
            firstFrame = 0
            cropArea = (0, 0, self.randomCropSize[0], self.randomCropSize[1])
            IFrameIndex = ((index) % (istep - 1)  + 1)
            returnIndex = IFrameIndex - 1
            frameRange = [0, IFrameIndex, istep]
            randomFrameFlip = 0
        
        # Loop over for all frames corresponding to the `index`.
        for frameIndex in frameRange:
            # Open image using pil and augment the image.
            image = _pil_loader(self.framesPath[index][frameIndex], cropArea=cropArea, frameFlip=randomFrameFlip)
            # Apply transformation if specified.
            if self.transform is not None:
                image = self.transform(image)
            sample.append(image)
        return sample, returnIndex

    def __len__(self):
        return len(self.framesPath)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str