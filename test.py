from settings import settings
from loader.model_loader import loadrobust
# for loading robust version
from robustness.model_utils import make_and_restore_model
from robustness.datasets import DATASETS
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import torchvision

# target layers can be changed in settings file. For this sample, it will hook layer ["1", "4", "7", "9", "11"] of AlexNet-R
features_blobs = []
def hook_feature(module, input, output): 
    features_blobs.append(output.data.cpu().numpy())

global features_blobs
dataset = DATASETS['imagenet']('robustness/dataset')

# Downloaded the Alexnet wieght from here: https://drive.google.com/drive/u/0/folders/1KdJ0aK0rPjmowS8Swmzxf8hX6gU5gG2U
# And gave it is directory to MODEL_PATH.
model, checkpoint = make_and_restore_model(arch=settings.MODEL[:-2],
                                        dataset=dataset,parallel=settings.MODEL_PARALLEL,
                                        resume_path=settings.MODEL_PATH)
model = loadrobust(hook_feature, model, checkpoint, settings.FEATURE_NAMES)

def modells(img):
    sftmx, _ = model(img)
    return sftmx

# Transformation applied on subset of validation set images
TEST_TRANSFORMS_IMAGENET = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()])    
# Data is downloadable from another repo here: https://github.com/Gullesh/VanillaVsARobust
val_dataset = torchvision.datasets.ImageFolder('/home/gullesh/Desktop/VanillaVsARobust/validationSample',transform = TEST_TRANSFORMS_IMAGENET)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=2)




# Evaluating the model:

correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        images = images.cuda()
        # calculate outputs by running images through the network
        outputs = modells(images)
        # the class with the highest energy is what we choose as prediction
        predicted = np.argmax(outputs.cpu().data.numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on the '+str(total)+ ' test images: %d %%' % (
    100 * correct / total))

