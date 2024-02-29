from torchvision.datasets import ImageFolder
from torchvision import transforms as tt
from torch.utils.data import DataLoader

#Initialize the data paths
data_path = "face-mask-12k-images-dataset\Face Mask Dataset"
train_path = "face-mask-12k-images-dataset\Face Mask Dataset\Train"
test_path = "face-mask-12k-images-dataset\Face Mask Dataset\Test"
val_path = "face-mask-12k-images-dataset\Face Mask Dataset\Validation"

# Data Augmentation
transforms = tt.Compose([
    tt.Resize((224,224)),
    tt.RandomHorizontalFlip(p=0.9),
    tt.ToTensor() ])


# Datasets with transformation
train_ds = ImageFolder(train_path,  transform=transforms)
val_ds = ImageFolder(val_path, transform=transforms)
test_ds = ImageFolder(test_path, transform=transforms)

#DataLoaders
batch_size = 32
train_dl = DataLoader(train_ds, batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size*2)
test_dl = DataLoader(test_ds, batch_size)