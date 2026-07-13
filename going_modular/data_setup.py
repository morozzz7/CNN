from torchvision import transforms, datasets
from torch.utils.data import DataLoader

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    train_transforms: transforms.Compose,
    test_transforms: transforms.Compose,
    batch_size: int
):
    train_data = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    class_names = train_data.classes
    return train_dataloader, test_dataloader, class_names
