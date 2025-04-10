from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple
import torchvision
from sklearn.model_selection import train_test_split


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_data(train_size: int = 0.8, batch_size: int = 16) -> Tuple[DataLoader]:
    # Загрузим датасеты для train и test
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)

    train_size = int(train_size * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=val_size, random_state=42, shuffle=True)

    # Загрузчики данных
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader