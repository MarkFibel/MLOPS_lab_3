import requests
import json
import torchvision
from torchvision import transforms
import torch

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# URL Flask-сервиса
url = 'http://localhost:5000/predict'

batch_size = 1
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Пример данных для предсказания (замени на свои)
data = [
    {'feature':test_loader[0]},  # Пример 1
    {"feature":test_loader[1]}   # Пример 2
]

# Отправка POST-запроса с данными
response = requests.post(url, json=data)

# Обработка ответа
if response.status_code == 200:
    prediction = response.json()
    print('Predictions:', prediction['prediction'])
else:
    print('Error:', response.status_code, response.text)
