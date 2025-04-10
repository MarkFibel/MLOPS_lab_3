import requests
import torchvision
from torchvision import transforms
import torch

# Применяем преобразования к тестовым данным
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

url = 'http://localhost:65535/predict'
batch_size = 16  # Устанавливаем размер батча
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Получаем несколько изображений для отправки
data = []

# Получаем данные из датасета в нужной форме
for images, _ in test_loader:
    # Убедимся, что у нас есть правильный формат (2, 3, 32, 32)
    data.append(images)  # Добавляем целый батч

    if len(data) >= 1:  # Ограничиваем себя одной партией
        break

# Преобразуем в нужный формат перед отправкой
batch_data = data[0].detach().cpu().numpy().tolist()  # Порядок: [количество изображений, каналы, высота, ширина]
# Отправка POST-запроса с данными
response = requests.post(url, json=batch_data)

# Обработка ответа
if response.status_code == 200:
    prediction = response.json()
    print('Predictions:', prediction['prediction'])
else:
    print('Error:', response.status_code, response.text)