from flask import Flask, request, jsonify
import torch
import torch.nn.functional as F
import pandas as pd
from model.net import Baseline  # Импорт вашей модели

# Инициализация Flask-приложения
app = Flask(__name__)

# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Baseline()
model.load_state_dict(torch.load('model.pth', map_location=device))
model.to(device)
model.eval()

# Эндпоинт для предсказаний
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()                 # Получаем данные
    df = pd.DataFrame(data)                   # Преобразуем в DataFrame
    inputs = torch.tensor(df.values, dtype=torch.float32).to(device)  # В тензор

    with torch.no_grad():
        outputs = model(inputs)               # Прямой проход
        if outputs.shape[1] > 1:
            predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        else:
            predictions = (outputs > 0.5).int().squeeze()

    return jsonify({'prediction': predictions.cpu().tolist()})

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
