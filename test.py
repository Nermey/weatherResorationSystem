from PIL import Image

from main import WeatherRestorationSystem
from ultralytics import YOLO


weather_restoration = WeatherRestorationSystem(device='mps')

model = YOLO('yolo11n')
model.to('mps')

results = model.predict('./DAWN/foggy-033.jpg')

for r in results:
    path = r.path.split('/')[-1]  # Имя файла
    boxes = r.boxes  # Информация о границах и классах

    if len(boxes) == 0:
        print(f"{path:<20} | {'[Ничего не найдено]':<15} | -")
        continue

    for box in boxes:
        # Извлекаем ID класса и его текстовое имя
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Извлекаем вероятность (confidence score)
        conf = float(box.conf[0])

        # Печатаем понятную строку
        print(f"{path:<20} | {class_name:<15} | {conf:.2%}")

print("=" * 50 + "\n")

img = Image.open('./DAWN/foggy-033.jpg').convert('RGB')
new_results = model.predict(weather_restoration.restore(img))

for r in new_results:
    path = r.path.split('/')[-1]  # Имя файла
    boxes = r.boxes  # Информация о границах и классах

    if len(boxes) == 0:
        print(f"{path:<20} | {'[Ничего не найдено]':<15} | -")
        continue

    for box in boxes:
        # Извлекаем ID класса и его текстовое имя
        class_id = int(box.cls[0])
        class_name = model.names[class_id]

        # Извлекаем вероятность (confidence score)
        conf = float(box.conf[0])

        # Печатаем понятную строку
        print(f"{path:<20} | {class_name:<15} | {conf:.2%}")

print("=" * 50 + "\n")