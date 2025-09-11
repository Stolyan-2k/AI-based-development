import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


saturation_scale=0.1
directory='./images'
 
# Обрабатывает изображение: изменяет насыщенность.
def process_image(image_path: str, saturation_scale: float = 1.5):

    image_bgr = cv2.imread(image_path)

    if image_bgr is None:
        print(f"Ошибка: не удалось загрузить '{image_path}'")
        return None

    hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] *= saturation_scale
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)

    result_bgr = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return image_rgb, result_rgb

# Отображает оригинальное и обработанное изображение рядом.
def show_images(original, processed, title: str, saturation_scale: float):
      
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(original)
    plt.title("Оригинал")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(processed)
    plt.title(f"{title} — насыщенность x{saturation_scale}")
    plt.axis("off")

    plt.show()

# Обрабатывает все изображения в директории и выводит результат.
def process_directory(directory: str, saturation_scale: float = 1.5):
    
    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_ext):
            path = os.path.join(directory, filename)
            result = process_image(path, saturation_scale)

            if result is not None:
                original, processed = result
                show_images(original, processed, filename, saturation_scale)


# Пример использования
process_directory(directory, saturation_scale)