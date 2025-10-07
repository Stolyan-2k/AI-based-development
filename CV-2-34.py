import cv2
import numpy as np
import os

# Настраиваемые параметры 
SHADOW_STRENGTH = 1        # сила тени
SCALE_Y = 0.15             # сжатие тени по вертикали
DX = 40                    # сдвиг тени вправо
DY_EXTRA = 10              # дополнительный сдвиг вниз
MASK_BLUR_KERNEL = (5, 5)  # сглаживание маски объекта
MASK_BLUR_SIGMA = 5
SHADOW_BLUR_KERNEL = (41, 41)
SHADOW_BLUR_SIGMA = 40
FINAL_BLUR_KERNEL = (101, 101)
FINAL_BLUR_SIGMA = 30


# Функция обработки одного изображения
def add_soft_shadow(
    image_path: str,
    output_path: str = "result.png"
):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Файл не найден: {image_path}")

    h, w = image.shape[:2]

    # 1. Маска объекта
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    mask = (mask > 0).astype(np.uint8) * 255

    # 2. Размытие маски для мягких краев
    mask_blur = cv2.GaussianBlur(mask, MASK_BLUR_KERNEL, MASK_BLUR_SIGMA)

    # 3. Размытие маски для тени
    shadow_mask = cv2.GaussianBlur(mask, SHADOW_BLUR_KERNEL, SHADOW_BLUR_SIGMA)

    # 4. Сжимаем тень по вертикали
    M_scale = np.array([[1, 0, 0], [0, SCALE_Y, 0]], dtype=np.float32)
    shadow_mask_scaled = cv2.warpAffine(
        shadow_mask, M_scale, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 5. Bounding box объекта
    ys, xs = np.where(mask > 0)
    y_min, y_max = ys.min(), ys.max()

    # 6. Сдвигаем тень вниз и вправо
    ys_s, _ = np.where(shadow_mask_scaled > 0)
    shadow_bottom = ys_s.max() if len(ys_s) > 0 else 0
    dy = (y_max - shadow_bottom) + DY_EXTRA
    M_shift = np.float32([[1, 0, DX], [0, 1, dy]])
    shadow_mask_final = cv2.warpAffine(
        shadow_mask_scaled, M_shift, (w, h),
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    # 7. Сглаживание тени
    shadow_mask_final = cv2.GaussianBlur(
        shadow_mask_final, FINAL_BLUR_KERNEL, FINAL_BLUR_SIGMA
    )

    # 8. Применяем тень к фону (безопасное вычисление shadow_alpha)
    s = float(SHADOW_STRENGTH)


    # создаём альфа-маску с ограничением диапазона 0..1
    shadow_alpha = shadow_mask_final.astype(np.float64) / 255.0
    shadow_alpha = np.clip(shadow_alpha * s, 0.0, 1.0).astype(np.float32)

    background = np.full((h, w, 3), 255, dtype=np.float32)
    background_dark = (background * (1.0 - shadow_alpha[:, :, None])).astype(np.uint8)

    # 9. Накладываем объект
    mask_blur_float = mask_blur.astype(np.float32) / 255.0
    foreground = (image.astype(np.float32) * mask_blur_float[:, :, None]).astype(np.uint8)
    background_part = (background_dark.astype(np.float32) * (1.0 - mask_blur_float[:, :, None])).astype(np.uint8)
    result = cv2.add(foreground, background_part)

    # 10. Сохраняем результат
    cv2.imwrite(output_path, result)
    print(f"Сохранено: {output_path}")


# Функция для обработки всех изображений в папке
def process_folder(
    input_folder: str,
    output_folder: str = "results"
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    supported_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(supported_ext):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"shadow_{filename}")
            try:
                add_soft_shadow(input_path, output_path)
            except Exception as e:
                print(f"Ошибка при обработке {filename}: {e}")

    print("🎉 Обработка завершена!")


# Пример использования:
# process_folder("images")  # обработает все изображения из папки "images"
# или:
# add_soft_shadow("images (2).jpg", "result.png")
