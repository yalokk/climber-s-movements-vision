import cv2
from matplotlib import pyplot as pl


# Импортируем изображение
directory = 'images2/climb7.png'
img = cv2.imread(directory)
orig_image = img.copy()

# Уменьшаем медианный шум - удаляем точки
img = cv2.medianBlur(img, 7)
pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pl.show()

# Уменьшаем шум на фото, но сохраняем контуры
img = cv2.bilateralFilter(img, 30, 35, 45)
pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pl.show()
