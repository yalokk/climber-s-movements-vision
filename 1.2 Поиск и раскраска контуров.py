import cv2
from matplotlib import pyplot as pl
import imutils
from random import randint


# Импортируем изображение
directory = 'images2/cont_climb4.jpg'
img = cv2.imread(directory)
orig_image = img.copy()

# Уменьшаем медианный шум - удаляем точки
img = cv2.medianBlur(img, 7)
# pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# pl.show()

# Уменьшаем шум на фото, но сохраняем контуры
img = cv2.bilateralFilter(img, 10, 35, 45)
# pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# pl.show()

# Переводим в оттенки серого
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Выделяем контуры
edges = cv2.Canny(gray, 30, 200)

pl.imshow(edges)
pl.show()

contours = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)

contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# Раскрашиваем контуры разными цветами

i = 0
for c in contours:
    # cv2.drawContours(img, contours, -1, (5 + i*20, 0, 0), 3)
    cv2.drawContours(img, contours, i, (randint(0, 255), randint(0, 255), randint(0, 255)), 2)
    print(i)
    print(c[0])
    i += 1

# Выводим изображение через matplotlib, переводим в rgb
pl.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
pl.show()

# Количество контуров
# print(len(cont))'
