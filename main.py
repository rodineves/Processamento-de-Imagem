from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import random
import numpy as np
import math

def negative(colored):
  l, c = colored.size

  for x in range(l):
    for y in range(c):
        pxl = colored.getpixel((x, y))

        if isinstance(pxl, tuple):
          inverted_pixel = tuple([255 - i for i in pxl])
        else:
          inverted_pixel = 255 - pxl

        colored.putpixel((x, y), inverted_pixel)
  return colored

def monochrome(colored):
  l, c = colored.size
  img = Image.new("RGB", (l, c))

  for x in range(l):
    for y in range(c):
      pxl = colored.getpixel((x, y))

      if isinstance(pxl, tuple):
        media = (pxl[1] + pxl[2] + pxl[0]) // 3

        img.putpixel((x, y), (media, media, media))
      else:
        img.putpixel((x, y), (pxl, pxl, pxl))

  return img

def binarization(colored):
  l, c = colored.size
  img = Image.new("RGB", (l, c))

  for x in range(l):
    for y in range(c):
      pxl = colored.getpixel((x, y))

      media = (pxl[1] + pxl[2] + pxl[0]) // 3

      if media < 128:
        img.putpixel((x, y), (0, 0, 0))
      else:
        img.putpixel((x, y), (255, 255, 255))

  return img

def thresholding(pb, thresholds):
  l, c = pb.size
  img = Image.new("RGB", (l, c))
  qtd = len(thresholds)

  if qtd == 1:
    for x in range(l):
      for y in range(c):
        pxl = pb.getpixel((x, y))

        media = (pxl[1] + pxl[2] + pxl[0]) // 3

        if media < thresholds[0]:
          img.putpixel((x, y), (0, 0, 0))
        else:
         img.putpixel((x, y), (255, 255, 255))
  else:
    for x in range(l):
      for y in range(c):
        pxl = pb.getpixel((x, y))

        media = (pxl[0] + pxl[1] + pxl[2]) // 3

        for i in range(qtd-1):
          if media <= thresholds[0]:
            img.putpixel((x, y), (0, 0, 0))

          if media >= thresholds[qtd-1]:
            img.putpixel((x, y), (255, 255, 255))
          if thresholds[i] <= media < thresholds[i+1]:
            img.putpixel((x, y), (thresholds[i], thresholds[i], thresholds[i]))

  return img

def slicing(pb):
  l, c = pb.size
  img = Image.new("RGB", (l, c))

  for x in range(l):
    for y in range(c):
      pxl = pb.getpixel((x, y))

      media = (pxl[0] + pxl[1] + pxl[2]) // 3

      if media <= 100 or media >= 150:
        img.putpixel((x, y), (0, 0, 0))
      else:
        img.putpixel((x, y), (media, media, media))

  return img

def contrast_flare(pb, idmenor, idmaior):
  l, c = pb.size
  img = Image.new("L", (l, c))
  pb = pb.convert('L')

  pixel_min = min(pb.getdata())
  pixel_max = max(pb.getdata())

  for x in range(l):
    for y in range(c):
      pxl = pb.getpixel((x, y))

      new_pxl = (pxl - pixel_min) * ((idmaior - idmenor) / (pixel_max - pixel_min)) + idmenor
      new_pxl = int(new_pxl)

      img.putpixel((x, y), (new_pxl))

  return img

def salt_and_pepper(pb):
  l, c = pb.size

  for x in range(l):
    for y in range(c):
      a = random.randint(0, 20)
      if a == 0:
        pb.putpixel((x, y), (0, 0, 0))
      if a == 1:
        pb.putpixel((x, y), (255, 255, 255))

  return pb

def histogram(pb):
  pb = pb.convert('L')

  histogram = {}
  for pixel in pb.getdata():
    histogram[pixel] = histogram.get(pixel, 0) + 1

  plt.bar(histogram.keys(), histogram.values())
  plt.show()

def gauss(pb):
  l, c = pb.size

  for x in range(l):
    for y in range(c):
       a = random.randint(0, 2)
       if a == 0:
        r = random.randint(0, 256)
        pb.putpixel((x, y), (r, r, r))

  return pb

def equalize(pb):
  pb = pb.convert('L') #So para ser um pixel :)
  l, c = pb.size
  total_pxl = l * c
  q = [0] * 256

  for x in range(l):
    for y in range(c):
      pxl = pb.getpixel((x, y))
      q[pxl] += 1

  for i in range(256):
    q[i] = q[i] / total_pxl
    q[i] = round(q[i], 2)

  s = [0] * 256
  for i in range(256):
    s[i] = round(sum(q[:i+1]) * 255)

  for x in range(l):
    for y in range(c):
      pxl = pb.getpixel((x, y))
      pb.putpixel((x, y), s[pxl])

  return pb

def mean(img, r):
    l, c = img.size
    new_img = Image.new("RGB", (l, c))
    img = img.convert("RGB")
    new_img = img.copy()
    for x in range(r, l - r):
      for y in range(r, c - r):
        soma_r, soma_g, soma_b = 0, 0, 0

        for i in range(x - r, x + r + 1):
          for j in range(y - r, y + r + 1):
            pxl = img.getpixel((i, j))
            soma_r += pxl[0]
            soma_g += pxl[1]
            soma_b += pxl[2]

        total = ((2 * r + 1) ** 2)
        media_r = int(soma_r / total)
        media_g = int(soma_g / total)
        media_b = int(soma_b / total)
        new_img.putpixel((x, y), (media_r, media_g, media_b))
    return new_img

def median(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  new_img = img.copy()
  for x in range(r, l - r):
    for y in range(r, c - r):
      tam_mask = ((2 * r + 1) ** 2)
      v = [0] * tam_mask
      cont = 0

      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          v[cont] = img.getpixel((i, j))
          cont=cont+1

      v = sorted(v)
      median = v[len(v) // 2]
      new_img.putpixel((x, y), (median))

  return new_img

def highpass(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma = 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma += pxl * filtro[i - (x - r)][j - (y - r)]
      new_img.putpixel((x, y), (int(soma)))

  return new_img

def line_hor(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro = np.array([[-1, -1, -1], [2, 2, 2], [-1, -1, -1]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma = 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma += pxl * filtro[i - (x - r)][j - (y - r)]
      new_img.putpixel((x, y), (int(soma)))

  return new_img

def line_vert(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro = np.array([[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma = 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma += pxl * filtro[i - (x - r)][j - (y - r)]
      new_img.putpixel((x, y), (int(soma)))

  return new_img

def line_45p(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro = np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma = 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma += pxl * filtro[i - (x - r)][j - (y - r)]
      new_img.putpixel((x, y), (int(soma)))

  return new_img

def line_45m(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro = np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma = 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma += pxl * filtro[i - (x - r)][j - (y - r)]
      new_img.putpixel((x, y), (int(soma)))

  return new_img

def roberts(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro1 = np.array([[0, 0, -1], [0, 1, 0], [0, 0, 0]])
  filtro2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 0]])

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma1, soma2 = 0, 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma1 += pxl * filtro1[i - (x - r)][j - (y - r)]
          soma2 += pxl * filtro2[i - (x - r)][j - (y - r)]

      novo_pxl = math.sqrt((soma1**2) + (soma2**2))
      if novo_pxl > 255:
        novo_pxl = 255
      if novo_pxl < 0:
        novo_pxl = 0
      new_img.putpixel((x, y), (int(novo_pxl)))

  return new_img

def sobel(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro1 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) / 4
  filtro2 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 4

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma1, soma2 = 0, 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma1 += pxl * filtro1[i - (x - r)][j - (y - r)]
          soma2 += pxl * filtro2[i - (x - r)][j - (y - r)]

      novo_pxl = math.sqrt((soma1**2) + (soma2**2))
      if novo_pxl > 255:
        novo_pxl = 255
      if novo_pxl < 0:
        novo_pxl = 0
      new_img.putpixel((x, y), (int(novo_pxl)))

  return new_img

def prewitt(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro1 = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]]) / 3
  filtro2 = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]) / 3

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma1, soma2 = 0, 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma1 += pxl * filtro1[i - (x - r)][j - (y - r)]
          soma2 += pxl * filtro2[i - (x - r)][j - (y - r)]

      novo_pxl = math.sqrt((soma1**2) + (soma2**2))
      if novo_pxl > 255:
        novo_pxl = 255
      if novo_pxl < 0:
        novo_pxl = 0
      new_img.putpixel((x, y), (int(novo_pxl)))

  return new_img

def feichen(img, r):
  l, c = img.size
  new_img = Image.new("L", (l, c))
  img = img.convert("L")
  filtro1 = np.array([[1, 0, -1],
                    [math.sqrt(2), 0, - math.sqrt(2)],
                      [1, 0, -1]]) / (2 + math.sqrt(2))
  filtro2 = np.array([[-1, - math.sqrt(2), -1],
                      [0, 0, 0],
                       [1, math.sqrt(2), 1]]) / (2 + math.sqrt(2))

  for x in range(r, l - r):
    for y in range(r, c - r):
      soma1, soma2 = 0, 0
      for i in range(x - r, x + r + 1):
        for j in range(y - r, y + r + 1):
          pxl = img.getpixel((i, j))
          soma1 += pxl * filtro1[i - (x - r)][j - (y - r)]
          soma2 += pxl * filtro2[i - (x - r)][j - (y - r)]

      novo_pxl = math.sqrt((soma1**2) + (soma2**2))
      if novo_pxl > 255:
        novo_pxl = 255
      if novo_pxl < 0:
        novo_pxl = 0
      new_img.putpixel((x, y), (int(novo_pxl)))

  return new_img

img = Image.open('image/gray.jpg')

print('Put a image in your past, and then choose a preprocesing selecting the number:')
print('1 - Negative                | 12 - Low-pass (Median)')
print('2 - Monochrome              | 13 - High-pass')
print('3 - Binarization            | 14 - Line (Horizontal)')
print('4 - Thresholding            | 15 - Line (Vertical)')
print('5 - Slicing                 | 16 - Line (+45)')
print('6 - Contrast Flare          | 17 - Line (-45)')
print('7 - Histogram               | 18 - Roberts')
print('8 - Histogram Equalization  | 19 - Sobel')
print('9 - Gaussian Noise          | 20 - Prewitt')
print('10 - Salt and Pepper        | 21 - Frei Chen')
print('11 - Low-pass (Mean)        | ')
a = int(input("Type a number: "))

if a == 1:
  inverted_img = negative(img)
  inverted_img.save('image/gatenho_invertido.jpg')
elif a == 2:
  mono_img = monochrome(img)
  mono_img.save('image/gatenho_mono.jpg')
elif a == 3:
  binary_img = binarization(img)
  binary_img.save('image/getenho_bi.jpg')
elif a == 4:
  mono_img = monochrome(img)
  thresholds = [85, 170]
  t_img = thresholding(mono_img, thresholds)
  t_img.save('image/gatenho_limiar.jpg')
elif a == 5:
  mono_img = monochrome(img)
  slice_img = slicing(mono_img)
  slice_img.save('image/gatenho_fatiado.jpg')
elif a == 6:
  idmenor= float(input('type id_small: '))
  idmaior = float(input('Digite id_bigger: '))
  mono_img = monochrome(img)
  contrast_img = contrast_flare(mono_img, idmenor, idmaior)
  contrast_img.save('image/gatenho_contrastado.jpg')
elif a == 7:
  mono_img = monochrome(img)
  histogram(mono_img)
elif a == 8:
  mono_img = monochrome(img)
  equalize_img = equalize(mono_img)
  equalize_img.save('image/gatenho_equalizado.jpg')
elif a == 9:
  mono_img = monochrome(img)
  gauss_img = gauss(mono_img)
  gauss_img.save('image/gatenho_gaussiano.jpg')
elif a == 10:
  mono_img = monochrome(img)
  sp_img = salt_and_pepper(mono_img)
  sp_img.save('image/gatenho_salpicado.jpg')
elif a == 11:
  r = 1
  media_img = mean(img, r)
  media_img.save('image/gatenho_media.jpg')
elif a == 12:
  mono_img = monochrome(img)
  r = 3
  median_img = median(mono_img, r)
  median_img.save("image/gatenho_mediana.jpg")
elif a == 13:
  r=1
  mono_img = monochrome(img)
  highpass_img = highpass(mono_img, r)
  highpass_img.save('image/gray_passalta.jpg')
elif a == 14:
  r=1
  mono_img = monochrome(img)
  horizontal_edge = line_hor(mono_img.convert('L'), r)
  horizontal_edge.save('image/gray_linha_hor.jpg')
elif a == 15:
  r=1
  mono_img = monochrome(img)
  vertical_edge = line_vert(mono_img.convert('L'), r)
  vertical_edge.save('image/gray_linha_vert.jpg')
elif a == 16:
  r=1
  mono_img = monochrome(img)
  diagonal_edge = line_45p(mono_img.convert('L'), r)
  diagonal_edge.save('image/gray_linha_45+.jpg')
elif a == 17:
  r=1
  mono_img = monochrome(img)
  diagonal_edge = line_45m(mono_img.convert('L'), r)
  diagonal_edge.save('image/gray_linha_45-.jpg')
elif a == 18:
  r=1
  mono_img = monochrome(img)
  roberts_img = roberts(mono_img.convert('L'), r)
  roberts_img.save('image/gray_roberts.jpg')
elif a == 19:
  r=1
  mono_img = monochrome(img)
  sobel_img = sobel(mono_img.convert('L'), r)
  sobel_img.save('image/gray_sobel.jpg')
elif a == 20:
  r=1
  mono_img = monochrome(img)
  prewitt_img = prewitt(mono_img.convert('L'), r)
  prewitt_img.save('image/gray_prewitt.jpg')
elif a == 21:
  r=1
  mono_img = monochrome(img)
  feichen_img = feichen(mono_img.convert('L'), r)
  feichen_img.save('image/gray_feichen.jpg')
