# takes file.png which is a [height x width x 4 channel] Tensor and 
# converts it to a  [height x width x 4 channel] Tensor to save as a jpeg file, file.jpg

from PIL import Image
im = Image.open("file.png")
bg = Image.new("RGB", im.size, (255,255,255))
bg.paste(im, (0,0), im)
bg.save("file.jpg", quality=95)
