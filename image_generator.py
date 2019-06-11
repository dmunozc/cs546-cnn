from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from os import listdir
from os.path import isfile, join

datagen = ImageDataGenerator(
        rotation_range=50,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest')
files =[f for f in listdir("images/person") if isfile(join("images/person", f))]
for file in files:
  img = load_img('images/person/' + file) 
  x = img_to_array(img) 
  x = x.reshape((1,) + x.shape)  
  i = 0
  for batch in datagen.flow(x, batch_size=1,
                            save_to_dir='images/person',
                            save_prefix='pers1', save_format='jpg'):
      i += 1
      if i > 10:
          break  
