import tensorflow as tf
import timeit
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from functions.functions_image_training import *

# This function is run first to get the list that one can shuffle 
mainfolder = '/Users/carsonlam/vp/quality_confirmed/'
rejects = '/Users/carsonlam/vp/rejects/'
img_dict, cat_list = classifier_folder_to_jpg_dict_list(mainfolder,rejects)

start_time = timeit.default_timer()
#size = []

#tensor_image = tf.Variable(image, name='x')
tensor_image = tf.placeholder("uint8", [None, None, 3])
tensor_shape = tf.shape(tensor_image)

model = tf.initialize_all_variables()

with tf.Session() as session: 
    
    session.run(model)
    
    for i in range(len(cat_list)):
        #print(len(cat_list[i]))
        #print(img_dict[i+1])
        classfolder = mainfolder + img_dict[i+1] + '/'

        j=0
        for filename in cat_list[i]:

            cycle_image = mpimg.imread(classfolder + filename)
            Tensor_shape = session.run(tensor_shape, feed_dict={tensor_image: cycle_image})
            print(Tensor_shape)
            # Tensor_shape is a numpy ndarray of shape (3,) of the dimension sizes 

            j+=1
            if j>3:
                break 

    
#print(set(size))

end_time = timeit.default_timer()
print('DONE, ran for %.2fm' % ((end_time - start_time) / 60.))
