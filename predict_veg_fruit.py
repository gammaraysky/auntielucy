import tensorflow as tf
import numpy as np
from keras.preprocessing import image


def predict_image(userimage, model):
    category={
        0: 'Apple', 1: 'Avocado', 2: 'Banana', 3: 'Bean', 4: 'Bitter Gourd', 5: 'Blueberry', 
        6: 'Bottle Gourd', 7: 'Brinjal', 8: 'Broccoli', 9: 'Cabbage', 10: 'Capsicum', 11: 'Carrot', 
        12: 'Cauliflower', 13: 'Coconut', 14: 'Cucumber', 15: 'Dragonfruit', 16: 'Durian', 17: 'Grape', 
        18: 'Guava', 19: 'Kiwi', 20: 'Lemon', 21: 'Lime', 22: 'Longan', 23: 'Mango', 
        24: 'Muskmelon', 25: 'Orange', 26: 'Papaya', 27: 'Passion Fruit', 28: 'Pear', 29: 'Pineapple', 
        30: 'Plumcot', 31: 'Pomegranate', 32: 'Pomelo', 33: 'Potato', 34: 'Pumpkin', 35: 'Radish', 
        36: 'Strawberry', 37: 'Tomato', 38: 'Watermelon',
    }


    thumbnail={
        0:  '/images/Apple.jpg',
        1:  '/images/Avocado.jpg',
        2:  '/images/Banana.jpg',
        3:  '/images/Bean.jpg',
        4:  '/images/Bittergourd.jpg',
        5:  '/images/Blueberry.jpg',
        6:  '/images/Bottlegourd.jpg',
        7:  '/images/Brinjal.jpg',
        8:  '/images/Broccoli.jpg',
        9:  '/images/Cabbage.jpg',
        10: '/images/Capsicum.jpg',
        11: '/images/Carrot.jpg',
        12: '/images/Cauliflower.jpg',
        13: '/images/Coconut.jpg',
        14: '/images/Cucumber.jpg',
        15: '/images/Dragonfruit.jpg',
        16: '/images/Durian.jpg',
        17: '/images/Grape.jpg',
        18: '/images/Guava.jpg',
        19: '/images/Kiwi.jpg',
        20: '/images/Lemon.jpg',
        21: '/images/Lime.jpg',
        22: '/images/Longan.jpg',
        23: '/images/Mango.jpg',
        24: '/images/Muskmelon.jpg',
        25: '/images/Orange.jpg',
        26: '/images/Papaya.jpg',
        27: '/images/Passionfruit.jpg',
        28: '/images/Pear.jpg',
        29: '/images/Pineapple.jpg',
        30: '/images/Plumcot.jpg',
        31: '/images/Pomegranate.jpg',
        32: '/images/Pomelo.jpg',
        33: '/images/Potato.jpg',
        34: '/images/Pumpkin.jpg',
        35: '/images/Radish.jpg',
        36: '/images/Strawberry.jpg',
        37: '/images/Tomato.jpg',
        38: '/images/Watermelon.jpg',
    }
    writeup={
        0:  'Apples isafruityoucaneat', 
        1:  'Avocado isafruityoucaneat', 
        2:  'Banana isafruityoucaneat', 
        3:  'Bean isafruityoucaneat', 
        4:  'Bitter Gourd isafruityoucaneat', 
        5:  'Blueberry isafruityoucaneat', 
        6:  'Bottle Gourd isafruityoucaneat', 
        7:  'Brinjal isafruityoucaneat', 
        8:  'Broccoli isafruityoucaneat', 
        9:  'Cabbage isafruityoucaneat', 
        10: 'Capsicum isafruityoucaneat', 
        11: 'Carrot isafruityoucaneat', 
        12: 'Cauliflower isafruityoucaneat', 
        13: 'Coconut isafruityoucaneat', 
        14: 'Cucumber isafruityoucaneat', 
        15: 'Dragonfruit isafruityoucaneat', 
        16: 'Durian isafruityoucaneat', 
        17: 'Grape isafruityoucaneat', 
        18: 'Guava isafruityoucaneat', 
        19: 'Kiwi isafruityoucaneat', 
        20: 'Lemon isafruityoucaneat', 
        21: 'Lime isafruityoucaneat', 
        22: 'Longan isafruityoucaneat', 
        23: 'Mango isafruityoucaneat', 
        24: 'Muskmelon isafruityoucaneat', 
        25: 'Orange isafruityoucaneat', 
        26: 'Papaya isafruityoucaneat', 
        27: 'Passion Fruit isafruityoucaneat', 
        28: 'Pear isafruityoucaneat', 
        29: 'Pineapple isafruityoucaneat', 
        30: 'Plumcot isafruityoucaneat', 
        31: 'Pomegranate isafruityoucaneat', 
        32: 'Pomelo isafruityoucaneat', 
        33: 'Potato isafruityoucaneat', 
        34: 'Pumpkin isafruityoucaneat', 
        35: 'Radish isafruityoucaneat', 
        36: 'Strawberry isafruityoucaneat', 
        37: 'Tomato isafruityoucaneat', 
        38: 'Watermelon isafruityoucaneat', 
    }
    
    # img_ = image.load_img(userimage, target_size=(150, 150))
    img_array = image.img_to_array(userimage)
    img_processed = np.expand_dims(img_array, axis=0) 
    img_processed /= 255.   
    
    prediction = model.predict(img_processed)
    index = np.argmax(prediction)
    value = prediction[0][index]

    return category[index], value, writeup[index], thumbnail[index]
