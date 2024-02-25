import gradio as gr 
import tensorflow as tf 
import numpy as np
import cv2

model = tf.keras.models.load_model('handwritten.model')

def recognize_character(image):
    ## methods used in jupiter notebook

    #scale image - draw in corner, take first 200,200
    # img_resize = np.resize(image['composite'], (200,200,3))
    img = np.invert(np.array([image]))
    prediction = model.predict(img)
    return prediction #np.argmax(prediction)
    

interface = gr.Interface(fn=recognize_character, 
                         inputs=gr.Image(sources='upload'),
                         outputs="text",
                         live=True)

interface.launch()