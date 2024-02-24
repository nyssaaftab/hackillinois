import gradio as gr 
import tensorflow as tf 
import numpy as np

model = tf.keras.models.load_model('handwritten.model')

def recognize_character(image):
    img = np.invert(np.array([image]))
    prediction = model.predict(img)
    return np.argmax(prediction)
    

interface = gr.Interface(fn=recognize_character, 
                         inputs=gr.Sketchpad(height = 200, width=200),
                         outputs="label",
                         live=True)

interface.launch()

