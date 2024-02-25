import gradio as gr 
import tensorflow as tf 
import numpy as np

model = tf.keras.models.load_model('handwritten1.model')

def recognize_character(image):
    img_resize = np.resize(image['composite'], (200, 200, 3))
    img = np.invert(np.array([img_resize]))
    prediction = model.predict(img)
    return np.argmax(prediction)
    

interface = gr.Interface(fn=recognize_character, 
                         inputs=gr.Sketchpad(height = 600, width=800, image_mode = 'RGB'),
                         outputs="text",
                         live=True)

interface.launch(share=True)

