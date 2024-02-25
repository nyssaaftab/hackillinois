import gradio as gr 
import tensorflow as tf 
import numpy as np

model = tf.keras.models.load_model('handwritten1.model')
class_indices = {'a': 0, 'ba': 1, 'be': 2, 'bu': 3, 'da': 4, 'de': 5, 'du': 6, 'e': 7, 'ga': 8, 'ge': 9, 'gu': 10, 'ha': 11, 'he': 12, 'hi': 13, 'hu': 14, 'i': 15, 'ka': 16, 'ke': 17, 'ku': 18, 'la': 19, 'le': 20, 'lu': 21, 'ma': 22, 'me': 23, 'mu': 24, 'na': 25, 'ne': 26, 'ni': 27, 'nu': 28, 'pa': 29, 'pe': 30, 'qa': 31, 'ra': 32, 're': 33, 'ru': 34, 'sa': 35, 'se': 36, 'si': 37, 'su': 38, 'ta': 39, 'te': 40, 'ti': 41, 'tu': 42, 'u': 43, 'wa': 44, 'wi': 45, 'ya': 46, 'za': 47, 'ze': 48, 'zi': 49, 'zu': 50}

def recognize_character(image):
    img = np.invert(np.array([image]))
    prediction = model.predict(img)
    predicted_label_numeric = np.argmax(prediction)

    # Map the numeric label to the corresponding character string
    predicted_label_string = [key for key, value in class_indices.items() if value == predicted_label_numeric][0]

    #Print the result
    #print(f"This character is probably a {predicted_label_string}")
    return (f"This character is probably a {predicted_label_string}")
    

interface = gr.Interface(fn=recognize_character, 
                         inputs=gr.Image(sources='upload'),
                         outputs="text",
                         live=True)

interface.launch(share=True)

