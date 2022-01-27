import gradio as gr
from tensorflow.keras.models import load_model
import prepro
import numpy as np
import nltk


def classify(text):
    nltk.download('stopwords')
    model= load_model('nlp1.h5')
    X= prepro.preprocess(text)
    prediction = model.predict(np.array(X))
    # return prediction
    if(prediction<=0.4):
        return "Sounds Positive. Giving a good impression to start reading this stuff. "    
    elif(prediction>0.4 and prediction<=0.6):
        return "Sounds Neutral. Speaks generally and not biased towards any value."
    else :       
        return "Looks like you are reading negative content. Some words sound negative in context."

iface= gr.Interface(
    inputs=[gr.inputs.Textbox(lines=5, label="Context", placeholder="Type a sentence or paragraph here.")],
    outputs=[gr.outputs.Textbox(label="Prediction")],
    fn=classify, 
    title='WATCHA-READIN',
    theme='dark-peach'       
)

iface.launch(share=True)