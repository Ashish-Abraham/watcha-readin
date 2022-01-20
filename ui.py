import gradio as gr
from tensorflow.keras.models import load_model

def classify(text):
    model= load_model('my_model.h5')
    prediction = model.predict(text)
    if(prediction<=0.4):
        return "Bullshit"
    elif(prediction>0.4 and prediction<=0.6):
        return "Neutral"
    else :       
        return "Neish"

iface= gr.Interface(
    inputs=[gr.inputs.Textbox(lines=5, label="Context", placeholder="Type a sentence or paragraph here.")],
    outputs=[gr.outputs.Textbox(label="Prediction")],
    fn=classify, 
    title='watch-ya-watchin',
    theme='dark-peach'       
)

iface.launch()