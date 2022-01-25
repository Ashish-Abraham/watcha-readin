import gradio as gr
#from tensorflow.keras.models import load_model

def classify(text):
    model= load_model('my_model.h5')
    prediction = model.predict(text)
    if(prediction<=0.4):
        return "Looks like you are reading negative content. Some words sound negative in context."
    elif(prediction>0.4 and prediction<=0.6):
        return "Sounds Neutral. Speaks generally and not biased towards any value."
    else :       
        return "Sounds Positive. Giving a good impression to start reading this stuff. "

iface= gr.Interface(
    inputs=[gr.inputs.Textbox(lines=5, label="Context", placeholder="Type a sentence or paragraph here.")],
    outputs=[gr.outputs.Textbox(label="Prediction")],
    fn=classify, 
    title='WATCHA-READIN',
    theme='dark-peach'       
)

iface.launch(share=True)