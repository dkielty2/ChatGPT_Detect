from flask import Flask, render_template, request
import torch
import numpy as np
from application import  preprocess, process_for_model, metrics, w2v, burstiness, Classifier
 
model  = Classifier(use_LSTM=True,N_metrics=5)
model_weights_path = 'model_weights_all_epoch12_seed6287_batch10.pt'
model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')) ) 
model.eval()

app = Flask(__name__, template_folder='.')

@app.route('/')
def index():
    return render_template('app.html')

@app.route('/', methods=['POST'])
def process_form():
    if request.method == 'POST':
        
        input_text = request.form['input_data']
        tensor,metrics = process_for_model(input_text)
        metrics[0] = (np.log10(metrics[0])-2.3)/2.3
        predictions = model(tensor.reshape(-1,1,300), metrics.reshape(1,-1))
        pred_GPT = 100*round(predictions[0][0].item(),4)
        burst = round(burstiness(input_text),4)
        
        result1 = f'The probability the text was Chat GPT created is {pred_GPT} %.'
        result2 = f'Burstiness: {burst}.'
        
        return render_template('app.html', result1=result1, result2=result2)


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)

