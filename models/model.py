
# from .deep import LSTM
from .CRF import load_CRF, evaluate_CRF
from .BiLSTM import load_BiLSTM, evaluate_BiLSTM
from .BiLSTM_CRF import load_BiLSTM_CRF, evaluate_BiLSTM_CRF
import pickle
import tensorflow as tf
import numpy as np
import requests
import json

PATH_CRF_WEIGHT  = "/home/vanhocvp/Code/AI/NLP/NER/demo/models/weights/model1.crfsuite"
PATH_BILSTM_WEIGHT = "/home/vanhocvp/Code/AI/NLP/NER/demo/models/weights/ulstm_ner_10.pt"
PATH_BILSTM_CRF_WEIGHT = "/home/vanhocvp/Code/AI/NLP/NER/demo/models/weights/lstm_ner_6.pt"
PATH_MODEL_LSTM = "/home/vanhocvp/Code/AI/NLP/WebDemo/models/weights/LSTM.h5"

LIST_LABEL = ['Chính trị-Xã hội' ,'Đời sống', 'Khoa học', 'Kinh doanh', 'Pháp luật',
 'Sức khoẻ', 'Thế giới', 'Thể thao', 'Văn hoá', 'Vi tính']

class Models:
    def __init__(self) -> None:
        self.crf = load_CRF(PATH_CRF_WEIGHT)
        print ("[*]___LOADED CRF___")
        self.bilstm = load_BiLSTM(PATH_BILSTM_WEIGHT)
        print ("[*]___LOADED BiLSTM___")
        self.bilstm_crf = load_BiLSTM_CRF(PATH_BILSTM_CRF_WEIGHT)
        print ("[*]___LOADED BiLSTM + CRF___")
        self.dict_model = {
            "crf":{
                'model':self.crf,
                'type':'ML',
                'name':'CRF'

            },
            "bilstm":{
                'model':self.bilstm,
                'type':'DL',
                'name':'BiLSTM'
            },
            "bilstm_crf":{
                'model':self.bilstm_crf,
                'type':'DL',
                'name':'BiLSTM + CRF'
            }

        }
    def predict(self, model_key, text):
        if model_key == "phobert":
            url = 'http://103.74.122.136:7259/phobert'
            data = {'text': text}

            response = requests.post(url, json = data)
            print (response.text)
            result = json.loads(response.text)['result']
            return {'result': result}
        model = self.dict_model[model_key]['model']
        if model_key == "crf":
            result = evaluate_CRF(model, text)
        if model_key == "bilstm":
            result = evaluate_BiLSTM(model, text)
        if model_key == "bilstm_crf":
            result = evaluate_BiLSTM_CRF(model, text)
        
        return {"result":result}
            
if __name__ == "__main__":
    models = Models()
    result = models.predict(model_key="logistic_regression", text= "ngày xửa ngày xưa")
    print (result)    