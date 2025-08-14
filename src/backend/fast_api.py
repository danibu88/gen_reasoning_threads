from typing import Union
from fastapi import FastAPI, Request
from pydantic import BaseModel
import nltk
import pandas as pd
import pickle
import uvicorn
import numpy as np

app = FastAPI()

w2v_model = "gensim_glove.pkl"

with open(w2v_model, 'rb') as file:
    gensimModel = pickle.load(file)

@api.get('/')
def root():
    return {'message': 'hello'}


@api.post('/predict')

async def basic_predict(request: Request):

    candidates = []

    input_data = await request.json()

    candidates.append(gensimModel.wv.most_similar(positive=[input_data]))

    return candidates



        



