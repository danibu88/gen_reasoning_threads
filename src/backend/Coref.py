from typing import Union

from fastapi import FastAPI, Request

from pydantic import BaseModel

import nltk

import pandas as pd

import pickle

import uvicorn 

import neuralcoref



app = FastAPI()

with open(model_, 'rb') as file:
    gensimModel = pickle.load(file)

@api.get('/')
def root():
    return {'message': 'hello'}



@api.post('/predict')

async def create_clusters(request: Request):

    clusters = list()
    for i in sent_tokens:
        doc = nlp(i)
        clusters.append(doc._.coref_clusters)
        return clusters





        




# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}

# def similarity(Keywords):
#     values  = list()
#     for i in Keywords:
#         values.append(print_closest_words(glove([i])))
#     return values
        


# @app.get("/")
# def read_root():
#     return {"Keyword": "XYZ"}


# @app.get("/items/{Convolutional_NN}")
# def read_item(Convolutional_NN: str, q: Union[str, None] = None):
#     return {"Keywords": 'Convolutional_NN'}
