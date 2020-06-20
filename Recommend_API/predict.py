import pickle
import pandas as pd
from heapq import nsmallest
from gensim.models import Word2Vec
from flask import Flask, request
from flask_restful import Resource, Api, reqparse


app = Flask(__name__)
api = Api(app)

class RegistUser(Resource):        
    def post(self):

        data = request.get_json()
        data = data['clicked_items']

        word2vec_model = Word2Vec.load('word2vec.model')

        word_vectors = word2vec_model.wv

        data = pd.DataFrame(data).transpose().astype(float)
        data = data.applymap(str)

        data = [word_vectors[data.loc[0,i]] for i in range(13)]
        data = pd.DataFrame(data)
        data = data.transpose()
        data = data.apply(pd.to_numeric)

        vocabs = word_vectors.vocab.keys()

        word_vectors_list = [word_vectors[v] for v in vocabs] 
        f = open('pred_model_64bit.pkl', 'rb')
        model = pickle.load(f)
        pred = model.predict(data)
        f.close()

        del model

        answer = list(map(lambda x: x[0], nsmallest(5, word_vectors_list, key=lambda x: abs(x-pred[0]))))
        answer = [int(float(word_vectors.index2word[word_vectors_list.index(temp)])) for temp in answer]

        return answer
        
api.add_resource(RegistUser, '/predict')

if __name__ == '__main__':
    app.run(host='0.0.0.0')