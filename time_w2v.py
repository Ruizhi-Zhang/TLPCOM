import gensim
import numpy as np
import pickle
import time

def learn_node_representation(time_walks, size=128):
    documents = []
    for walk in time_walks:
        documents.append([str(w) for w in walk])

    # w2v model
    model = gensim.models.Word2Vec(documents, size=size, window=10, min_count=1, workers=4, negative=10)
    model.train(documents, total_examples=len(documents), epochs=200)

    node_representation = {}
    for key, val in model.wv.vocab.items():
        node_representation[int(key)] = model.wv[key]
    return node_representation


def main():

    path = 'D:/.../.../.../contact.time.walks'
    w2v_embedding_path = 'D:/.../.../.../contact.w2v.pkl'

    with open(path, 'rb') as f:
        time_walks = pickle.load(f)
        print(time_walks[0:20])

    node_representation = learn_node_representation(time_walks)

    with open(w2v_embedding_path, 'wb') as f:
        pickle.dump(node_representation, f)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('总时间：', end - start)
