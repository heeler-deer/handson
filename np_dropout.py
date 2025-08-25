import numpy as np

def train(rate, x,w1, b1, w2,b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    #伯努利分布
    mask1 = np.random.binomial(1, 1-rate, layer1.shape)
    layer1 = layer1*mask1
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    mask2 = np.random.binomial(1, 1-rate, layer2.shape)
    layer2 = layer2*mask2
    return layer2
def test(rate,x, w1,b1,w2,b2):
    layer1 = np.maximum(0, np.dot(w1, x) + b1)
    layer1 = layer1*(1-rate)
    layer2 = np.maximum(0, np.dot(w2, layer1) + b2)
    layer2 = layer2*(1-rate)
    return layer2