import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np
import tensorflow_probability as tfp

tfd = tfp.distributions
tfpl = tfp.layers
tfb = tfp.bijectors
tfk = tf.keras

def invsigmoid(x):
    xclip = tf.clip_by_value(x, 1e-6, 1.0-1e-6)
    #xclip = x
    return tf.math.log(xclip/(1.0-xclip))

# construct NAF model
def NAF2(inputdim, conddim, nafdim, depth=1, permute=True):

    xin = tfk.layers.Input(shape=(inputdim+conddim, ))

    xcondin = xin[:, inputdim:]

    xfeatures = xin[:, :inputdim]
    netout = None
    nextfeature = xfeatures
    for idepth in range(depth):
        #permutation = tf.random.shuffle(tf.range(inputdim))
        if permute:
            randperm = np.random.permutation(inputdim).astype('int32')
            permutation = tf.constant(randperm, name=f'permutation{idepth}')
            #permutation = tf.Variable(randperm, name=f'permutation{idepth}', trainable=False)
        else:
            permutation = tf.range(inputdim, dtype='int32',  name=f'permutation{idepth}')
        permuter = tfb.Permute(permutation=permutation, name=f'permute{idepth}')
        xfeatures_permuted = permuter.forward(nextfeature)
        outlist = []
        for iv in range(inputdim):
            xiv = tf.reshape(xfeatures_permuted[:, iv], [-1, 1])
            net = xiv
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w1 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            b1 = tfk.layers.Dense(nafdim, activation=None)(condnet)

            net1 = tf.nn.sigmoid(w1 * net + b1)
            condnet = xcondin
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            condnet = tfk.layers.Dense(128, activation=tf.nn.swish)(condnet)
            w2 = tfk.layers.Dense(nafdim, activation=tf.nn.softplus)(condnet)
            w2 = w2/ (1.0e-3 + tf.reduce_sum(w2, axis=1,keepdims=True)) # normalize

            net = invsigmoid(tf.reduce_sum(net1 * w2, axis=1, keepdims=True))
            outlist.append(net)
            xcondin = tf.concat([xcondin, xiv], axis=1)
        outputlayer_permuted = tf.concat(outlist, axis=1)
        outputlayer = permuter.inverse(outputlayer_permuted)
        nextfeature = outputlayer

    return tfk.Model(xin, outputlayer)
