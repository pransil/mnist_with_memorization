"""planar_data_classification.py - code from Week 3"""

# Package imports
import numpy as np
import utils


def compute_error(A, margin):

    """Arguments:
    A       -- The tanh output of the activation, of shape (categories, number of examples)
    margin  -- How close the output needs to be to 1 to be considered  the 'winner' or 'confident'
    Returns:
    error   -- How far 'winner' is from 1, and how far 'losers' are from 0
    target  -- [1, 0, 1, ...]
    """

    (Ln, m) = A.shape
    # error pushes losers away from zero and winner toward zero
    target = A > (1 - margin)
    error = target - A                              # find distance from 1 (winners), 0 (losers)
    return error, target


def back_prop_one_layer(model, cache, layer, error, margin):
    """
    Arguments:
    model   -- python dictionary containing weights, biases, whatever else we need...
    cache   -- a dictionary containing "Z1", "A1", "Z2", "A2"
    layer    -- from this layer (Ln) to prev (Lm)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)
    Am, An, Zm, Zn = utils.get_cache_AmAmZmZn(cache, layer)
    d = float(Am.shape[1])                  # # of data samples

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZn = - error
    target = An > (1 - margin)
    dZn_boosted = boost_dZ(dZn, target)

    #dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))        SAVE THIS --- WILL NEED LATER
    dW_boosted = (1 / d) * np.dot(dZn_boosted, Am.T)
    db = (1 / d) * np.sum(dZn, axis=1, keepdims=True)
    db = db.T

    assert dW_boosted.shape == W.shape
    assert db.shape == b.shape
    assert dZn_boosted.shape == Zn.shape

    grads = {"dW1": dW_boosted,
             "db1": db }

    return grads

# For each winner, boost the contribution to dW. Hack needed because learning is moving
# units too far from their initial memory
def boost_dZ(dZn, target):
    lr_boost = -0.0                                 # ToDo - should pass in. Hack!!!
    dZ_winners = dZn * target                       # 1 except in winner locations that have been 'knocked off'
    dZ_winners_boosted = dZ_winners * lr_boost
    dZ_boosted = dZn + dZ_winners_boosted

    return dZ_boosted

# Is this deprecated????????????????????
'''
def back_prop(model, cache, error):
    """
    Arguments:
    model   -- python dictionary containing weights, biases, whatever else we need...
    cache   -- a dictionary containing "Z1", "A1", "Z2", "A2"
   layer    -- from this layer (Ln) to prev (Lm)

    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    """
    d = X.shape[1]

    W1 = model["W1"]
    W2 = model["W2"]
    A1 = cache['A1']
    Z2 = cache['Z2']

    # Backward propagation: calculate dW1, db1, dW2, db2.
    dZ2 = Z2 - error

    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    assert dW1.shape == W1.shape
    assert dW2.shape == W2.shape
    assert db1.shape == model['b1'].shape
    assert db2.shape == model['b2'].shape
    assert dZ1.shape == cache['Z1'].shape
    assert dZ2.shape == Z2.shape

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads
'''

def update_parameters(model, grads, d, learning_rate=0.02):
    """
    Arguments:
    model -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients
    d     --    Number of data samples

    Returns:
    model -- wth updated W and b
    """

    layer = 1                   # ToDo - generalize
    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)

    dW = grads['dW1']          # Todo - generalize
    db = grads['db1']

    # Update rule for each parameter
    W = W - learning_rate * dW
    b = b - (learning_rate * db)/(d*100)        # ToDo

    model = utils.set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var)
    return model


