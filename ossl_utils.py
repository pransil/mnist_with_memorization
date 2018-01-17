"""utils.pyc"""
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_model_names(layer):
    # Return strings used to extract from model
    assert layer >= 1
    Ln_name = 'L' + str(layer)  # This layer
    Lm_name = 'L' + str(layer - 1)  # Previous layer
    W_name = 'W' + str(layer)
    b_name = 'b' + str(layer)
    G_name = 'G' + str(layer)
    Wc_name = 'Wc' + str(layer)
    Mean_name = 'Mean' + str(layer)
    Var_name = 'Var' + str(layer)
    return W_name, b_name, Lm_name, Ln_name, G_name, Wc_name, Mean_name, Var_name


def get_model_WbLGWcMV(model, layer):
    W_name, b_name, Lm_name, Ln_name, G_name, Wc_name, Mean_name, Var_name = make_model_names(layer)
    Ln = model[Ln_name]                                 # # of units in this layer
    Lm = model[Lm_name]                                 # # of units in previous layer
    W = model[W_name]
    b = model[b_name]
    G = model[G_name]
    Wc = model[Wc_name]
    Mean = model[Mean_name]
    Var = model[Var_name]
    return W, b, Lm, Ln, G, Wc, Mean, Var


def set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var):
    assert layer >= 1
    W_name, b_name, Lm_name, Ln_name, G_name, Wc_name, Mean_name, Var_name = make_model_names(layer)
    model[W_name] = W
    model[b_name] = b
    model[Lm_name] = Lm                                # # of units in previous layer
    model[Ln_name] = Ln                                # # of units in this layer
    model[G_name] = G
    model[Wc_name] = Wc
    model[Mean_name] = Mean
    model[Var_name] = Var
    return model


def make_cache_names(layer):
    # Return strings used to extract from model
    assert layer >= 1
    An_name = 'A' + str(layer)  # This layer
    Am_name = 'A' + str(layer - 1)  # Previous layer
    Zn_name = 'Z' + str(layer)
    Zm_name = 'Z' + str(layer)
    return Am_name, An_name, Zm_name, Zn_name


def get_cache_AmAmZmZn(cache, layer):
    Am_name, An_name, Zm_name, Zn_name = make_cache_names(layer)
    Am = cache[Am_name]                                 # Activation from previous layer
    An = cache[An_name]                                 # Activation in this layer
    Zm = cache[Zm_name]
    Zn = cache[Zn_name]
    return Am, An, Zm, Zn


def set_cache_AmAnZmZn(cache, layer, Am, An, Zm, Zn):
    Am_name, An_name, Zm_name, Zn_name = make_cache_names(layer)
    cache[Am_name] = Am                                 # # of units in this layer
    cache[An_name] = An                                 # # of units in previous layer
    cache[Zm_name] = Zm
    cache[Zn_name] = Zn
    return cache


def model_init():
    W0 = np.array([])
    b0 = np.array([])
    W1 = np.array([])
    b1 = np.array([])
    L0 = 0
    L1 = 0
    G0 = 0
    Wc0 = 0
    Mean0 = 0
    Var0 = 0
    G1 = np.array([])
    Wc1 = np.array([])
    Mean1 = np.array([])
    Var1 = np.array([])
    model = {'W0': W0, 'W1': W1, 'b0': b0, 'b1': b1,
             'L0': L0, 'L1': L1, 'G0': G0, 'G1': G1,
             'Wc0': Wc0, 'Wc1': Wc1,
             'Mean0': Mean0, 'Mean1': Mean1,
             'Var0': Var0, 'Var1': Var1}
    return model


def cache_init(X):
    A0 = X
    A1 = np.array([])
    Z0 = 0
    Z1 = np.array([])
    cache = {'A0': A0, 'A1': A1, 'Z0': Z0, 'Z1': Z1}
    return cache

