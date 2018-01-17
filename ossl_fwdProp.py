"""planar_data_classification.py - code from Week 3"""

# Package imports
import numpy as np
import modelMods
import modelDefinition
import utils

def forward_prop_one_layer(model, cache, layer):
    """
    :param model:
    :param cache:
    :param layer: target layer, activation going from layer-1 to layer
    :return:
    """

    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)
    Am, An, Zm, Zn = utils.get_cache_AmAmZmZn(cache, layer)

    if W.size == 0:                                 # Start building from zero! ToDo - generalize for n layers
        memory = True
        model = modelDefinition.add_unit(model, layer, memory, Am, d_index=0)
        W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)

    Zn = np.dot(W, Am) + b.T          # ??? not b.T  ????
    An = Zn                         # ???????? Review this. ToDo
    #An = utils.sigmoid(Zn)

    d = Am.shape[1]
    assert Zn.shape == (Ln, d)
    assert An.shape == (Ln, d)
    model = utils.set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var)
    cache = utils.set_cache_AmAnZmZn(cache, layer, Am, An, Zm, Zn)

    return model, cache, An


def forward_propagation(X, model, cache):
    """
    Argument:
    X -- input data of size (L0, m)
    model --  dictionary
    cache --  dictionary

    Returns:
    An -- The sigmoid output of the top layer
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """
    if not model:                       # Start building from zero!
        model, cache = modelDefinition.genesis(X)

    layer = 0
    model, cache, An = forward_prop_one_layer(model, cache, layer)        # ToDo - more than layer1

    return model, cache, An
