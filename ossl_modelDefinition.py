"""modelDefinition.py"""

# Package imports
import numpy as np
from math import sqrt
import utils
import modelMods

def create_non_memory_unit(L):
    """
    L       - # of units in previous layer (number of weights coming into this unit)
    """
    W = np.random.randn(1, L) * np.sqrt(2.0/L)
    W = W.reshape((1, L))
    b = np.random.randn(1) * sqrt(2 / L)         # Adding only one bias weight
    b = b.reshape((1,1))
    return W, b

def adjust_units(model, target, A):
    layer = 1
    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)
    # Kill units with too few wins
    wins_per_unit = np.sum(target, axis=1)
    min_win_rate = 0.3
    for u in range(Ln):
        win_rate = float(wins_per_unit[u]) / target.shape[1]
        if win_rate <= min_win_rate:
            model = modelMods.kill_unit(model, layer, u)
        mean, var = find_cluster_stats(target, A, index=u)
        if Mean.size == 0:
            Mean = mean
            Mean = Mean.reshape((1, mean.size))
            Var = var
            Var = Var.reshape((1,var.size))
        else:
            Mean = np.vstack((Mean, mean))
            Var = np.vstack((Var, var))

        model = utils.set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var)
    find_cluster_drift(W, b, G, Mean)
    return model


def find_cluster_drift(W, b, G, Mean):
    Zn = np.dot(W, G.T) # + b.T          # ??? not b.T  ????
    An = Zn                         # ???????? Review this. ToDo
    drift = An - Mean
    variance = np.square(drift)

    return drift, variance


def find_cluster_stats(winners, A, index):
    r, c = winners.shape
    wi = winners[index, ...]
    wi = wi.reshape(1, c)
    winner_count = np.sum(wi)
    Ai_winners = A * wi
    Ai_winner_sum = np.sum(Ai_winners, axis = 1)
    Ai = A[...,index]
    Ai = Ai.reshape((1, Ai.size))
    Ai_length = np.sqrt(np.sum(np.square(Ai_winner_sum)))
    Ai_mean = Ai_winner_sum / Ai_length
    Ai_mean_reshaped = Ai_mean.reshape((Ai.size, 1))
    Ai_diff = (Ai_winners - Ai_mean_reshaped) * wi
    Ai_sqr = np.square(Ai_diff)
    Ai_var = np.sum(Ai_sqr, axis = 1)
    Ai_var_reshaped = Ai_var.reshape((1, Ai.size))
    Ai_mean_reshaped = Ai_mean.reshape((1, Ai.size))

    return Ai_mean_reshaped, Ai_var_reshaped


def memorize_input(A, G, index=0):
    # w[ln,lm] = a[ln,lm](len(A[...,lm)); where w is ln cols because we have memorized ln vectors
    Ai = A[...,index]
    Ai = Ai.reshape((1, Ai.size))
    Ai_length = np.sum(np.square(Ai))
    W = Ai / Ai_length                                  # Set weights so that dot(W,A) = 1
    W = W.reshape((1,Ai.size))
    b = np.zeros((1,1))
    if G.size == 0:
        G = Ai
        G = G.reshape((1,Ai.size))
    else:
        G = np.vstack((G, A[...,index]))                    # Record the 'genesis memory' input vector
    return W, b, G                                        # ToDo - fix G - just appending current element, need new element

food_init = 10

def genesis(model, X):
    """
    Create the very first node
    model   - model parameters
    X       - input data
    Return: - model (updated)
    """
    np.random.seed(2)                                # So we can get consistent results when debugging

    W1, b1, L0, L1, G1, Wc1, Mean, Var = utils.get_model_WbLGWcMV(model, 1)
    L0 = X.shape[0]
    W1, b1 = create_non_memory_unit(L0)
    # Change bias to 'memorize'
    W1, b1, G1 = memorize_input(X, G1)
    L1 += 1
    layer = 1
    model = utils.set_model_WbLGWcMV(model, layer, W1, b1, L0, L1, G1, Wc1, Mean, Var)
    return model


def add_unit(model, layer, memory=False, Am=False, d_index=False):
    """
    Arguments:
        model   - Model dict
        layer   - Layer new unit goes onto
        memory  - Add memory unit if True
        Am      - np.array(Ln,1) - Activation from previous level; A=X when building in L1
                  the pattern being 'memorized' is A[...,m_index], all rows, one col from m
        d_index - The column from A to use (incoming data/activation)
    Returns:
        Updated model
    """
    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)

    if W.size == 0:                                 # Start building from zero!
        model = genesis(model, Am)
        Ln = 1
    else:
        assert layer >= 1                               # Don't create units on input layer, L0
        np.random.seed(2)                               # So we can get consistent results when debugging
        W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)

        if memory:
            W_new, b_new, G = memorize_input(Am, G, d_index)
        else:
            W_new, b_new = create_non_memory_unit(Lm)   # ToDo - does this need G init =0???

        Ln += 1
        W = np.vstack((W, W_new))                       # Stack the new row onto Wn
        b = np.append(b, b_new)
        b = b.reshape((1,Ln))
        model = utils.set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var)

    #model = modelMods.adjust_food_when_adding_unit(model, Ln)
    return model

''' deprecated??????
def add_memory_unit(model, n, A, d_index):
    """
    Arguments:
        model   dictionary of model parameters: W, b, Lm, Ln, G
                - L     - np.array(L) of layer structures {'L0': #units in layer0 (input), 'L1': ...
                - G     - np.array(L?) of 'genisis memories', one for each unit built as a 'one-shot'; Only layer1 and above
                - Wn    - np.array(Ln, Ln-1) weight matrix; for W0, L(n-1) is input
                - bn    - np.array(Ln, 1) bias vector
        n       Layer unit will be added to
        A       - np.array(Ln,1) - Activation from previous level
                  the pattern being 'memorized' is A[...,m_index], all rows, one col from m
        d_index - The column from A to use
    Returns:
        Updated model
    """
    memory = True
    model = add_unit(model, n, memory, A, d_index)

    return model

'''