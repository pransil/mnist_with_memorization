"""modelMods.py - Adding new units to top layer

"""
import numpy as np
from scipy import spatial
import utils


def kill_unit(model, layer, u):
    W, b, Lm, Ln, G, Wc, Mean, Var = utils.get_model_WbLGWcMV(model, layer)
    W = np.delete(W, u, axis=0)
    b = np.delete(b, u)
    Ln -= 1
    b = b.reshape(1,Ln)
    G = np.delete(G, u, axis=0)
    Wc = np.delete(Wc, u, axis=0)

    model = utils.set_model_WbLGWcMV(model, layer, W, b, Lm, Ln, G, Wc, Mean, Var)
    return model


def find_winners(h, margin):
    h_len = F.norm(h, axis=1)
    winners = h_len > (1 - margin)
    return winners





def set_win_count(model, layer, wins):
    Wc = np.sum(wins, axis=1)
    model['Wc'+str(layer)] = Wc
    return model


def find_dups(Am, An, margin):
    dups = []
    rows, cols = An.shape
    m_dist = spatial.distance.pdist(Am.T)
    n_dist = spatial.distance.pdist(An.T)
    for r in range(rows-1):
        for c in range(r, rows):
            if dist[0] < margin:
                dups.append((r,c+1))
            dist = dist[1:]

    return dups


def remove_one_dup(model, cache, index, layer):
    Wn_name = 'W' + str(layer)
    bn_name = 'b' + str(layer)
    model[Wn_name] = np.delete(model[Wn_name], index, axis=0)
    model[bn_name] = np.delete(model[bn_name], index, axis=0)

    mem = 'G' + str(layer)
    del model[mem[index]]

    Zn_name = 'Z' + str(layer)
    An_name = 'a' + str(layer)
    cache[Zn_name] = np.delete(cache[Zn_name], index, axis=1)
    cache[An_name] = np.delete(cache[An_name], index, axis=1)

    return model, cache


def remove_duplicate_units(model, cache, layer, margin):
    """
    layer - Layer number
    """

    An_name = 'A' + str(layer)
    An = cache[An_name]
    Am_name = 'A' + str(layer-1)
    Am = cache[Am_name]
    dups = find_dups(Am, An, margin)
    if len(dups) < 0:
        model, cache = remove_one_dup(model, cache, index, layer)
        model, cache = remove_duplicate_units(model, cache, layer, margin)

    return model, cache
