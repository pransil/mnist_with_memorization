"""main.py"""

import numpy as np
import dataUtils
import utils
import modelDefinition
import fwdProp
import backProp
import modelMods

if __name__ == '__main__':
    d = 10               # Number of data samples
    L0 = 10             # Width of input vector - X[L0, m]
    margin = 0.35
    learning_rate = 0.0002
    epochs = 1000000

    # Load the data and define the network
    X = dataUtils.load_simple_data(d, L0)
    d = X.shape[1]
    cache = utils.cache_init(X)
    model = utils.model_init()

    for e in range(epochs):
        layer = 1
        model, cache, A = fwdProp.forward_prop_one_layer(model, cache, 1)
        error, target = backProp.compute_error(A, margin)
        model = modelMods.set_win_count(model, layer, target)

        # For every sample with no winner (target[...,d] == 0) create new unit
        Ln = A.shape[0]
        num_winners = np.sum(target, axis=0)       # Number of winner in each column
        for index in range(d):
            if num_winners[index] == 0:
                memory = True
                model = modelDefinition.add_unit(model, layer, memory, X, index)
                break                               # If unit added, need to re-run

        if Ln == model['L1']:                        # ToDo - generalize
            # Train only if no units were added since fwd_prop and compute_errro
            #model, cache, A = fwdProp.forward_prop_one_layer(model, cache, layer)
            #model, cache = modelMods.remove_duplicate_units(model, cache, layer, margin)
            #error, target = backProp.compute_error(A, margin)

            grads = backProp.back_prop_one_layer(model, cache, layer, error, margin)
            model = backProp.update_parameters(model, grads, d, learning_rate)

            model = modelDefinition.adjust_units(model, target, A)
            # Print the cost every 10 iterations
            if e % 10000 == 0 :
                print("Error after iteration", e)
                print (error)
                #print('A1', An)
                #print ('Target in main\n',target)
    blah = 0
