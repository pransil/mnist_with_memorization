"""model.py"""

# Package imports
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import utils


# Network definition
class MLP(chainer.Chain):
    def layer_setup(self, layer):
        """ init a layer and add unit_count, unit_win_counts, unit_centroids"""
        if layer == 1:
            self.l1 = L.Linear(None, 1)  # start with 1 unit per layer
            l = self.l1
            l.margin = 0.2                  # ToDo - put in arg parse params???
        elif layer == 2:
            self.l2 = L.Linear(None, 1)
            l = self.l2
            l.margin = 0.05                 # ToDo - put in arg parse params???

        l.unit_count = chainer.Variable(None)
        l.unit_count.array = np.array(0)

        # Count every time each unit has a win
        l.unit_win_counts = chainer.Variable(None)
        l.unit_win_counts.array = np.reshape(np.array(0), (1,))

        # Record input index (x[i] not h[i]) for each unit win
        #l.unit_win_lists = chainer.Variable(None)
        l.unit_win_lists = [-1]

        # Each win modifies centroid by win_vec/unit_win_counts
        l.unit_centroids = chainer.Variable(None)
        l.unit_centroids.array = np.reshape(np.array(0.0), (1,1))

        # Keep variance for each unit
        l.unit_vars = chainer.Variable(None)
        l.unit_vars.array = np.reshape(np.array(0.0), (1,1))


    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            self.layer_setup(layer=1)
            self.layer_setup(layer=2)
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


    def update_centroids_and_vars(self, l, hm, hn):
        """
        Update the centroids of any units that won for a single input hm
        hm      - m-dim activation vector from lower level; use to modify centroids of winners
        hn      - n-dim activation vector used to make a mask, 1 for ln winners, 0 for losers [0, 0 ,1, 0, 1, ...]
        Create d_centroid[n,m], 0 in loser rows, am in winner rows (scaled by 1/win_count)
        and add this to l.unit_centroids
        """
        m = hm.shape[0]
        n = hn.shape[0]
        winners = hn > (1 - l.margin)
        mask = np.ones((n,m)) * winners[..., None]
        #mask = np.ones((n,m)) * winners.reshape((n,1))
        hm_masked = mask * hm
        win_counts = l.unit_win_counts[..., None]
        #win_counts = l.unit_win_counts.reshape((n,1))
        d_centroids = hm_masked / (win_counts.array + 0.5)
        l.unit_centroids.array += d_centroids
        d_vars = hm_masked / (win_counts.array + 0.5)
        l.unit_vars.array += d_vars


    def update_win_counts_and_lists(self, l, xi, hn):
        """
        Update the l.unit_win_counts of any units that won for a single input
        l       - layer to update
        xi      - index of input that created the wins
        hn      - n-dim activation vector used to make a mask, 1 for ln winners, 0 for losers [0, 0 ,1, 0, 1, ...]
        """
        winners = hn > (1 - l.margin)
        l.unit_win_counts.array += winners

        winners_list = np.where(winners)

        for i in winners_list:
            ii = i[0]
            #wi = winners_list[i]
            l.unit_win_lists[ii] += [xi]


    def hist_unit_win_lists(self, layer):
        l = self.get_layer(layer)
        unit_count = l.unit_count
        unit_win_lists = l.unit_win_lists
        assert unit_count.data == len(unit_win_lists)
        hist = []
        for i in range(unit_count.data):
            hist += [len(unit_win_lists[i])]
        return hist


    def count_winners(self, layer, hi):
        """ Count winners and """
        l = self.get_layer(layer)
        if l.unit_count.array == 0:
            win_count = False
        else:
            winners = hi > (1 - l.margin)
            win_count = np.sum(winners)
            if win_count == 0:
                win_count = False
        return win_count


    def add_weights(self, centroid, hm_new_components):
        new_centroid = np.hstack((centroid, hm_new_components))
        new_cent_len_sq = np.sum(np.square(new_centroid))
        W_new = new_centroid / new_cent_len_sq
        return W_new


    def add_connections(self, hm, layer):
        """
        hm      - activation from below, including new units
        """
        lm = self.get_layer(layer-1)
        ln = self.get_layer(layer)
        old_count = ln.W.shape[1]
        new_count = lm.W.shape[0]
        hm_new_components = hm.array[..., old_count:new_count]
        for i in range(old_count, new_count):
            self.add_weights(ln.unit_centroids, hm_new_components)
            l.W.array = np.hstack((l.W.array, W_new))       # ToDo - off by bias: 1=Wx +b


    def create_memory_unit(self, layer, h, i):
        """
        Adds a unit, 'memorizing' the input vector h[i,...],
            creating W, b
            adding one more 'win' to mem_count
            updates centroid, shifting some due to new 'win'
        layer - the layer the new unit will be added to
        h - output of l-1, input to l, the vector to memorize; h[batch_size]
        i - index into h for the sample to be memorized;

        """

        l = self.get_layer(layer)
        mem_vec = h[i, ...]
        mem_vec = mem_vec.reshape((1, mem_vec.shape[0]))
        mem_vec_squared = np.square(mem_vec)
        mem_vec_squared = mem_vec_squared.reshape((1, mem_vec_squared.shape[1]))
        length = np.sum(mem_vec_squared)
        W_new = mem_vec / length
        width = h.shape[1]
        #W_new = W_new[None, ...]
        W_new = W_new.reshape((1, width))
        if l.unit_count.array == 0:
            l.W.array[0] = W_new          # Already have l.b
            l.unit_win_counts.array += 1
            l.unit_centroids.array = mem_vec
            l.unit_vars.array = np.zeros(mem_vec.shape, dtype=np.float32)
            l.unit_win_lists = [[i]]
            assert i == 0
        else:
            l.W.array = np.vstack((l.W.array, W_new))
            l.b.array = np.hstack((l.b.array, np.array(0, dtype=np.float32)))
            l.unit_win_counts.array = np.hstack((l.unit_win_counts.array, np.array(1, dtype=np.float32)))
            l.unit_centroids.array = np.vstack((l.unit_centroids.array, mem_vec))
            l.unit_vars.array = np.vstack((l.unit_vars.array, mem_vec_squared))
            l.unit_win_lists += [[i]]

        l.unit_count.array += 1
        return

    def get_layer(self, number):
        ln = 'l' + str(number)
        l = self[ln]
        return l


def train_layer1(model, x):
    samples, width = x.shape
    l = model.predictor.l1
    for s in range(samples):
        h = F.relu(l(x))
        if l.unit_count.array == 0:
            print("Make first unit for layer1")
            MLP.create_memory_unit(model, layer=1, h=x, i=s)
        else:
            winners = MLP.count_winners(model, layer=1, hi=h.array[s])
            if winners == False:
                print("No L1 Winners for x[" + str(s) + "]")
                MLP.create_memory_unit(model, layer=1, h=x, i=s)
            else:
                MLP.update_centroids_and_vars(model, l, x[s], h[s].array)
                MLP.update_win_counts_and_lists(model, l, xi=s, hn=h[s].array)
                #print("Winners!")
    hist = MLP.hist_unit_win_lists(model, layer=1)
    return

model = MLP


def train_layer2(model, x):
    samples, width = x.shape
    l2 = model.l2
    for s in range(samples):
        h1 = F.relu(model.l1(x))
        h2 = F.relu(l2(h1))
        if l2.unit_count.array == 0:
            print("Make first unit for layer2")
            MLP.create_memory_unit(model, layer=2, h=h1.array, i=s)
        else:
            winners = MLP.count_winners(model, layer=2, hi=h2.array[s])
            if winners == False:
                print("No L2 Winners for x[" + str(s) + "]")
                MLP.create_memory_unit(model, layer=2, h=h1.array, i=s)
            else:
                MLP.update_centroids_and_vars(model, l2, h1.array[s], h2[s].array)
                MLP.update_win_counts_and_lists(model, l2, xi=s, hn=h2[s].array)
                #print("Winners!")
    print("Layer1 Weights:" + str(model.l1.W.shape))
    print("Layer2 Weights:" + str(model.l2.W.shape))

    return


def train_layer3(model, x):
    samples, width = x.shape
    l2 = model.l2
    l3 = model.l3
    for s in range(samples):
        h1 = F.relu(model.l1(x))
        h2 = F.relu(l2(h1))
        model.l3(h2)
        if l2.unit_count.array == 0:
            print("Make first unit for layer2")
            MLP.create_memory_unit(model, layer=2, h=h1.array, i=s)
        else:
            winners = MLP.count_winners(model, layer=2, hi=h2.array[s])
            if winners == False:
                print("No L2 Winners for x[" + str(s) + "]")
                MLP.create_memory_unit(model, layer=2, h=h1.array, i=s)
            else:
                MLP.update_centroids_and_vars(model, l2, h1.array[s], h2[s].array)
                MLP.update_win_counts_and_lists(model, l2, xi=s, hn=h2[s].array)
                #print("Winners!")
    print("Layer1 Weights:" + str(model.l1.W.shape))
    print("Layer2 Weights:" + str(model.l2.W.shape))