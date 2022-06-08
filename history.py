import numpy as np

class History():
    def __init__(self):
        self.value = None

    def update(self, V):
        if self.value is None : # if there is no history, we can just initialize it to the first value of V.
            self.value = np.array([[V, V]])
        else : # we can insert a new value
            self.insert(V)

    def reset(self):
        self.value = None

    def insert(self, V):
        hist = self.value
        l = len(hist)

        if V >= hist[-1, 0] :
            # we need to do some forgetting cause our max was higer than previous maxes
            last_good_index = 0

            for i in range(l): # we are gonna progressively reverse thru the array to find out up to which point we must go.
                decreasing_index = l-1-i
                h = hist[decreasing_index, 0]

                if h > V: # assuming hist is properly reduced, the first value we encounter that is > V is the last good value.
                    last_good_index = decreasing_index
                    hist[last_good_index + 1, :] = V # setting both values to V since min gets erased in the process.
                    break
            else : # if we never broke, it means that there is no good max (either we are higher than all of them, or there is just a single vector in the history and it's not good).
                hist = np.array([[V, V]]) # since we start with max, we reset the whole thing.


        elif V <= hist[-1, 1]:
            # we also need to do some forgetting, this time cause our min was smaller than previous mins

            for i in range(l):
                decreasing_index = l - i - 1
                h = hist[decreasing_index, 1]

                if h < V :
                    last_good_index = decreasing_index
                    hist[last_good_index + 1, 1] = V
                    hist = hist[:last_good_index + 2, :] # +2 bc +1 cuts at last_good_index and we replaced the next one to be good so it gets included too
                    break
            else : # if we never broke, it means that there is no good min (either we are lower than all of them, or there is just a single vector in the history and it's not good).
                hist = hist[0, :].reshape(1, 2)
                hist[0, 1] = V # means we must set the first min, though we'll keep the preceding max safe.


        else : # don't have to do any forgetting. We got a new history vector to add!
            hist = np.concatenate((hist, np.array([V, V]).reshape(1, 2)), axis = 0)

        self.value = hist # updating history

