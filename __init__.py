import numpy as np
from scipy.interpolate import interp2d

class Integral_Preisach():
    """An implementation of the Preisach model using integration to get values. Very slow, but useful for testing results of other methods against."""
    def __init__(self, mu, bounds):

        self.mu = mu
        self.bounds = tuple(bounds)
        self.history = None # this will be a n by 2 array . __history[i, :] = [M_i, m_i]


    def generate_hysteron_state_function(self):
        """NOT USED ANYMORE BUT CAN HELP WITH PLOTTING. Returns a function defined on the alpha-beta plane whose value at (alpha, beta) is that of the hysteron with parameters (alpha, beta)."""
        hist = self.history

        if hist is None :
            return lambda x, y: -1 # if there is no history, by convention we say that we are at negative saturation
        else :
            l = hist.shape[0]

            def delta(alpha, beta):
                for x in np.flip(hist, axis = 0): # going through the history backwards.
                    M = x[0]
                    m = x[1]

                    if beta > alpha :
                        return 0
                    elif beta >= m :
                        return -1
                    elif alpha <= M:
                        return 1
                else :
                    return -1

            return delta

    # MANAGING PREISACH HISTORY
    def update_history(self, V):
        if self.history is None : # if there is no history, we can just initialize it to the first value of V.
            self.history = np.array([[V, V]])
        else : # we can insert a new value
            self.insert_value(V)

    def reset_history(self):
        self.history = None

    def insert_value(self, V):
        hist = self.history
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

        self.history = hist # updating history

    def make_integrand_bound_functions(self):
        hist = self.history
        flipped_hist = np.flip(hist, axis = 0)

        bounds = self.bounds

        # generating only the lower points of the staircase front
        hist_alpha, hist_beta = hist.T
        hist_beta = np.roll(hist_beta, 1)
        hist_beta[0] = bounds[0]
        lower_front = np.zeros_like(hist)
        lower_front[:, 0] = hist_alpha
        lower_front[:, 1] = hist_beta

        flipped_lower_front = np.flip(lower_front, axis = 0)

        @np.vectorize
        def spin_up_max_beta_bound(alpha):
            if alpha <= hist[-1, 1]:
                return alpha

            else :
                for i, x in enumerate(flipped_hist):
                    M, m = x
                    if alpha <= M :
                        return m
                else :
                    return bounds[0]

        @np.vectorize
        def spin_down_min_alpha_bound(beta):
            if beta >= hist[-1, 1]:
                return beta
            else :
                for i, x in enumerate(flipped_lower_front):
                    prev_M, m = x
                    if beta >= m:
                        return prev_M # no neeed for an "else" clause cause this should make a good partition of the whole interval

        return spin_up_max_beta_bound, spin_down_min_alpha_bound

    def get_value(self):
        hist = self.history
        bounds = self.bounds
        if hist is None :
            return 0
        else :
            mu = self.mu
            mu_rev = lambda x, y : mu(y, x)

            spin_up_max_beta_bound, spin_down_min_alpha_bound = self.make_integrand_bound_functions()

            integral_up, _ = dblquad(mu, bounds[0], hist[0, 0], lambda x : bounds[0], spin_up_max_beta_bound)
            integral_down, _ = dblquad(mu_rev, *bounds, spin_down_min_alpha_bound, lambda x : bounds[1])
            return integral_up - integral_down

    def to_value(self, V):
        self.update_history(V)
        return self.get_value()

    def make_flipping_front(self):
        hist = self.history

        m, M = self.bounds

        if hist is None :
            return np.array([[m, m],
                             [m, m]])
        else :
            hist_shape = hist.shape
            corners_shape = (hist_shape[0]*2, hist_shape[1])
            corners = np.zeros(corners_shape)
            up_corners = hist

            hist_alpha, hist_beta = hist.T
            hist_beta = np.roll(hist_beta, 1)
            hist_beta[0] = m
            down_corners = np.zeros_like(up_corners)
            down_corners[:, 0] = hist_alpha
            down_corners[:, 1] = hist_beta

            corners[::2, :] = down_corners
            corners[1::2, :] = up_corners

            if corners[-1, 0] != corners[-1, 1]:
                corners = np.concatenate((corners, np.ones((1, 2))*corners[-1, -1]))

            return corners
