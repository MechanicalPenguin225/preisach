import numpy as np
from scipy.integrate import dblquad
from .history import History

class Integral_Preisach():
    def __init__(self, mu, bounds):

        self.mu = mu
        self.bounds = tuple(bounds)
        self.__history_object = History()

    @property
    def history(self):
        return self.__history_object.value

    def reset_history(self):
        self.__history_object.reset()

    def generate_hysteron_state_function(self):
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


    def make_integrand_bound_functions(self):
        hist = self.history
        flipped_hist = np.flip(hist, axis = 0)

        bounds = self.bounds

        # generating poonly the lower points of the staircase front
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
        self.__history_object.update(V)
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
