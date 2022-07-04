import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import bisplrep, bisplev
from .history import History

### FACTORING OUT HISTORY MANAGEMENT (it's the same for all simple Preisach models)
class History_primitive():
    def __init__(self):
        self.history_object = History()

    @property
    def history(self):
        return self.history_object.value

    @history.setter
    def history(self, hist_list):
        self.history_object.value = hist_list

    def reset_history(self):
        self.history_object.reset()

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

# INTEGRAL-BASED IMPLEMENTATION (it slo)
class Integral_Preisach(History_primitive):
    def __init__(self, mu, bounds):

        self.mu = mu
        self.bounds = tuple(bounds)
        self.history_object = History()


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
        self.history_object.update(V)
        return self.get_value()

# SUM-BASED IMPLEMENTATION (it fast but it don't work)
class Preisach(History_primitive):
    def __init__(self, preisach_coords, measured_preisach_mesh, strict_f = False):
        # centering the f values and saving the corresponding offset

        offset = (np.max(measured_preisach_mesh) + np.min(measured_preisach_mesh))/2
        measured_preisach_mesh = measured_preisach_mesh - offset
        self.measured_preisach_mesh = measured_preisach_mesh
        self.preisach_coords = preisach_coords
        self.offset = offset

        self.bounds = (np.min(preisach_coords), np.max(preisach_coords))

        self.f = self.generate_model(preisach_coords, measured_preisach_mesh, strict_f)
        self.history_object = History() # this will be a n by 2 array . history[i, :] = [M_i, m_i]
        self.f_plus = self.f(self.bounds[-1], self.bounds[-1])

    def generate_model(self, preisach_coords, measured_preisach_mesh, strict_f = True):
        """Interpolates the $f_{\alpha, \beta}$ function from measurements and returns it."""
        alpha_values = preisach_coords[:, 0]
        beta_values = preisach_coords[:, 1]

        min_u = np.min(alpha_values)
        max_u = np.max(alpha_values)
        self.model_bounds = self.bounds

        spline_representation = bisplrep(alpha_values, beta_values, measured_preisach_mesh, kx = 5, ky = 5)
        def interpolated_mesh(x, y):
            if x > max_u :
                x = max_u

            if y < min_u :
                y = min_u

            return bisplev(x, y, spline_representation)

        if strict_f :
            def mesh(alpha, beta):
                if alpha < beta :
                    raise ValueError(f"Preisach function called outside Preisach triangle : {alpha}, {beta}")
                else :
                    return interpolated_mesh(alpha, beta)
        else :
            mesh = interpolated_mesh

        mesh = np.vectorize(mesh)

        return mesh

    def get_value(self, **kwargs):

        f_func = self.f

        if "history" in kwargs.keys():
            hist = kwargs["history"]
        else :
            hist = self.history

        max_values = hist[:, 0] # the list of M_k values
        mins_k = hist[:, 1] # the list of m_k values
        mins_km1 = np.roll(mins_k, 1) # roll it, and then
        mins_km1[0] = self.model_bounds[0] # ... set a value for m_0 to get the list of k_{-1}

        terms = f_func(max_values, mins_k) - f_func(max_values, mins_km1)

        return self.offset - self.f_plus + np.sum(terms)

    def to_value(self, V):
        self.history_object.update(V)
        return self.get_value()

    def plot_mesh(self, **kwargs):
        beta = self.preisach_coords[:, 1]
        alpha = self.preisach_coords[:, 0]
        plt.tripcolor(beta, alpha, self.f(alpha, beta), **kwargs)
