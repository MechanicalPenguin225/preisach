import numpy as np
from warnings import warn
from .preisach import Preisach

class Transducer(Preisach):
    def __init__(self, preisach_coords, measured_preisach_mesh, second_order = True, alpha = 0.9, tolerance = 10e6, convergence_limit = 500, gamma = None):
        super().init(preisach_coords, measured_preisach_mesh, second_order)
        self.alpha = alpha
        self.tolerance = tolerance
        self.output_bounds = (np.min(measured_preisach_mesh), np.max(measured_preisach_mesh))
        self.convergence_limit = convergence_limit

        if gamma is None :
            delta_V = self.bounds[1] - self.bounds[0]
            delta_f = self.output_bounds[1] - self.output_bounds[0]
            self.gamma = delta_f/delta_V
        else :
            self.gamma = gamma

    def check_validity(self, setpoint):
        if setpoint < self.output_bounds[0] :
            warn("Preisach setpoint out of reachable values - too low.")
            return self.output_bounds[0]
        elif setpoint > self.output_bounds[1]:
            warn("Preisach setpoint out of reachable values - too high.")
            return self.output_bounds[1]
        else :
            return setpoint

    def aim(self, setpoint):
        setpoint = self.check_validity(setpoint)
        f_start = self.get_value()
        real_history = self.history
        delta_V_final = 0.

        counter = 0
        delta_f = np.abs(f_start - setpoint)

        while delta_f >= tolerance :

            if counter > self.convergence_limit :
                raise Error("Preisach failed to converge.")
            else :
                counter += 1

            delta_V_lin = delta_f / gamma
            delta_V_lin_scaled = self.alpha*delta_V_lin

            current_f = self.to_value(self.current_input_value + delta_V_lin_scaled)
            delta_f = np.abs(current_f - setpoint)

            delta_V_final += delta_V_lin_scaled

        self.history = real_history
        next_voltage = delta_V_final + self.current_input_value
        self.history_object.update(next_voltage)
        return next_voltage


