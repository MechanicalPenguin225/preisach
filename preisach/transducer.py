import numpy as np
from warnings import warn
from .models import Preisach
from copy import deepcopy

class Transducer(Preisach):
    def __init__(self, preisach_coords, measured_preisach_mesh, second_order = True, alpha = 0.9, tolerance = 1e6, convergence_limit = 500, gamma = None, initial_value = None):
        super().__init__(preisach_coords, measured_preisach_mesh, second_order)
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

        if initial_value is None :
            self.history_object.update(self.bounds[0]) # initializing at min possible voltage if no initial value is provided.
        else :
            self.history_object.update(initial_value) # else, initializing to that value.

    def check_validity(self, setpoint):
        if setpoint < self.output_bounds[0] :
            raise ValueError("Preisach setpoint out of reachable values - too low.")
            return self.output_bounds[0]
        elif setpoint > self.output_bounds[1]:
            raise ValueError("Preisach setpoint out of reachable values - too high.")
            return self.output_bounds[1]
        else :
            return setpoint

    def aim(self, setpoint):
        setpoint = self.check_validity(setpoint)
        f_start = self.get_value()
        real_history = deepcopy(self.history)
        V_final = self.current_input_value

        clipped = False # whether we clipped in a previos step.

        counter = 0
        initial_delta_f = setpoint - f_start

        delta_f = initial_delta_f

        while np.abs(delta_f) >= self.tolerance :
            if counter > self.convergence_limit :
                raise ValueError("Preisach failed to converge.")
            else :
                counter += 1

            if delta_f*initial_delta_f < 0: # checking if we overshot, at which case we redo

                prev_alpha = deepcopy(self.alpha)
                self.alpha /= 2

               # print(f"Overshot. New alpha = {self.alpha}")

                self.history = real_history

                ans = self.aim(setpoint)
                self.alpha = prev_alpha

                return ans
            elif clipped : # if we already clipped once and neither overshot nor satisfied the preicision conditions, we're screwed. Best thing we can do is throw a warning and return the clipped value anyway.
                warn("Clipped value to model bound. Precision was not met.")
                break

            delta_V_lin = delta_f / self.gamma
            delta_V_lin_scaled = self.alpha*delta_V_lin

            new_V_final, clipped = self.clip_input(V_final + delta_V_lin_scaled)

            current_f = self.to_value(new_V_final)
            delta_f = setpoint - current_f

            V_final = new_V_final

        self.history = real_history
        self.history_object.update(V_final)
        return V_final


