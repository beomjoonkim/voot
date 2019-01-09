import numpy as np
import helper


class BO(object):
    def __init__(self, model, acq_fcn, domain, opt_n=1e4):
        self.model = model
        self.evaled_x = []
        self.evaled_y = []
        self.acq_fcn = acq_fcn
        self.domain = domain
        self.opt_n = opt_n

    def choose_next_point(self, evaled_x, evaled_y):
        self.model.update(evaled_x, evaled_y)
        x, acq_fcn_val = helper.global_minimize(self.acq_fcn,
                                                self.acq_fcn.fg,
                                                self.domain,
                                                self.opt_n)
        return x


    def generate_evals(self, T):
        for i in range(T):
            yield self.choose_next_point()