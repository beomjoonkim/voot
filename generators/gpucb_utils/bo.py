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
        print 'GP updating the model, n_data is ', len(evaled_x)
        self.model.update(evaled_x, evaled_y)
        print 'GP model updated!'
        if len(evaled_x) == 0 or np.all(evaled_y) == -2:
            print 'GP choosing to random sample'
            dim_x = self.domain.domain.shape[-1]
            domain_min = self.domain.domain[0]
            domain_max = self.domain.domain[1]
            x = np.random.uniform(domain_min, domain_max, (1, dim_x)).squeeze()
        else:
            x, acq_fcn_val = helper.global_minimize(self.acq_fcn,
                                                    self.acq_fcn.fg,
                                                    self.domain,
                                                    self.opt_n)
        return x


    def generate_evals(self, T):
        for i in range(T):
            yield self.choose_next_point()