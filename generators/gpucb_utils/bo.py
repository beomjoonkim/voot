import helper
import numpy as np
from generators.gpucb_utils.functions import Domain


class BO(object):
    def __init__(self, model, acq_fcn, domain, opt_n=1e4):
        self.model = model
        self.evaled_x = []
        self.evaled_y = []
        self.acq_fcn = acq_fcn
        self.domain = domain
        self.opt_n = opt_n

    def choose_next_point(self, evaled_x, evaled_y):
        update_hyper_params = len(evaled_x) % 10 == 0
        self.model.update(evaled_x, evaled_y, update_hyper_params)
        x, acq_fcn_val = helper.global_minimize(self.acq_fcn,
                                                self.acq_fcn.fg,
                                                self.domain,
                                                self.opt_n)
        return x

    def generate_evals(self, T):
        for i in range(T):
            yield self.choose_next_point()


class AddBO(BO):
    def __init__(self, model, acq_fcn, domain, opt_n=1e4):
        BO.__init__(self, model, acq_fcn, domain, opt_n=1e4)
        domain_min = domain.domain[0]
        domain_max = domain.domain[1]
        dim_x = len(domain_min)
        if dim_x == 10:
            decomposed_min = [domain_min[0:3], domain_min[3:6], domain_min[6:9], [domain_min[-1]]]
            decomposed_max = [domain_max[0:3], domain_max[3:6], domain_max[6:9], [domain_max[-1]]]
        elif dim_x == 20:
            decomposed_min = [domain_min[0:4], domain_min[4:8], domain_min[8:12], domain_min[12:16], domain_min[16:20]]
            decomposed_max = [domain_max[0:4], domain_max[4:8], domain_max[8:12], domain_max[12:16], domain_max[16:20]]
        else:
            raise NotImplemented

        self.decomposed_domain = []
        for min_, max_ in zip(decomposed_min, decomposed_max):
            print np.array([min_, max_])
            domain_ = Domain(0, np.array([min_, max_]))
            self.decomposed_domain.append(domain_)

    def choose_next_point(self, evaled_x, evaled_y):
        # go through each function in acq_function
        n_evals = len(evaled_x)
        update_hyper_params = n_evals % 20 == 0
        self.model.update(evaled_x, evaled_y, update_hyper_params)  # update every 10th
        x = []
        for idx, ucb in enumerate(self.acq_fcn.add_ucb):
            domain_ = self.decomposed_domain[idx]
            ucb.zeta = np.sqrt(0.2 * self.acq_fcn.decomposition_dim[idx] * np.log(2*n_evals))
            x_, _ = helper.global_minimize(ucb,
                                           ucb.fg,
                                           domain_,
                                           self.opt_n)
            x.append(x_)
        return np.hstack(x)



