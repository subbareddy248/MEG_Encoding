import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Ridge, Reservoir, ESN
from joblib import Parallel, delayed



from multiprocessing import Manager

import numpy as np
from joblib import Parallel, delayed

from reservoirpy._base import _Node, call
from reservoirpy.utils import progress, verbosity
from reservoirpy.utils.graphflow import dispatch
from reservoirpy.utils.model_utils import to_data_mapping
from reservoirpy.utils.validation import is_mapping
from reservoirpy.nodes.esn import _sort_and_unpack

    
class Seq2VecESN(ESN):
    
    def __init__(self, **kwargs):
        super(Seq2VecESN, self).__init__(**kwargs)
        
    def run(
        self,
        X=None,
        forced_feedbacks=None,
        from_state=None,
        stateful=True,
        reset=False,
        shift_fb=True,
        return_states=None,
    ):

        X, forced_feedbacks = to_data_mapping(self, X, forced_feedbacks)

        self._initialize_on_sequence(X[0], forced_feedbacks[0])

        def run_fn(idx, x, forced_fb):
            
            all_states = {}
            states = self.reservoir.run(x[self.reservoir.name])
            final_state = np.atleast_2d(states[-1, :])    
            
            if return_states is not None and "reservoir" in return_states:
                all_states["reservoir"] = states
                
            y = self.readout(final_state)
            
            all_states["readout"] = y

            return idx, all_states

        backend = backend=self.backend

        seq = progress(X, f"Running {self.name}")

        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers, backend=backend) as parallel:
                states = parallel(
                    delayed(run_fn)(idx, x, y)
                    for idx, (x, y) in enumerate(zip(seq, forced_feedbacks))
                )

        return _sort_and_unpack(states, return_states=return_states)

    def fit(self, X=None, Y=None, from_state=None, stateful=True, reset=False):

        X, Y = to_data_mapping(self, X, Y)
        self._initialize_on_sequence(X[0], Y[0])

        self.initialize_buffers()

        if (self.workers > 1 or self.workers < 0) and self.backend not in (
            "sequential",
            "threading",
        ):
            lock = Manager().Lock()
        else:
            lock = None

        def run_partial_fit_fn(x, y):
            states = self.reservoir.run(x[self.reservoir.name])
            final_state = np.atleast_2d(states[-1, :])

            # Avoid any problem related to multiple
            # writes from multiple processes
            # if lock is not None:
            # with lock:  # pragma: no cover
            self.readout.partial_fit(final_state, y[self.readout.name], lock=lock)
            # else:
            #     self.readout.partial_fit(states, y[self.readout.name])

        backend = self.backend

        seq = progress(X, f"Running {self.name}")
        with self.with_state(from_state, reset=reset, stateful=stateful):
            with Parallel(n_jobs=self.workers, backend=backend) as parallel:
                parallel(delayed(run_partial_fit_fn)(x, y) for x, y in zip(seq, Y))

            if verbosity():  # pragma: no cover
                print(f"Fitting node {self.name}...")

            self.readout.fit()

        return self


def MEG2phoneme_seq(params):
    
    P_res = params.reservoir
    P_read = params.readout
    
    reservoir = Reservoir(
        P_res.units,
        P_res.lr,
        P_res.sr,
        P_res.input_bias,
        P_res.input_scaling,
        P_res.bias_scaling,
    )
    
    readout = Ridge(P_res.ridge)
    
    model = ESN(reservoir=reservoir, readout=readout)

    return model


def MEG2phoneme_vec(params):
    
    P_res = params.reservoir
    P_read = params.readout
    
    reservoir = Reservoir(
        P_res.units,
        P_res.lr,
        P_res.sr,
        P_res.input_bias,
        P_res.input_scaling,
        P_res.bias_scaling,
    )
    
    readout = Ridge(P_res.ridge)

    model = Seq2VecESN(reservoir=reservoir, readout=readout)
    
    return model


if __name__ == "__main__":
    
    rpy.set_seed(42)
    rpy.verbosity(0)
    
    res = Reservoir(50)
    readout = Ridge()
    
    model = Seq2VecESN(reservoir=res, readout=readout)
    
    x = np.ones((10, 100, 5))
    y = np.ones((10, 1, 1))
    
    model.fit(x, y)
    
    y_pred = model.run(x)
    
    assert len(y_pred) == 10
    assert y_pred[0].shape == (1, 1)
    
    states = model.run(x, return_states=["reservoir", "readout"])
    
    assert states["readout"][0].shape == (1, 1)
    assert states["reservoir"][0].shape == (100, 50)
