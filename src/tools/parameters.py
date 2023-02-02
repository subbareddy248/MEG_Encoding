class Parameters(dict):
    def __getattr__(self, attr):
        try:
            if isinstance(self[attr], dict):
                return Parameters(**self[attr])
            return self[attr]
        except KeyError:
            raise AttributeError(attr)

    def __getitem__(self, item):
        if item in self:
            if isinstance(self.get(item), dict):
                return Parameters(**self.get(item))
            return self.get(item)
        else:
            raise KeyError(item)

    def __repr__(self):
        return "Parameters " + super(Parameters, self).__repr__()

    def __str__(self):
        return "Parameters " + super(Parameters, self).__str__()
    
    @property
    def sfreq(self):
        if "epochs" in self and "decim" in self.epochs:
            return self.sampling_rate / self.epochs.decim
        else:
            return self.sampling_rate


P_ph_meg = Parameters(
    bandpass=dict(high=30.0, low=0.5),
    scaler="RobustScaler",
    quantile_range=(25.0, 75.0),
    seed=42,
    epochs=dict(decim=10, tmin=-0.2, tmax=0.6, baseline=(-0.2, 0.0)),
    sampling_rate=1000,
    threshold=20.0,
)