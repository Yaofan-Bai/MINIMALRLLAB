import numpy as np


class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count
        new_var = m2 / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = total_count


class ObsNormalizer:
    def __init__(self, obs_dim, clip=10.0, eps=1e-8):
        self.rms = RunningMeanStd(obs_dim)
        self.clip = clip
        self.eps = eps

    def normalize(self, obs, update=True):
        obs = np.asarray(obs, dtype=np.float32)
        if update:
            self.rms.update(obs)
        normed = (obs - self.rms.mean) / np.sqrt(self.rms.var + self.eps)
        if self.clip is not None:
            normed = np.clip(normed, -self.clip, self.clip)
        return normed.astype(np.float32)

    def state_dict(self):
        return {
            "mean": self.rms.mean,
            "var": self.rms.var,
            "count": self.rms.count,
            "clip": self.clip,
            "eps": self.eps,
        }

    def load_state_dict(self, state):
        self.rms.mean = state["mean"]
        self.rms.var = state["var"]
        self.rms.count = state["count"]
        self.clip = state.get("clip", self.clip)
        self.eps = state.get("eps", self.eps)

    def save(self, path: str):
        np.savez(
            path,
            mean=self.rms.mean,
            var=self.rms.var,
            count=self.rms.count,
            clip=self.clip,
            eps=self.eps,
        )

    def load(self, path: str):
        data = np.load(path)
        self.rms.mean = data["mean"]
        self.rms.var = data["var"]
        self.rms.count = float(data["count"])
        self.clip = float(data["clip"])
        self.eps = float(data["eps"])
