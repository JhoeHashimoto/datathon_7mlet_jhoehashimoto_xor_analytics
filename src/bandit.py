"""
Implementação do algoritmo Thompson Sampling usado no Datathon.
Mantido separado do notebook para poder ser importado pela API (Etapa 5).
"""
import numpy as np


class ThompsonSampling:
    """Bandit multi-braço com priors Beta(alpha, beta) por braço."""

    def __init__(self, arms, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.arms = list(arms)
        self.alpha = {a: alpha_prior for a in self.arms}
        self.beta = {a: beta_prior for a in self.arms}

    def select_arm(self) -> str:
        samples = {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: int) -> None:
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def state(self) -> dict:
        return {"alpha": self.alpha, "beta": self.beta, "arms": self.arms}

    @classmethod
    def from_state(cls, state: dict) -> "ThompsonSampling":
        obj = cls(state["arms"])
        obj.alpha = state["alpha"]
        obj.beta = state["beta"]
        return obj
