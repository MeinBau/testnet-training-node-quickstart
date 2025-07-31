import optuna
from dataclasses import dataclass
from typing import Callable

@dataclass
class OptunaConfig:
    n_trials: int = 20
    direction: str = "maximize"

class OptunaLoraTuner:
    def __init__(self, train_fn: Callable, eval_fn: Callable, optuna_config: OptunaConfig):
        """
        train_fn: (params) → score 를 반환하는 학습 함수
        eval_fn: 학습 완료 후 평가 점수를 반환하는 함수
        optuna_config: Optuna 설정
        """
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self.optuna_config = optuna_config
        self.study = None

    def objective(self, trial):
        params = {
            "lora_rank": trial.suggest_int("lora_rank", 8, 64),
            "lora_alpha": trial.suggest_int("lora_alpha", 8, 64),
            "lora_dropout": trial.suggest_float("lora_dropout", 0.0, 0.2),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        }
        score = self.train_fn(params)   
        return score

    def run(self):
        self.study = optuna.create_study(direction=self.optuna_config.direction)
        self.study.optimize(self.objective, n_trials=self.optuna_config.n_trials)
        return self.study.best_params

    def get_best_params(self):
        if self.study is None:
            raise RuntimeError("Optuna 최적화를 먼저 실행해야 합니다.")
        return self.study.best_params
