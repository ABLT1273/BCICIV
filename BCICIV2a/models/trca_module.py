from __future__ import annotations

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class TRCAFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    用 TRCA(Task-Related Component Analysis) 提取多类 BCI 特征。

    这里采用“每个类别各自学习一组 TRCA 空间滤波器”的做法：
    1. 对某一类的训练 trial，最大化 trial 之间的相关协方差。
    2. 求出该类别最稳定、最“任务相关”的空间滤波器。
    3. 用这些滤波器把任意 trial 投影到低维空间，再提取 log-variance 与模板相关性特征。

    对于运动想象任务，这种做法能更强调“同类 trial 之间的稳定模式”，
    往往比纯频带能量更贴近任务相关神经活动。
    """

    def __init__(self, n_components: int = 3, reg: float = 1e-6):
        self.n_components = n_components
        self.reg = reg
        self.classes_: np.ndarray | None = None
        self.filters_: dict[object, np.ndarray] = {}
        self.templates_: dict[object, np.ndarray] = {}

    @staticmethod
    def _center_trial(trial: np.ndarray) -> np.ndarray:
        """去掉每个通道的均值，避免直流偏置干扰协方差估计。"""

        return trial - trial.mean(axis=1, keepdims=True)

    def _fit_single_class(self, X_class: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        对某一类 trial 求 TRCA 滤波器。

        X_class 形状: (n_trials, n_channels, n_samples)
        """

        centered_trials = [self._center_trial(trial) for trial in X_class]
        n_channels = centered_trials[0].shape[0]

        # S 统计“同一类别不同 trial 之间”的互协方差，是 TRCA 的任务相关部分。
        S = np.zeros((n_channels, n_channels), dtype=np.float64)
        for i in range(len(centered_trials)):
            Xi = centered_trials[i]
            for j in range(i + 1, len(centered_trials)):
                Xj = centered_trials[j]
                S += Xi @ Xj.T + Xj @ Xi.T

        # Q 统计 trial 自身的总协方差，用于约束滤波器不要只拟合噪声。
        Q = np.zeros((n_channels, n_channels), dtype=np.float64)
        for Xi in centered_trials:
            Q += Xi @ Xi.T

        Q += self.reg * np.eye(n_channels)

        eigenvalues, eigenvectors = eigh(S, Q)
        order = np.argsort(eigenvalues)[::-1]
        filters = eigenvectors[:, order[: self.n_components]]
        template = np.mean(centered_trials, axis=0)
        return filters, template

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCAFeatureExtractor":
        self.classes_ = np.unique(y)
        self.filters_.clear()
        self.templates_.clear()

        for class_id in self.classes_:
            X_class = X[y == class_id]
            filters, template = self._fit_single_class(X_class)
            self.filters_[class_id] = filters
            self.templates_[class_id] = template

        return self

    def _extract_single_trial(
        self,
        trial: np.ndarray,
    ) -> tuple[list[float], list[float]]:
        if self.classes_ is None:
            raise RuntimeError("请先调用 fit 再提取 TRCA 特征。")

        centered_trial = self._center_trial(trial)
        trial_features: list[float] = []
        template_scores: list[float] = []

        for class_id in self.classes_:
            filters = self.filters_[class_id]
            template = self.templates_[class_id]

            projected_trial = filters.T @ centered_trial
            projected_template = filters.T @ template

            log_variances = np.log(np.var(projected_trial, axis=1) + 1e-10)
            trial_features.extend(log_variances.tolist())

            component_correlations: list[float] = []
            for component_index in range(projected_trial.shape[0]):
                trial_component = projected_trial[component_index]
                template_component = projected_template[component_index]

                if np.std(trial_component) < 1e-10 or np.std(template_component) < 1e-10:
                    correlation = 0.0
                else:
                    correlation = float(
                        np.corrcoef(trial_component, template_component)[0, 1]
                    )

                component_correlations.append(correlation)
                trial_features.append(correlation)

            # 模板分数单独保存，后面可以直接作为“TRCA 模板匹配概率”的基础。
            template_scores.append(float(np.mean(component_correlations)))

        return trial_features, template_scores

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("请先调用 fit 再调用 transform。")

        features: list[list[float]] = []

        for trial in X:
            trial_features, _ = self._extract_single_trial(trial)
            features.append(trial_features)

        return np.asarray(features, dtype=np.float32)

    def get_template_scores(self, X: np.ndarray) -> np.ndarray:
        """返回每个 trial 对每个类别模板的相关分数。"""

        if self.classes_ is None:
            raise RuntimeError("请先调用 fit 再调用 get_template_scores。")

        template_scores: list[list[float]] = []
        for trial in X:
            _, scores = self._extract_single_trial(trial)
            template_scores.append(scores)
        return np.asarray(template_scores, dtype=np.float32)

    def get_feature_names(self) -> list[str]:
        if self.classes_ is None:
            return []

        feature_names: list[str] = []
        for class_id in self.classes_:
            for component_index in range(self.n_components):
                feature_names.append(
                    f"trca_class_{class_id}_component_{component_index}_logvar"
                )
            for component_index in range(self.n_components):
                feature_names.append(
                    f"trca_class_{class_id}_component_{component_index}_corr"
                )
        return feature_names


class TRCAHybridClassifier(BaseEstimator):
    """
    更强的 TRCA 分类头：
    1. 一路使用 TRCA 模板相关性打分，保留 TRCA 的“任务模板匹配”优势。
    2. 一路使用 RBF-SVM，对 TRCA 特征做非线性分类。
    3. 最后把两路概率加权融合，兼顾模板鲁棒性与判别能力。
    """

    def __init__(
        self,
        n_components: int = 3,
        reg: float = 1e-6,
        svm_c: float = 2.0,
        svm_gamma: str = "scale",
        template_weight: float = 0.4,
    ):
        self.n_components = n_components
        self.reg = reg
        self.svm_c = svm_c
        self.svm_gamma = svm_gamma
        self.template_weight = template_weight
        self.extractor = TRCAFeatureExtractor(n_components=n_components, reg=reg)
        self.classifier = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "svm",
                    SVC(
                        kernel="rbf",
                        C=svm_c,
                        gamma=svm_gamma,
                        probability=True,
                    ),
                ),
            ]
        )
        self.classes_: np.ndarray | None = None

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TRCAHybridClassifier":
        self.extractor.fit(X, y)
        trca_features = self.extractor.transform(X)
        self.classifier.fit(trca_features, y)
        self.classes_ = self.classifier.named_steps["svm"].classes_
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.extractor.transform(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("请先调用 fit 再调用 predict_proba。")

        trca_features = self.extractor.transform(X)
        svm_probs = self.classifier.predict_proba(trca_features)
        template_scores = self.extractor.get_template_scores(X)
        template_probs = self._softmax(template_scores)

        return (
            self.template_weight * template_probs
            + (1.0 - self.template_weight) * svm_probs
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(X)
        return self.classes_[np.argmax(probabilities, axis=1)]
