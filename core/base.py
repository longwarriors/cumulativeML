"""
基础抽象类模块：定义项目中所有组件的基础接口和抽象类，确保一致的设计模式。
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
import numpy as np
import pandas as pd
import logging


class BaseProcessor(ABC):
    """
    基础处理器类：所有数据处理器的基类，定义了通用的处理器接口和基础功能。
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.is_fitted = False
        self.metadata = {}

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'BaseProcessor':
        """
        拟合处理器，通常用于学习数据的统计特性或模型参数。
        :param data: 输入数据，通常是一个 pandas DataFrame。
        :param kwargs: 其他可选参数。
        :return self: 返回自身以支持链式调用。
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        转换数据，应用处理器的逻辑到输入数据上。
        :param data: 输入数据，通常是一个 pandas DataFrame。
        :param kwargs: 其他可选参数。
        :return: 转换后的数据，通常是一个 pandas DataFrame。
        """
        pass

    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        拟合并转换数据，先调用 fit 方法，然后调用 transform 方法。
        :param data: 输入数据，通常是一个 pandas DataFrame。
        :param kwargs: 其他可选参数。
        :return: 转换后的数据，通常是一个 pandas DataFrame。
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)

    def get_metadata(self) -> Dict[str, Any]:
        """
        获取处理器的元数据，通常包含处理器的配置信息和状态。
        :return: 包含元数据的字典。
        """
        return self.metadata.copy()

    def _validate_fitted(self):
        """
        验证处理器是否已拟合。
        :raises RuntimeError: 如果处理器未拟合。
        """
        if not self.is_fitted:
            raise RuntimeError(f"{self.__class__.__name__} 尚未拟合，请先调用fit方法")

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")

    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BaseModel(ABC):
    """
    基础模型类：所有机器学习模型的基类，定义了通用的模型接口和基础功能。
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.is_trained = False
        self.model = None
        self.training_history = {}
        self.feature_names = []

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> Dict:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        pass

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        预测概率（如果支持），默认实现返回 None，子类可覆盖。
        :param X: 输入特征数据，通常是一个 pandas DataFrame。
        :param kwargs: 其他可选参数。
        :return: 预测概率数组。
        """
        raise NotImplementedError(f"该模型 {self.__class__.__name__} 不支持概率预测")

    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        获取特征重要性（如果支持），默认实现返回 None，子类可覆盖。
        :return: 特征重要性 Series。
        """
        return None

    def get_training_history(self) -> Dict:
        """
        获取训练历史记录，通常包含训练过程中的损失值或其他指标。
        :return: 包含训练历史的字典。
        """
        return self.training_history.copy()

    def _validate_trained(self):
        """
        验证模型是否已训练。
        :raises RuntimeError: 如果模型未训练。
        """
        if not self.is_trained:
            raise RuntimeError(f"{self.__class__.__name__} 尚未训练，请先调用train方法")

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")

    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BaseEvaluator(ABC):
    """
    基础评估器类：所有模型评估器的基类，定义了通用的评估接口和基础功能。
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.evaluation_results = {}

    @abstractmethod
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> Dict:
        """
        评估模型性能，计算各种指标。
        :param y_true: 真实标签数组。
        :param y_pred: 预测标签数组。
        :param kwargs: 其他可选参数。
        :return: 包含评估结果的字典。
        """
        pass

    def get_evaluation_results(self) -> Dict:
        """获取评估结果"""
        return self.evaluation_results.copy()

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")

    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class BasePipeline(ABC):
    """
    基础管道类：定义数据处理、模型训练和评估的完整流程。
    """

    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        self.config = config or {}
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.steps = []
        self.results = {}

    @abstractmethod
    def run(self, **kwargs) -> Dict:
        """运行流水线"""
        pass

    def add_step(self, name: str, processor: Union[BaseProcessor, BaseModel]):
        """添加流水线步骤"""
        self.steps.append((name, processor))
        self._log_info(f"添加步骤: {name}")

    def get_results(self) -> Dict:
        """获取流水线结果"""
        return self.results.copy()

    def _log_info(self, message: str):
        """记录信息日志"""
        self.logger.info(f"[{self.__class__.__name__}] {message}")

    def _log_warning(self, message: str):
        """记录警告日志"""
        self.logger.warning(f"[{self.__class__.__name__}] {message}")

    def _log_error(self, message: str):
        """记录错误日志"""
        self.logger.error(f"[{self.__class__.__name__}] {message}")


class SingletonMeta(type):
    """
    单例模式元类：确保类只能有一个实例。
    """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


if __name__ == "__main__":
    class MyClass(metaclass=SingletonMeta):
        def __init__(self, value):
            self.value = value

    obj1 = MyClass(1)
    obj2 = MyClass(2)
    print(obj1.value)  # 输出: 1
    print(obj2.value)  # 输出: 1
    print(obj1 is obj2)