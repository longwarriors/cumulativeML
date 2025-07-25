"""
数据加载模块：负责加载项目的所有数据文件，支持数据验证、缓存和内存优化。
"""
import os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from core.base import BaseProcessor
from core.utils import reduce_memory_usage, get_memory_usage, save_object, load_object


class DataLoader(BaseProcessor):
    def __init__(self, config: Optional[Dict] = None, logger: Optional[Any] = None):
        super().__init__(config, logger)
        self.data = {}
        self.data_info = {}

    def fit(self, data: pd.DataFrame = None, **kwargs) -> 'DataLoader':
        """
        拟合数据加载器，主要用于设置配置。
        :param data: 通常为 None，因为 DataLoader 不需要拟合数据，保持接口一致性。
        :param kwargs: 其他可选参数。
        :return self: 返回自身以支持链式调用。
        """
        self.is_fitted = True
        return self

    def transform(self, data: pd.DataFrame = None, use_cache: Optional[bool] = None, **kwargs) -> Dict[str, pd.DataFrame]:
        """
        转换数据，加载并返回所有数据文件。
        :param data: 通常为 None，因为 DataLoader 不需要转换数据，保持接口一致性。
        :param use_cache: 是否使用缓存，默认为 False，表示不使用缓存。
        :param kwargs: 其他可选参数。
        :return: 包含所有加载数据的字典。
        """
        self._log_info("开始加载所有数据集")
        # 获取配置
        data_paths = self.config.get('data_files', {})
        caching_config = self.config.get('caching', {})
        memory_config = self.config.get('memory_optimization', {})
        if use_cache is None:
            use_cache = caching_config.get('enable', False)

        # 检查缓存
        if use_cache:
            cached_data = self._load_from_cache()

    def _load_from_cache(self) -> Optional[Dict[str, pd.DataFrame]]:
        """
        从缓存中加载数据。
        :return: 如果缓存存在，返回缓存的数据字典，否则返回 None。
        """
        try:
            cache_config = self.config.get('caching', {})
            cache_dir = Path(cache_config.get('cache_dir', "./cache/data"))
