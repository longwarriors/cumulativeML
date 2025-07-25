"""
提供统一的配置管理功能，支持YAML配置文件的加载、验证和管理。
"""
import os, yaml, logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from .base import SingletonMeta


class ConfigManager(metaclass=SingletonMeta):
    """"配置管理器，使用单例模式确保全局配置的一致性。"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._configs = {}
        self._config_paths = {}
        self._default_config_dir = Path(__file__).parent / 'config'

    def load_config(self, config_path: Union[str, Path], config_name: Optional[str] = None) -> Dict[str, Any]:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        if config_name is None:
            config_name = config_path.stem

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)

            self._configs[config_name] = config
            self._config_paths[config_name] = str(config_path)
            self.logger.info(f"成功加载配置: {config_name} from {config_path}")
            return config
        except yaml.YAMLError as e:
            self.logger.error(f"YAML解析错误: {e}")
            raise
        except Exception as e:
            self.logger.error(f"加载配置文件失败: {e}")
            raise

    def load_all_configs(self, config_dir: Optional[str] = None) -> Dict[str, str]:
        if config_dir is None:
            config_dir = self._default_config_dir
        else:
            config_dir = Path(config_dir)
        if not config_dir.exists() or not config_dir.is_dir():
            self.logger.warning(f"配置目录不存在: {config_dir}")
            return {}

        configs = {}
        for file in config_dir.glob('*.yaml'):
            try:
                name = file.stem
                configs[name] = self.load_config(file, name)
            except Exception as e:
                self.logger.error(f"加载配置文件 {file} 失败: {e}")

        return configs

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        if config_name not in self._configs:
            default_path = self._default_config_dir / f"{config_name}.yaml"
            if default_path.exists():
                return self.load_config(default_path, config_name)
            else:
                raise KeyError(f"配置不存在: {config_name}")
        return self._configs[config_name].copy()

    def get_data_config(self) -> Dict[str, Any]:
        """获取数据配置"""
        return self.get_config("data_config")

    def get_feature_config(self) -> Dict[str, Any]:
        return self.get_config("feature_config")

    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return self.get_config("model_config")

    def get_pipeline_config(self) -> Dict[str, Any]:
        """获取管道配置"""
        return self.get_config("pipeline_config")

    def update_config(self, config_name: str, new_config: Dict[str, Any]):
        """更新指定配置"""
        if config_name in self._configs:
            self._configs[config_name].update(new_config)
            self.logger.info(f"更新配置: {config_name}")
        else:
            raise KeyError(f"配置不存在: {config_name}")

    def save_config(self, config_name: str, save_path: Optional[Union[str, Path]] = None):
        """保存指定配置到文件"""
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")

        if save_path is None:
            save_path = self._default_config_dir / f"{config_name}.yaml"
        else:
            save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            config = self._configs[config_name]
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            self._config_paths[config_name] = str(save_path)
            self.logger.info(f"成功保存配置: {config_name} to {save_path}")
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
            raise

    def validate_config(self, config_name: str, schema: Dict[str, Any]) -> bool:
        """
        验证配置是否符合指定的schema。
        :param config_name: 配置名称。
        :param schema: 验证schema，通常是一个字典，定义了配置的结构和类型。
        :return: 如果配置符合schema返回True，否则返回False。
        """
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")

        config = self._configs[config_name]
        for key, expected_type in schema.items():
            if key not in config:
                self.logger.warning(f"配置缺少键: {key}")
                return False
            if not isinstance(config[key], expected_type):
                self.logger.error(f"配置键 {key} 的类型不匹配: 期望 {expected_type}, 实际 {type(config[key])}")
                return False

        self.logger.info(f"配置 {config_name} 验证通过")
        return True

    def get_nested_value(self, config_name: str, keys: str, default: Any = None) -> Any:
        """
        获取嵌套配置值。
        :param config_name: 配置名称。
        :param keys: 键路径，用点分隔，如 "model.xgboost.max_depth"。
        :param default: 如果未找到值，返回的默认值。
        :return: 嵌套值或默认值。
        """
        config = self.get_config(config_name)
        keys = keys.split(".")
        try:
            for key in keys:
                value = config[key]
            return value
        except (KeyError, TypeError):
            self.logger.warning(f"未找到配置键: {keys}，返回默认值: {default}")
            return default

    def set_nested_value(self, config_name: str, keys: str, value: Any):
        """
        设置嵌套配置值。
        :param config_name: 配置名称。
        :param keys: 键路径，用点分隔，如 "model.xgboost.max_depth"。
        :param value: 要设置的值。
        """
        if config_name not in self._configs:
            raise KeyError(f"配置不存在: {config_name}")

        config = self.get_config(config_name)
        keys = keys.split(".")
        for key in keys[:-1]: # 导航到最后一级的父级
            if key not in config:
                config[key] = {}
            config = config[key]

        # 设置最终值 - 注意：这里直接修改的是原始配置的引用，无需update_config
        old_value = config.get(keys[-1])
        config[keys[-1]] = value
        self.logger.info(f"设置配置值: {config_name}.{keys} = {value}")
        if old_value != value:
            self.logger.debug(f"原值: {old_value}")

    def list_configs(self) -> List[str]:
        """列出所有已加载的配置名称"""
        return list(self._configs.keys())

    def clear_configs(self):
        """清除所有已加载的配置"""
        self._configs.clear()
        self._config_paths.clear()
        self.logger.info("已清除所有配置")

    def get_config_info(self) -> Dict[str, str]:
        """获取配置信息"""
        return {name: self._config_paths.get(name, "内存中") for name in self._configs.keys()}