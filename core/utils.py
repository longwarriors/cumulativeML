"""
通用工具函数模块：提供项目中常用的工具函数和辅助功能。
"""
import os, sys, json, gc, psutil, pickle, joblib, warnings
from typing import Any, Dict, List, Union, Optional, Tuple
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd


def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    减少 DataFrame 的内存使用量。
    :param df: 输入的 pandas DataFrame。
    :param verbose: 是否打印内存使用信息。
    :return: 内存优化后的 DataFrame。
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        # 保护目标变量不被类型转换，可以避免精度损失
        if col == 'TARGET':
            continue
        col_type = df[col].dtype
        if pd.api.types.is_string_dtype(col_type):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # 对于浮点数，至少使用float32避免精度问题
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        elif col_type == 'object':
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(f"内存使用量从 {start_mem:.2f} MB 减少到 {end_mem:.2f} MB "
              f'(减少了 {100 * (start_mem - end_mem) / start_mem:.1f}%)')
    return df


def get_memory_usage() -> Dict[str, Union[int, float]]:
    """
    获取当前进程的内存使用情况。
    :return: 包含内存使用信息的字典。
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / (1024 ** 2),  # 物理内存大小，单位 MB
        'vms': mem_info.vms / (1024 ** 2),  # 虚拟内存大小，单位 MB
        'percent': process.memory_percent()  # 内存使用百分比
    }


def clean_memory():
    """
    清理内存，释放未使用的对象和缓存。
    """
    gc.collect()
    if 'torch' in sys.modules:
        import torch
        torch.cuda.empty_cache()
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        try:
            tf.keras.backend.clear_session()  # TensorFlow 2.x
        except AttributeError:
            try:
                tf.compat.v1.reset_default_graph()  # TensorFlow 1.x
            except AttributeError:
                pass


def save_object(obj: Any, filepath: Union[str, Path], method: str = 'joblib'):
    """
    保存对象到文件。
    :param obj: 要保存的对象。
    :param filepath: 文件路径。
    :param method: 保存方法，'pickle' 或 'joblib'。
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    if method == 'joblib':
        joblib.dump(obj, filepath)
    elif method == 'pickle':
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    else:
        raise ValueError(f"不支持的保存方法: {method}")


def load_object(filepath: Union[str, Path], method: str = 'joblib') -> Any:
    """
    从文件加载对象。
    :param filepath: 文件路径。
    :param method: 加载方法，'pickle' 或 'joblib'。
    :return: 加载的对象。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    if method == 'joblib':
        return joblib.load(filepath)
    elif method == 'pickle':
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"不支持的加载方法: {method}")


def save_json(data: Dict[str, Any], filepath: Union[str, Path], indent: Optional[int] = 4):
    """
    保存字典到 JSON 文件。
    :param data: 要保存的数据。
    :param filepath: 文件路径。
    :param indent: 缩进级别，默认为 4。
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    从 JSON 文件加载字典。
    :param filepath: 文件路径。
    :return: 加载的字典。
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_directory(path: Union[str, Path], exist_ok: bool = True):
    """
    创建目录，如果目录已存在则不做任何操作。
    :param path: 目录路径。
    :param exist_ok: 如果目录已存在，是否忽略错误。
    """
    Path(path).mkdir(parents=True, exist_ok=exist_ok)


def get_timestamp(style: str = "%Y%m%d_%H%M%S") -> str:
    """
    获取当前时间戳，格式化为指定的字符串。
    :param style: 时间戳格式，默认为 "%Y%m%d_%H%M%S"。
    :return: 格式化后的时间戳字符串。
    """
    return datetime.now().strftime(style)


def safe_divide(numerator: Union[float, np.ndarray],
                denominator: Union[float, np.ndarray],
                fill_value: float = 0.0) -> Union[float, np.ndarray]:
    """
    安全除法函数，处理除数为零的情况。
    :param numerator: 分子。
    :param denominator: 分母。
    :param fill_value: 除零时的填充值。
    :return: 除法结果
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = np.divide(numerator, denominator)
    if isinstance(result, np.ndarray):
        result = np.where(np.isfinite(result), result, fill_value)
    else:
        if not np.isfinite(result):
            result = fill_value
    return result


def clip_outliers(series: pd.Series, method: str = 'iqr', factor: float = 1.5) -> pd.Series:
    """
    剪切异常值，将超出指定范围的值替换为边界值。
    :param series: 输入的 pandas Series。
    :param method: 异常值检测方法，'iqr' 或 'zscore'。
    :param factor: 异常值检测的系数，默认为 1.5。
    :return: 剪切后的 pandas Series。
    """
    if method == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        return series.clip(lower=lower_bound, upper=upper_bound)
    elif method == 'zscore':
        mean = series.mean()
        std = series.std()
        lower_bound = mean - factor * std
        upper_bound = mean + factor * std
        return series.clip(lower=lower_bound, upper=upper_bound)
    else:
        raise ValueError(f"不支持的异常值检测方法: {method}")


def get_feature_names_from_pipeline(pipeline: Any, feature_names: List[str]) -> List[str]:
    """
    从sklearn pipeline中获取特征名称，适用于包含特征选择或转换的管道。
    :param pipeline: sklearn管道对象或转换器。
    :param feature_names: 原始特征名称列表。
    :return: 处理后的特征名称列表。
    """
    try:
        if hasattr(pipeline, 'get_feature_names_out'):
            return pipeline.get_feature_names_out(feature_names).tolist()
        elif hasattr(pipeline, 'get_feature_names'):  # 兼容旧版本的sklearn
            return pipeline.get_feature_names(feature_names).tolist()
        else:
            return feature_names
    except:
        return feature_names  # 如果获取特征名称失败，返回原始特征名称


def validate_dataframe(df: pd.DataFrame,
                       required_columns: Optional[List[str]] = None,
                       check_missing: bool = True,
                       check_duplicates: bool = True) -> Dict[str, Any]:
    """
    验证 DataFrame 的结构和内容。
    :param df: 待验证的 pandas DataFrame。
    :param required_columns: 必须的列名列表，如果提供，则检查这些列是否存在。
    :param check_missing: 是否检查缺失值，默认为 True。
    :param check_duplicates: 是否检查重复行，默认为 True。
    :return results: 验证结果的字典。
    """
    results = {'valid': True, 'errors': [], 'warnings': [], 'info': {}}

    # 检查必需的列
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            results['valid'] = False
            results['errors'].append(f"缺少必需的列: {', '.join(missing_columns)}")

    # 检查缺失值
    if check_missing:
        missing_info = df.isnull().sum()
        missing_columns = missing_info[missing_info > 0]
        if missing_columns:
            results['warnings'].append(f"存在缺失值的列: {missing_columns.to_dict()}")
            results['info']['missing_values'] = missing_columns.to_dict()

    # 检查重复行
    if check_duplicates:
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            results['warnings'].append(f"存在 {duplicate_rows} 行重复数据")
            results['info']['duplicates'] = duplicate_rows

    # 记录 DataFrame 的基本信息
    results['info']['shape'] = df.shape
    results['info']['dtypes'] = df.dtypes.value_counts().to_dict()
    results['info']['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024 ** 2

    return results


def timer(func):
    """
    装饰器：用于测量函数执行时间。
    :param func: 被装饰的函数。
    :return: 包装后的函数。
    """

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"函数 {func.__name__} 执行时间: {duration:.4f} 秒")
        return result

    return wrapper
