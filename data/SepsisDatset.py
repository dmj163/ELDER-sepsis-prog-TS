# 自定义数据集类
import torch
from torch.utils.data import Dataset

# 通过数据和标签创建数据集
# 通过时间戳（时间索引）从全量时间序列中提取每个时间窗口输入数据
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, index_label_file, time_series_data_file, drop_columns=None):
        """
        Args:
            index_label_file (str): 包含索引和标签的 pickle 文件路径。
            time_series_data_file (str): 包含完整时间序列数据的文件路径 (e.g., HDF5)。
            transform (callable, optional): 可选的转换函数。
        """
        # 在初始化时加载所有时间序列数据时间索引和标签到内存
        loaded_data = pd.read_pickle(index_label_file)
        self.labels = loaded_data['labels']  # numpy数组
        self.patient_ids = loaded_data['patient_ids']  # 列表
        self.timestamps = loaded_data['timestamps']  # 列表


        self.time_series_file_path = time_series_data_file
        self.drop_columns = drop_columns

        # 在初始化时加载所有时间序列数据到内存
        print("Loading time series data into memory...")
        self.time_series_data = self._load_all_time_series_dat(time_series_data_file)
        print(f"Loaded data for {len(self.time_series_data)} patients")

    def _load_all_time_series_dat(self, time_series_file_path):
        """根据 ICU ID 从主数据文件加载其完整时间序列"""
        try:
            # 读取 feather 文件
            data = pd.read_feather(self.time_series_file_path)


            # 删除不需要的列（请根据实际需要修改列名列表）
            columns_to_drop = self.drop_columns  # 替换为实际要删除的列名
            available_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
            if available_columns_to_drop:
                data = data.drop(columns=available_columns_to_drop, errors='ignore')

            # 按 patient_id 分组并转换为字典存储
            grouped_data = {}
            for stay_id, group in data.groupby('stay_id'):
                # 删除用于分组的 stay_id 列
                if 'stay_id' in group.columns:
                    patient_data = group.drop(columns=['stay_id']).values
                else:
                    patient_data = group.values
                grouped_data[stay_id] = patient_data

            return grouped_data

        except FileNotFoundError:
            print(f"Warning: Feather file not found at {time_series_file_path}")
            return {}
        except Exception as e:
            print(f"Warning: Error loading time series data: {e}")
            return {}

    def __len__(self):
        """返回数据集大小"""
        return len(self.patient_ids)
    def _extract_window_and_process(self, full_series, window_end_time):
        """
        从完整序列中提取指定时间窗口的数据，并进行预处理（对齐、填充、掩码等）。
        Args:
            full_series (np.ndarray): 完整 ICU 记录的时间序列数据 (T_full, num_features_raw)。
            window_end_time (pd.Timestamp): 窗口结束时间。
        Returns:
            tuple: (processed_window_data, mask, seq_len) 或类似结构。
        """
        # --- 1. 时间窗口切片 ---
        # 如果存储了完整时间戳，使用它来精确切片
        # 固定开始点为0，结束点为window_end_time
        try:
            # 固定开始点为0，结束点为window_end_time
            end_idx = min(int(window_end_time), len(full_series))
            if end_idx <= 0:
                end_idx = len(full_series)  # 确保至少有一个时间点
            windowed_data = full_series[1:end_idx]
        except (IndexError, TypeError) as e:
            print(f"Warning: Error when slicing data. window_end_time: {window_end_time}, "
                  f"data length: {len(full_series)}. Error: {e}")
            # 返回默认值
            feature_dim = full_series.shape[1] if len(full_series.shape) > 1 else 1
            return np.zeros((180, feature_dim), dtype=np.float32)

        if windowed_data.size == 0:
            print(f"Warning: Window [0, {window_end_time}] is empty for this ICU stay.")
            # 返回默认值
            feature_dim = full_series.shape[1] if len(full_series.shape) > 1 else 1
            return np.zeros((180, feature_dim), dtype=np.float32)

        # --- 2. 标准化 ---
        # 使用Z-score标准化: (x - mean) / std
        # 对每个特征维度独立进行标准化
        try:
            mean = np.mean(windowed_data, axis=0)
            std = np.std(windowed_data, axis=0)

            # 避免除以0的情况
            std = np.where(std == 0, 1.0, std)
            windowed_data = (windowed_data - mean) / std
        except Exception as e:
            print(f"Warning: Error during normalization: {e}")
            # 如果标准化失败，继续使用原始数据

        # 返回处理后的数据、掩码、实际长度等
        # return processed_data.astype(np.float32), mask.astype(np.float32), seq_len
        # 简化示例，只返回数据
        return windowed_data.astype(np.float32) #, mask, seq_len # 根据需要调整返回值


    def __getitem__(self, idx):
        """
        获取单个样本。
        Args:
            idx (int): 样本索引。
        Returns:
            tuple: (data, targets) 其中 data 是处理后的时间窗口张量，targets 是标签字典。
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 1. 从索引文件获取当前样本的信息
        patient_id = self.patient_ids[idx]
        window_end_time = self.timestamps[idx]

        # 获取标签（labels是numpy数组）
        label_data = self.labels[idx]

        # 2. 从内存中获取该患者的完整时间序列
        if patient_id in self.time_series_data:
            full_time_series_data = self.time_series_data[patient_id]
        else:
            print(f"Warning: No data found for ICU stay ID {patient_id}")
            # 返回默认值
            feature_dim = len([col for col in self.time_series_data[next(iter(self.time_series_data))].columns]) \
                if self.time_series_data else 10  # 假设默认特征数
            full_time_series_data = np.zeros((100, feature_dim))  # 默认数据

        # 3. 根据窗口时间索引从完整序列中提取窗口并预处理
        processed_window_data = self._extract_window_and_process(
            full_time_series_data, window_end_time  # 固定开始时间为0
            )

        # 4. 转换为 PyTorch Tensor
        window_tensor = torch.from_numpy(processed_window_data)

        # 5. 获取标签
        targets = {
            'death': torch.tensor(label_data[0], dtype=torch.float32),  # 死亡标签
            'icu_los_reg': torch.tensor(label_data[1], dtype=torch.float32),  # ICU LOS回归标签
            'icu_los_class': torch.tensor(label_data[2], dtype=torch.long),  # ICU LOS分类标签
            'hosp_los_reg': torch.tensor(label_data[3], dtype=torch.float32),  # 医院LOS回归标签
            'hosp_los_class': torch.tensor(label_data[4], dtype=torch.long),  # 医院LOS分类标签
            'sofa': torch.tensor(label_data[5], dtype=torch.float32),  # SOFA分数标签
            # 根据需要添加其他标签
        }


        # 7. 返回数据和标签
        # return (window_tensor, mask_tensor, seq_len_tensor), targets # 如果返回多个
        return window_tensor, targets


def collate_fn(batch):
    """
    自定义collate函数，用于在batch级别进行序列补齐
    Args:
        batch: 一个batch的数据，格式为[(data1, targets1), (data2, targets2), ...]
    Returns:
        tuple: (padded_data, targets_dict)
    """
    # 分离数据和标签
    data_list, targets_list = zip(*batch)

    # 获取batch中序列的最大长度
    max_seq_len = max([data.shape[0] for data in data_list])
    feature_dim = data_list[0].shape[1]
    batch_size = len(data_list)

    # 创建补齐后的数据张量
    padded_data = torch.zeros((batch_size, max_seq_len, feature_dim), dtype=torch.float32)

    # 对每个序列进行补齐（在序列开头补齐）
    for i, data in enumerate(data_list):
        seq_len = data.shape[0]
        padded_data[i, -seq_len:, :] = torch.from_numpy(data)  # 在末尾对齐

    # 合并标签
    targets_dict = {}
    for key in targets_list[0].keys():
        targets_dict[key] = torch.stack([targets[key] for targets in targets_list])

    return padded_data, targets_dict
