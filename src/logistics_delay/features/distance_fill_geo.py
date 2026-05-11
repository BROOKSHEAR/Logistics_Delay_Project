"""
基于地理邻近性的运输距离填补算法

针对印度倒三角地形，采用「经度→纬度」排序生成标准化城市对，
并通过扫描附近城市填补缺失的 TRANSPORTATION_DISTANCE_IN_KM 值。
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import math


def parse_lat_lon(lat_lon_str: str) -> Tuple[float, float]:
    """
    解析经纬度字符串，格式如 "28.6139, 77.2090"
    """
    try:
        lat_str, lon_str = str(lat_lon_str).split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return lat, lon
    except (ValueError, AttributeError):
        return None, None


def normalize_city_pair(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    标准化城市对：按照「经度→纬度」字典序排序

    针对印度倒三角地形：
    - 东西跨度大（68°E-97°E），南北跨度大（8°N-37°N）
    - 北部城市多且东西分布广，南部城市少且集中
    - 先按经度排序，将东、西区域自然分开；再按纬度排序

    Args:
        lat1, lon1: 城市1的纬度、经度
        lat2, lon2: 城市2的纬度、经度

    Returns:
        标准化后的城市对 ((lon1, lat1), (lon2, lat2))
    """
    city1 = (lon1, lat1)  # 经度在前！
    city2 = (lon2, lat2)

    # 按字典序排序（先比较经度，再比较纬度）
    if city1 < city2:
        return (city1, city2)
    else:
        return (city2, city1)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两个经纬度点之间的球面距离（公里）
    """
    R = 6371.0  # 地球半径，单位：公里

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c


class DistanceFiller:
    """
    基于地理邻近性的运输距离填补器
    """

    def __init__(self, df: pd.DataFrame):
        """
        初始化填充器

        Args:
            df: 包含以下列的DataFrame：
                - Org_lat_lon: 起始地经纬度字符串
                - Des_lat_lon: 目的地经纬度字符串
                - TRANSPORTATION_DISTANCE_IN_KM: 运输距离（可能包含NaN）
        """
        self.df = df.copy()

        # 解析经纬度
        self.df['orig_lat'], self.df['orig_lon'] = zip(*df['Org_lat_lon'].apply(parse_lat_lon))
        self.df['dest_lat'], self.df['dest_lon'] = zip(*df['Des_lat_lon'].apply(parse_lat_lon))

        # 提取城市名称（用于调试和展示）
        self.df['orig_city'] = df['Origin_Location'].astype(str).str.split(',').str[1].str.strip().str.upper().fillna('UNKNOWN')
        self.df['dest_city'] = df['Destination_Location'].astype(str).str.split(',').str[1].str.strip().str.upper().fillna('UNKNOWN')

        # 构建已知距离的城市对索引
        self._build_distance_index()

    def _build_distance_index(self):
        """构建已知距离的城市对索引"""
        # 获取有距离数据的行
        known_distances = self.df[self.df['TRANSPORTATION_DISTANCE_IN_KM'].notna()]

        # 先累计每个标准化城市对的距离和出现次数，最后再求真实平均值
        self.distance_dict = {}
        pair_stats = {}
        self.city_coords = {}  # 城市名 -> (纬度, 经度)

        for _, row in known_distances.iterrows():
            # 标准化城市对
            norm_pair = normalize_city_pair(
                row['orig_lat'], row['orig_lon'],
                row['dest_lat'], row['dest_lon']
            )

            # 存储距离统计量（如果有多个相同城市对，最终取总体平均值）
            distance = row['TRANSPORTATION_DISTANCE_IN_KM']
            if norm_pair in pair_stats:
                pair_stats[norm_pair]['sum'] += distance
                pair_stats[norm_pair]['count'] += 1
            else:
                pair_stats[norm_pair] = {'sum': distance, 'count': 1}

            # 存储城市坐标（用于附近搜索）
            if row['orig_city'] not in self.city_coords:
                self.city_coords[row['orig_city']] = (row['orig_lat'], row['orig_lon'])
            if row['dest_city'] not in self.city_coords:
                self.city_coords[row['dest_city']] = (row['dest_lat'], row['dest_lon'])

        self.distance_dict = {
            pair: stats['sum'] / stats['count']
            for pair, stats in pair_stats.items()
        }

    def _find_nearby_pair(self, target_pair: Tuple,
                          lat_threshold: float = 0.5,
                          lon_threshold: float = 1.0) -> Optional[float]:
        """
        在给定阈值内搜索附近的城市对

        Args:
            target_pair: 目标标准化城市对 ((lon1, lat1), (lon2, lat2))
            lat_threshold: 纬度搜索阈值（度）
            lon_threshold: 经度搜索阈值（度）

        Returns:
            找到的距离值，如果未找到则返回None
        """
        target_city1, target_city2 = target_pair

        best_distance = None
        best_diff = float('inf')

        for known_pair, known_dist in self.distance_dict.items():
            known_city1, known_city2 = known_pair

            # 计算两个城市对的坐标差异
            # 城市1差异
            lat_diff1 = abs(target_city1[1] - known_city1[1])  # 纬度在元组第二位
            lon_diff1 = abs(target_city1[0] - known_city1[0])  # 经度在元组第一位

            # 城市2差异
            lat_diff2 = abs(target_city2[1] - known_city2[1])
            lon_diff2 = abs(target_city2[0] - known_city2[0])

            # 检查是否在阈值范围内
            if (lat_diff1 <= lat_threshold and lon_diff1 <= lon_threshold and
                lat_diff2 <= lat_threshold and lon_diff2 <= lon_threshold):

                # 计算总差异（用于选择最佳匹配）
                total_diff = lat_diff1 + lon_diff1 + lat_diff2 + lon_diff2

                if total_diff < best_diff:
                    best_diff = total_diff
                    best_distance = known_dist

        return best_distance

    def fill_missing_distances(self, max_search_radius: float = 5.0, verbose: bool = False) -> pd.DataFrame:
        """
        填补缺失的运输距离

        Args:
            max_search_radius: 最大搜索半径（经纬度阈值逐步扩大到该值）
            verbose: 是否逐条打印填补记录，默认 False 只打印汇总

        Returns:
            填补后的DataFrame
        """
        result_df = self.df.copy()
        missing_indices = result_df[result_df['TRANSPORTATION_DISTANCE_IN_KM'].isna()].index

        print(f"发现 {len(missing_indices)} 条缺失运输距离的记录")

        filled_count = 0
        logged = 0  # 只打印前10条

        for idx in missing_indices:
            row = result_df.loc[idx]

            # 标准化城市对
            norm_pair = normalize_city_pair(
                row['orig_lat'], row['orig_lon'],
                row['dest_lat'], row['dest_lon']
            )

            # 尝试直接匹配
            if norm_pair in self.distance_dict:
                result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = self.distance_dict[norm_pair]
                filled_count += 1
                continue

            # 逐步扩大搜索范围
            found_distance = None
            for radius in np.arange(0.5, max_search_radius + 0.5, 0.5):
                lat_threshold = radius
                lon_threshold = radius * 2  # 经度阈值更大，因为印度东西跨度大

                found_distance = self._find_nearby_pair(norm_pair, lat_threshold, lon_threshold)
                if found_distance is not None:
                    result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = found_distance
                    filled_count += 1
                    if verbose and logged < 10:
                        print(f"  索引 {idx}: 在半径 {radius}° 内找到附近城市对，距离 ≈ {found_distance:.1f} km")
                        logged += 1
                    break

            if found_distance is None:
                if verbose and logged < 10:
                    print(f"  索引 {idx}: 在 {max_search_radius}° 半径内未找到匹配，将使用全局中位数")
                    logged += 1
                # 使用全局中位数作为后备
                global_median = result_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
                result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = global_median

        print(f"\n填补完成: {filled_count} 条记录通过地理邻近性填补")
        print(f"剩余 {len(missing_indices) - filled_count} 条记录使用中位数填补")

        # 验证填补结果
        still_missing = result_df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
        if still_missing > 0:
            print(f"警告: 仍有 {still_missing} 条记录缺失，使用中位数填补")
            median_val = result_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
            result_df['TRANSPORTATION_DISTANCE_IN_KM'] = result_df['TRANSPORTATION_DISTANCE_IN_KM'].fillna(median_val)

        return result_df

    def analyze_distance_distribution(self):
        """分析距离分布，生成统计信息"""
        original_missing = self.df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
        total_rows = len(self.df)

        print("=== 运输距离分析 ===")
        print(f"总记录数: {total_rows}")
        print(f"原始缺失记录: {original_missing} ({original_missing/total_rows*100:.2f}%)")

        if original_missing > 0:
            # 按城市对统计
            missing_df = self.df[self.df['TRANSPORTATION_DISTANCE_IN_KM'].isna()]
            print(f"\n缺失距离的城市对数量: {len(missing_df)}")

            # 检查是否有重复城市对
            city_pairs = []
            for _, row in missing_df.iterrows():
                pair = normalize_city_pair(
                    row['orig_lat'], row['orig_lon'],
                    row['dest_lat'], row['dest_lon']
                )
                city_pairs.append(pair)

            unique_pairs = set(city_pairs)
            print(f"唯一城市对数量: {len(unique_pairs)}")

        # 距离分布统计
        if original_missing < total_rows:
            known_distances = self.df['TRANSPORTATION_DISTANCE_IN_KM'].dropna()
            print(f"\n已知距离统计:")
            print(f"  最小值: {known_distances.min():.1f} km")
            print(f"  中位数: {known_distances.median():.1f} km")
            print(f"  平均值: {known_distances.mean():.1f} km")
            print(f"  最大值: {known_distances.max():.1f} km")
            print(f"  标准差: {known_distances.std():.1f} km")


def main():
    """示例用法"""
    import warnings
    warnings.filterwarnings('ignore')

    from logistics_delay.utils.paths import DATA_RAW

    # 读取数据
    print("正在读取数据...")
    df = pd.read_excel(DATA_RAW)

    # 创建填充器
    filler = DistanceFiller(df)

    # 分析原始数据
    filler.analyze_distance_distribution()

    # 填补缺失距离
    print("\n开始填补缺失距离...")
    filled_df = filler.fill_missing_distances(max_search_radius=3.0)

    # 验证填补效果
    print("\n=== 填补效果验证 ===")
    still_missing = filled_df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
    print(f"填补后缺失记录: {still_missing} ({still_missing/len(filled_df)*100:.2f}%)")

    # 与原始中位数方法比较
    original_median = df['TRANSPORTATION_DISTANCE_IN_KM'].median()
    filled_median = filled_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
    print(f"原始中位数: {original_median:.1f} km")
    print(f"填补后中位数: {filled_median:.1f} km")
    print(f"变化: {((filled_median - original_median) / original_median * 100):+.2f}%")


if __name__ == "__main__":
    main()
