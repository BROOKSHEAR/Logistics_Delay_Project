"""
测试经纬度排序逻辑和地理填补算法
不依赖实际数据文件，只验证核心算法
"""
import pandas as pd
import sys

from logistics_delay.features.distance_fill_geo import DistanceFiller, parse_lat_lon, normalize_city_pair, haversine_distance


def test_parse_lat_lon():
    """测试经纬度解析"""
    print("测试经纬度解析...")

    test_cases = [
        ("28.6139, 77.2090", (28.6139, 77.2090)),  # 新德里
        ("19.0760, 72.8777", (19.0760, 72.8777)),  # 孟买
        ("13.0827, 80.2707", (13.0827, 80.2707)),  # 金奈
        ("12.9716, 77.5946", (12.9716, 77.5946)),  # 班加罗尔
        ("22.5726, 88.3639", (22.5726, 88.3639)),  # 加尔各答
    ]

    for input_str, expected in test_cases:
        lat, lon = parse_lat_lon(input_str)
        assert abs(lat - expected[0]) < 0.0001, f"纬度解析错误: {input_str}"
        assert abs(lon - expected[1]) < 0.0001, f"经度解析错误: {input_str}"
        print(f"  ✓ {input_str} -> ({lat:.4f}, {lon:.4f})")

    # 测试无效输入
    invalid_cases = ["", "invalid", "28.6139", "28.6139,abc"]
    for invalid in invalid_cases:
        lat, lon = parse_lat_lon(invalid)
        assert lat is None and lon is None, f"无效输入处理错误: {invalid}"

    print("✓ 所有解析测试通过\n")


def test_normalize_city_pair():
    """测试城市对标准化排序（经度→纬度）"""
    print("测试城市对标准化...")

    # 印度主要城市坐标
    delhi = (28.6139, 77.2090)    # 纬度28.6, 经度77.2 (北部)
    mumbai = (19.0760, 72.8777)   # 纬度19.1, 经度72.9 (西部)
    chennai = (13.0827, 80.2707)  # 纬度13.1, 经度80.3 (东南部)
    kolkata = (22.5726, 88.3639)  # 纬度22.6, 经度88.4 (东北部)

    # 测试用例：不同组合
    test_cases = [
        # (城市1, 城市2, 预期排序后的第一个城市)
        (delhi, mumbai, mumbai),   # 孟买经度72.9 < 德里经度77.2
        (delhi, chennai, delhi), # 金奈经度80.3 > 德里经度77.2，但先比较经度
        (mumbai, kolkata, mumbai), # 孟买经度72.9 < 加尔各答经度88.4
        (chennai, kolkata, chennai), # 金奈经度80.3 < 加尔各答经度88.4
    ]

    for (lat1, lon1), (lat2, lon2), expected_first in test_cases:
        norm_pair = normalize_city_pair(lat1, lon1, lat2, lon2)
        first_city, second_city = norm_pair

        # 验证排序正确性
        assert first_city <= second_city, f"排序错误: {first_city} > {second_city}"

        # 验证第一个城市是否符合预期
        expected_lat, expected_lon = expected_first
        assert abs(first_city[0] - expected_lon) < 0.0001, f"第一个城市经度错误"
        assert abs(first_city[1] - expected_lat) < 0.0001, f"第一个城市纬度错误"

        print(f"  ✓ ({lat1:.1f},{lon1:.1f}) & ({lat2:.1f},{lon2:.1f}) -> "
              f"({first_city[0]:.1f},{first_city[1]:.1f}), ({second_city[0]:.1f},{second_city[1]:.1f})")

    print("✓ 所有标准化测试通过\n")


def test_haversine():
    """测试距离计算"""
    print("测试球面距离计算...")

    # 德里到孟买的实际距离约1150公里
    delhi = (28.6139, 77.2090)
    mumbai = (19.0760, 72.8777)

    distance = haversine_distance(*delhi, *mumbai)
    print(f"  德里 → 孟买: {distance:.1f} km (预期约1150 km)")

    # 简单验证：正距离，非零
    assert distance > 0, "距离应为正数"
    assert distance < 2000, "德里到孟买距离应小于2000km"

    # 相同点距离应为0
    zero_distance = haversine_distance(*delhi, *delhi)
    assert abs(zero_distance) < 0.001, "相同点距离应为0"

    print("✓ 距离计算测试通过\n")


def test_india_specific_logic():
    """测试印度特定逻辑：经度优先排序的合理性"""
    print("测试印度地形特定逻辑...")

    # 模拟印度北部和南部城市
    # 北部城市：经度范围大
    north_west = (30.0, 70.0)   # 西部边界
    north_east = (30.0, 90.0)   # 东部边界

    # 南部城市：相对集中
    south_west = (10.0, 75.0)
    south_east = (10.0, 80.0)

    # 测试1：北部东西城市对
    pair1 = normalize_city_pair(*north_west, *north_east)
    print(f"  北部东西城市对排序: {pair1}")
    # 西部城市应在前面（经度70 < 90）
    assert pair1[0][0] == 70.0, "西部城市应在前"

    # 测试2：南部城市对
    pair2 = normalize_city_pair(*south_west, *south_east)
    print(f"  南部东西城市对排序: {pair2}")
    # 同样，经度小的在前
    assert pair2[0][0] == 75.0, "经度较小的南部城市应在前"

    # 测试3：跨南北城市对
    pair3 = normalize_city_pair(*north_west, *south_east)
    print(f"  跨南北城市对排序: {pair3}")
    # 北部城市纬度更高，但排序应基于经度
    # 北部西部(70.0) vs 南部东部(80.0) -> 70.0在前
    assert pair3[0][0] == 70.0, "经度决定排序顺序"

    print("✓ 印度地形逻辑测试通过\n")


def test_duplicate_pair_uses_true_mean():
    """Test that repeated city pairs use the arithmetic mean across all rows."""
    print("Testing duplicate-pair averaging...")

    df = pd.DataFrame({
        'Org_lat_lon': ['10.0, 20.0', '10.0, 20.0', '10.0, 20.0'],
        'Des_lat_lon': ['30.0, 40.0', '30.0, 40.0', '30.0, 40.0'],
        'Origin_Location': ['X, A', 'X, A', 'X, A'],
        'Destination_Location': ['Y, B', 'Y, B', 'Y, B'],
        'TRANSPORTATION_DISTANCE_IN_KM': [100.0, 200.0, 500.0],
    })

    filler = DistanceFiller(df)
    norm_pair = normalize_city_pair(10.0, 20.0, 30.0, 40.0)
    actual = filler.distance_dict[norm_pair]
    expected = (100.0 + 200.0 + 500.0) / 3

    assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"
    print(f"  OK: arithmetic mean = {actual:.6f} km")


def main():
    """运行所有测试"""
    print("=" * 60)
    print("地理填补算法单元测试")
    print("=" * 60)

    try:
        test_parse_lat_lon()
        test_normalize_city_pair()
        test_haversine()
        test_india_specific_logic()
        test_duplicate_pair_uses_true_mean()

        print("=" * 60)
        print("所有测试通过！")
        print("地理填补算法逻辑正确，可以应用于实际数据。")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
