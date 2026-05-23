"""
Test coordinate sorting and geographic imputation logic
Does not depend on actual data files, only tests core algorithm
"""
import pandas as pd
import sys

from logistics_delay.features.distance_fill_geo import DistanceFiller, parse_lat_lon, normalize_city_pair, haversine_distance


def test_parse_lat_lon():
    """Test coordinate parsing"""
    print("Testing coordinate parsing...")

    test_cases = [
        ("28.6139, 77.2090", (28.6139, 77.2090)),  # New Delhi
        ("19.0760, 72.8777", (19.0760, 72.8777)),  # Mumbai
        ("13.0827, 80.2707", (13.0827, 80.2707)),  # Chennai
        ("12.9716, 77.5946", (12.9716, 77.5946)),  # Bangalore
        ("22.5726, 88.3639", (22.5726, 88.3639)),  # Kolkata
    ]

    for input_str, expected in test_cases:
        lat, lon = parse_lat_lon(input_str)
        assert abs(lat - expected[0]) < 0.0001, f"Latitude parsing error: {input_str}"
        assert abs(lon - expected[1]) < 0.0001, f"Longitude parsing error: {input_str}"
        print(f"  ✓ {input_str} -> ({lat:.4f}, {lon:.4f})")

    # Test invalid input
    invalid_cases = ["", "invalid", "28.6139", "28.6139,abc"]
    for invalid in invalid_cases:
        lat, lon = parse_lat_lon(invalid)
        assert lat is None and lon is None, f"Invalid input handling error: {invalid}"

    print("? All parsing tests passed\n")


def test_normalize_city_pair():
    """Test city pair normalization (lon→lat)"""
    print("Testing city pair normalization...")

    # Major Indian city coordinates
    delhi = (28.6139, 77.2090)    # lat 28.6, lon 77.2 (north)
    mumbai = (19.0760, 72.8777)   # lat 19.1, lon 72.9 (west)
    chennai = (13.0827, 80.2707)  # lat 13.1, lon 80.3 (southeast)
    kolkata = (22.5726, 88.3639)  # lat 22.6, lon 88.4 (northeast)

    # Test cases: different combinations
    test_cases = [
        # (city1, city2, expected first city after sort)
        (delhi, mumbai, mumbai),   # Mumbai lon 72.9 < Delhi lon 77.2
        (delhi, chennai, delhi), # Chennai lon 80.3 > Delhi lon 77.2, compare lon first
        (mumbai, kolkata, mumbai), # Mumbai lon 72.9 < Kolkata lon 88.4
        (chennai, kolkata, chennai), # Chennai lon 80.3 < Kolkata lon 88.4
    ]

    for (lat1, lon1), (lat2, lon2), expected_first in test_cases:
        norm_pair = normalize_city_pair(lat1, lon1, lat2, lon2)
        first_city, second_city = norm_pair

        # Verify sort correctness
        assert first_city <= second_city, f"Sort error: {first_city} > {second_city}"

        # Verify first city matches expectation
        expected_lat, expected_lon = expected_first
        assert abs(first_city[0] - expected_lon) < 0.0001, f"First city longitude error"
        assert abs(first_city[1] - expected_lat) < 0.0001, f"First city latitude error"

        print(f"  ✓ ({lat1:.1f},{lon1:.1f}) & ({lat2:.1f},{lon2:.1f}) -> "
              f"({first_city[0]:.1f},{first_city[1]:.1f}), ({second_city[0]:.1f},{second_city[1]:.1f})")

    print("? All normalization tests passed\n")


def test_haversine():
    """Test distance calculation"""
    print("Testing spherical distance...")

    # Delhi to Mumbai actual distance ~1150 km
    delhi = (28.6139, 77.2090)
    mumbai = (19.0760, 72.8777)

    distance = haversine_distance(*delhi, *mumbai)
    print(f"  Delhi → Mumbai: {distance:.1f} km (expected ~1150 km)")

    # Basic: positive distance, non-zero
    assert distance > 0, "distance should be positive"
    assert distance < 2000, "Delhi to Mumbai should be < 2000km"

    # Same point distance should be 0
    zero_distance = haversine_distance(*delhi, *delhi)
    assert abs(zero_distance) < 0.001, "same point distance should be 0"

    print("? Distance calculation test passed\n")


def test_india_specific_logic():
    """Test India-specific logic: longitude-first sorting rationale"""
    print("Testing India terrain-specific logic...")

    # Simulate northern and southern Indian cities
    # Northern cities: wide longitude range
    north_west = (30.0, 70.0)   # western boundary
    north_east = (30.0, 90.0)   # eastern boundary

    # Southern cities: relatively clustered
    south_west = (10.0, 75.0)
    south_east = (10.0, 80.0)

    # Test 1: Northern E-W city pair
    pair1 = normalize_city_pair(*north_west, *north_east)
    print(f"  Northern E-W city pair sorted: {pair1}")
    # Western city should come first (lon 70 < 90)
    assert pair1[0][0] == 70.0, "western city should come first"

    # Test 2: Southern city pair
    pair2 = normalize_city_pair(*south_west, *south_east)
    print(f"  Southern E-W city pair sorted: {pair2}")
    # Same: smaller longitude first
    assert pair2[0][0] == 75.0, "southern city with smaller longitude should come first"

    # Test 3: Cross-north-south city pair
    pair3 = normalize_city_pair(*north_west, *south_east)
    print(f"  Cross N-S city pair sorted: {pair3}")
    # Northern city has higher latitude, but sort is longitude-based
    # NW (70.0) vs SE (80.0) -> 70.0 first
    assert pair3[0][0] == 70.0, "Longitude determines sort order"

    print("? India terrain logic test passed\n")


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
    """Run all tests"""
    print("=" * 60)
    print("Geographic imputation unit tests")
    print("=" * 60)

    try:
        test_parse_lat_lon()
        test_normalize_city_pair()
        test_haversine()
        test_india_specific_logic()
        test_duplicate_pair_uses_true_mean()

        print("=" * 60)
        print("All tests passed!")
        print("Geographic imputation logic is correct and ready for real data.")
        print("=" * 60)

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
