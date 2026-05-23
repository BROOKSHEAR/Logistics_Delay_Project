"""
Geographic proximity-based transportation distance imputation

For India's inverted-triangle terrain, uses longitude→latitude sorting to generate standardized city pairs,
and fills missing TRANSPORTATION_DISTANCE_IN_KM by scanning nearby cities.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import math


def parse_lat_lon(lat_lon_str: str) -> Tuple[float, float]:
    """
    Parse a coordinate string like "28.6139, 77.2090"
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
    Normalize city pair: sort by longitude→latitude lexicographically

    For India's inverted-triangle terrain:
    - Large E-W span (68°E-97°E), large N-S span (8°N-37°N)
    - Many northern cities spread E-W, fewer southern cities clustered
    - Sort by longitude first to separate E/W, then by latitude

    Args:
        lat1, lon1: City 1 latitude, longitude
        lat2, lon2: City 2 latitude, longitude

    Returns:
        Normalized city pair ((lon1, lat1), (lon2, lat2))
    """
    city1 = (lon1, lat1)  # longitude first!
    city2 = (lon2, lat2)

    # Sort lexicographically (compare longitude first, then latitude)
    if city1 < city2:
        return (city1, city2)
    else:
        return (city2, city1)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute spherical distance between two lat/lon points (km)
    """
    R = 6371.0  # Earth radius in km

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
    Geographic proximity-based transportation distance imputer
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the imputer

        Args:
            df: DataFrame with the following columns:
                - Org_lat_lon: Origin coordinate string
                - Des_lat_lon: Destination coordinate string
                - TRANSPORTATION_DISTANCE_IN_KM: Distance (may contain NaN)
        """
        self.df = df.copy()

        # Parse coordinates
        self.df['orig_lat'], self.df['orig_lon'] = zip(*df['Org_lat_lon'].apply(parse_lat_lon))
        self.df['dest_lat'], self.df['dest_lon'] = zip(*df['Des_lat_lon'].apply(parse_lat_lon))

        # Extract city names (for debugging and display)
        self.df['orig_city'] = df['Origin_Location'].astype(str).str.split(',').str[1].str.strip().str.upper().fillna('UNKNOWN')
        self.df['dest_city'] = df['Destination_Location'].astype(str).str.split(',').str[1].str.strip().str.upper().fillna('UNKNOWN')

        # Build known distance city pair index
        self._build_distance_index()

    def _build_distance_index(self):
        """Build known distance city pair index"""
        # Get rows with distance data
        known_distances = self.df[self.df['TRANSPORTATION_DISTANCE_IN_KM'].notna()]

        # Accumulate distances and counts per normalized city pair, then compute true average
        self.distance_dict = {}
        pair_stats = {}
        self.city_coords = {}  # city name -> (latitude, longitude)

        for _, row in known_distances.iterrows():
            # Normalize city pair
            norm_pair = normalize_city_pair(
                row['orig_lat'], row['orig_lon'],
                row['dest_lat'], row['dest_lon']
            )

            # Store distance stats (if multiple same city pairs, take overall mean)
            distance = row['TRANSPORTATION_DISTANCE_IN_KM']
            if norm_pair in pair_stats:
                pair_stats[norm_pair]['sum'] += distance
                pair_stats[norm_pair]['count'] += 1
            else:
                pair_stats[norm_pair] = {'sum': distance, 'count': 1}

            # Store city coordinates (for nearby search)
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
        Search for nearby city pairs within given thresholds

        Args:
            target_pair: Target normalized city pair ((lon1, lat1), (lon2, lat2))
            lat_threshold: Latitude search threshold (degrees)
            lon_threshold: Longitude search threshold (degrees)

        Returns:
            Found distance value, or None if not found
        """
        target_city1, target_city2 = target_pair

        best_distance = None
        best_diff = float('inf')

        for known_pair, known_dist in self.distance_dict.items():
            known_city1, known_city2 = known_pair

            # Compute coordinate differences between city pairs
            # City 1 difference
            lat_diff1 = abs(target_city1[1] - known_city1[1])  # latitude is 2nd element
            lon_diff1 = abs(target_city1[0] - known_city1[0])  # longitude is 1st element

            # City 2 difference
            lat_diff2 = abs(target_city2[1] - known_city2[1])
            lon_diff2 = abs(target_city2[0] - known_city2[0])

            # Check if within threshold
            if (lat_diff1 <= lat_threshold and lon_diff1 <= lon_threshold and
                lat_diff2 <= lat_threshold and lon_diff2 <= lon_threshold):

                # Compute total difference (for best match selection)
                total_diff = lat_diff1 + lon_diff1 + lat_diff2 + lon_diff2

                if total_diff < best_diff:
                    best_diff = total_diff
                    best_distance = known_dist

        return best_distance

    def fill_missing_distances(self, max_search_radius: float = 5.0, verbose: bool = False) -> pd.DataFrame:
        """
        Fill missing transportation distances

        Args:
            max_search_radius: Maximum search radius (lat/lon threshold, expanded stepwise)
            verbose: Whether to print each fill, default False (summary only)

        Returns:
            DataFrame with filled distances
        """
        result_df = self.df.copy()
        missing_indices = result_df[result_df['TRANSPORTATION_DISTANCE_IN_KM'].isna()].index

        print(f"Found {len(missing_indices)} records with missing distance")

        filled_count = 0
        logged = 0  # only print first 10

        for idx in missing_indices:
            row = result_df.loc[idx]

            # Normalize city pair
            norm_pair = normalize_city_pair(
                row['orig_lat'], row['orig_lon'],
                row['dest_lat'], row['dest_lon']
            )

            # Try exact match
            if norm_pair in self.distance_dict:
                result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = self.distance_dict[norm_pair]
                filled_count += 1
                continue

            # Expand search radius stepwise
            found_distance = None
            for radius in np.arange(0.5, max_search_radius + 0.5, 0.5):
                lat_threshold = radius
                lon_threshold = radius * 2  # larger longitude threshold due to India's E-W span

                found_distance = self._find_nearby_pair(norm_pair, lat_threshold, lon_threshold)
                if found_distance is not None:
                    result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = found_distance
                    filled_count += 1
                    if verbose and logged < 10:
                        print(f"  Index {idx}: found nearby city pair within radius {radius}°, distance ≈ {found_distance:.1f} km")
                        logged += 1
                    break

            if found_distance is None:
                if verbose and logged < 10:
                    print(f"  Index {idx}: no match within {max_search_radius}° radius, falling back to global median")
                    logged += 1
                # Use global median as fallback
                global_median = result_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
                result_df.at[idx, 'TRANSPORTATION_DISTANCE_IN_KM'] = global_median

        print(f"\nImputation complete: {filled_count} records filled via geographic proximity")
        print(f"Remaining {len(missing_indices) - filled_count} records filled with median")

        # Verify imputation results
        still_missing = result_df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
        if still_missing > 0:
            print(f"Warning: {still_missing} records still missing, filled with median")
            median_val = result_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
            result_df['TRANSPORTATION_DISTANCE_IN_KM'] = result_df['TRANSPORTATION_DISTANCE_IN_KM'].fillna(median_val)

        return result_df

    def analyze_distance_distribution(self):
        """Analyze distance distribution, generate statistics"""
        original_missing = self.df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
        total_rows = len(self.df)

        print("=== Distance Analysis ===")
        print(f"Total records: {total_rows}")
        print(f"Originally missing: {original_missing} ({original_missing/total_rows*100:.2f}%)")

        if original_missing > 0:
            # Group by city pair
            missing_df = self.df[self.df['TRANSPORTATION_DISTANCE_IN_KM'].isna()]
            print(f"\nCity pairs missing distance: {len(missing_df)}")

            # Check for duplicate city pairs
            city_pairs = []
            for _, row in missing_df.iterrows():
                pair = normalize_city_pair(
                    row['orig_lat'], row['orig_lon'],
                    row['dest_lat'], row['dest_lon']
                )
                city_pairs.append(pair)

            unique_pairs = set(city_pairs)
            print(f"Unique city pairs: {len(unique_pairs)}")

        # Distance distribution statistics
        if original_missing < total_rows:
            known_distances = self.df['TRANSPORTATION_DISTANCE_IN_KM'].dropna()
            print(f"\nKnown distance statistics:")
            print(f"  Min: {known_distances.min():.1f} km")
            print(f"  Median: {known_distances.median():.1f} km")
            print(f"  Mean: {known_distances.mean():.1f} km")
            print(f"  Max: {known_distances.max():.1f} km")
            print(f"  Std: {known_distances.std():.1f} km")


def main():
    """Example usage"""
    import warnings
    warnings.filterwarnings('ignore')

    from logistics_delay.utils.paths import DATA_RAW

    # Load data
    print("Reading data...")
    df = pd.read_excel(DATA_RAW)

    # Create imputer
    filler = DistanceFiller(df)

    # Analyze original data
    filler.analyze_distance_distribution()

    # Fill missing distances
    print("\nStarting distance imputation...")
    filled_df = filler.fill_missing_distances(max_search_radius=3.0)

    # Verify imputation quality
    print("\n=== Imputation Validation ===")
    still_missing = filled_df['TRANSPORTATION_DISTANCE_IN_KM'].isna().sum()
    print(f"Post-fill missing records: {still_missing} ({still_missing/len(filled_df)*100:.2f}%)")

    # Compare with original median method
    original_median = df['TRANSPORTATION_DISTANCE_IN_KM'].median()
    filled_median = filled_df['TRANSPORTATION_DISTANCE_IN_KM'].median()
    print(f"Original median: {original_median:.1f} km")
    print(f"Post-fill median: {filled_median:.1f} km")
    print(f"Change: {((filled_median - original_median) / original_median * 100):+.2f}%")


if __name__ == "__main__":
    main()
