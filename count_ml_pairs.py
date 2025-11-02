#!/usr/bin/env python3
"""
Count ML training pairs with step-by-step filtering.

Filters applied:
1. Remove single-perimeter incidents
2. Remove pairs with same timestamp
3. Remove pairs with gap > 6 days
"""

import geopandas as gpd


def main():
    print("Loading processed fire perimeters...")
    gdf = gpd.read_parquet("data/processed/fire_perimeters.parquet")

    print(f"\n{'=' * 70}")
    print("ML TRAINING PAIRS - STEP-BY-STEP FILTERING")
    print(f"{'=' * 70}")

    # Step 0: Starting point
    print(f"\n{'STEP 0: Starting dataset':<50}")
    print(f"  Total perimeters: {len(gdf):,}")
    print(f"  Total incidents: {gdf['incident_id'].nunique():,}")

    # Step 1: Filter to multi-perimeter incidents only
    print(f"\n{'STEP 1: Filter to multi-perimeter incidents':<50}")
    perimeters_per_incident = gdf.groupby("incident_id").size()
    single_perim = len(perimeters_per_incident[perimeters_per_incident == 1])
    multi_perim_incidents = perimeters_per_incident[perimeters_per_incident >= 2]

    print(f"  Before: {len(perimeters_per_incident):,} incidents")
    print(f"  Removed (single perimeter): {single_perim:,} incidents")
    print(f"  After: {len(multi_perim_incidents):,} incidents")
    print(f"  Total pairs: {(multi_perim_incidents - 1).sum():,}")

    # Step 2: Filter to incidents with complete dates
    print(f"\n{'STEP 2: Filter to incidents with all dates':<50}")
    incidents_with_all_dates = []

    for incident_id in multi_perim_incidents.index:
        group = gdf[gdf["incident_id"] == incident_id]
        if group["perimeter_date"].notna().all():
            incidents_with_all_dates.append(incident_id)

    before_date_filter = len(multi_perim_incidents)
    after_date_filter = len(incidents_with_all_dates)

    # Calculate pairs before filtering same timestamp
    total_pairs_with_dates = 0
    for incident_id in incidents_with_all_dates:
        group = gdf[gdf["incident_id"] == incident_id]
        total_pairs_with_dates += len(group) - 1

    print(f"  Before: {before_date_filter:,} incidents")
    print(
        f"  Removed (missing dates): {before_date_filter - after_date_filter:,} incidents"
    )
    print(f"  After: {after_date_filter:,} incidents")
    print(f"  Total pairs: {total_pairs_with_dates:,}")

    # Step 3: Remove pairs with same timestamp
    print(f"\n{'STEP 3: Remove pairs with same timestamp':<50}")

    filtered_incidents = []
    pairs_before_timestamp_filter = 0
    pairs_after_timestamp_filter = 0
    same_timestamp_pairs_removed = 0

    for incident_id in incidents_with_all_dates:
        group = gdf[gdf["incident_id"] == incident_id]
        sorted_group = group.sort_values("perimeter_date")
        datetimes = sorted_group["perimeter_date"].values

        # Count all pairs before filtering
        pairs_before_timestamp_filter += len(datetimes) - 1

        # Count pairs with different timestamps
        unique_pairs = 0
        for i in range(len(datetimes) - 1):
            if datetimes[i + 1] != datetimes[i]:
                unique_pairs += 1
            else:
                same_timestamp_pairs_removed += 1

        # Only keep incident if it has at least one unique timestamp pair
        if unique_pairs > 0:
            filtered_incidents.append(incident_id)
            pairs_after_timestamp_filter += unique_pairs

    print(
        f"  Before: {after_date_filter:,} incidents, {pairs_before_timestamp_filter:,} pairs"
    )
    print(f"  Removed pairs (same timestamp): {same_timestamp_pairs_removed:,}")
    print(
        f"  After: {len(filtered_incidents):,} incidents, {pairs_after_timestamp_filter:,} pairs"
    )

    # Step 4: Remove pairs with gap > 6 days
    print(f"\n{'STEP 4: Remove pairs with gap > 6 days':<50}")

    final_incidents = []
    pairs_before_day_filter = pairs_after_timestamp_filter
    pairs_after_day_filter = 0
    pairs_removed_long_gap = 0

    for incident_id in filtered_incidents:
        group = gdf[gdf["incident_id"] == incident_id]
        sorted_group = group.sort_values("perimeter_date")
        dates = sorted_group["perimeter_date"].dt.date.values
        datetimes = sorted_group["perimeter_date"].values

        # Count pairs with gap <= 6 days and different timestamps
        valid_pairs = 0
        for i in range(len(dates) - 1):
            day_diff = (dates[i + 1] - dates[i]).days
            timestamp_diff = datetimes[i + 1] != datetimes[i]

            if timestamp_diff:
                if day_diff <= 6:
                    valid_pairs += 1
                else:
                    pairs_removed_long_gap += 1

        # Only keep incident if it has at least one valid pair
        if valid_pairs > 0:
            final_incidents.append(incident_id)
            pairs_after_day_filter += valid_pairs

    print(
        f"  Before: {len(filtered_incidents):,} incidents, {pairs_before_day_filter:,} pairs"
    )
    print(f"  Removed pairs (gap > 6 days): {pairs_removed_long_gap:,}")
    print(
        f"  After: {len(final_incidents):,} incidents, {pairs_after_day_filter:,} pairs"
    )

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"\nUsable for ML training:")
    print(f"  Incidents: {len(final_incidents):,}")
    print(f"  Training pairs: {pairs_after_day_filter:,}")
    print(f"\nReduction from original dataset:")
    print(f"  Started with: {gdf['incident_id'].nunique():,} incidents")
    print(
        f"  Ended with: {len(final_incidents):,} incidents ({100 * len(final_incidents) / gdf['incident_id'].nunique():.1f}%)"
    )
    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()
