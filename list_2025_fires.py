#!/usr/bin/env python3
"""
List all fires from the 2025 YearToDate shapefile with their start dates and names.
"""

import geopandas as gpd
import pandas as pd


def main():
    # Read the 2025 Year to Date shapefile
    shp_file = "data/raw/WFIGS_Interagency_Perimeters_YearToDate_-5395415287356828930/Perimeters.shp"

    print("Reading 2025 Year to Date shapefile...")
    gdf = gpd.read_file(shp_file)

    print(f"Total perimeters in YearToDate: {len(gdf):,}\n")

    # Get fire discovery/start dates and names
    # Try multiple date fields to find fire start date
    gdf['fire_start'] = pd.to_datetime(gdf.get('attr_Fir_7'), errors='coerce')  # FireDiscoveryDateTime
    gdf['fire_name'] = gdf.get('attr_Inc_2', gdf.get('attr_Inc_1'))  # IncidentName

    # Group by incident to get unique fires
    # Use attr_Uniqu for UniqueFireIdentifier
    gdf['incident_id'] = gdf.get('attr_Uniqu', gdf.get('attr_Irwin'))

    # Get unique fires
    unique_fires = gdf.dropna(subset=['incident_id']).groupby('incident_id').agg({
        'fire_name': 'first',
        'fire_start': 'first',
        'poly_Acres': 'max'  # Max acres for the incident
    }).reset_index()

    # Sort by start date
    unique_fires = unique_fires.sort_values('fire_start')

    print(f"Unique fires in 2025: {len(unique_fires):,}\n")
    print(f"{'Fire Name':<40} {'Start Date':<12} {'Incident ID':<30} {'Max Acres':<12}")
    print("-" * 100)

    for idx, row in unique_fires.iterrows():
        name = str(row['fire_name'])[:40] if pd.notna(row['fire_name']) else 'Unknown'
        start = row['fire_start'].strftime('%Y-%m-%d') if pd.notna(row['fire_start']) else 'Unknown'
        incident_id = str(row['incident_id'])[:30]
        acres = f"{row['poly_Acres']:,.0f}" if pd.notna(row['poly_Acres']) else 'Unknown'

        print(f"{name:<40} {start:<12} {incident_id:<30} {acres:<12}")


if __name__ == "__main__":
    main()
