#!/usr/bin/env python3
"""
Inspect all shapefile columns to ensure we haven't missed any important fields.
"""

from pathlib import Path
import geopandas as gpd
import pandas as pd

def inspect_shapefiles():
    """Inspect all shapefiles and print their column names"""

    raw_data_dir = Path("data/raw")

    print("="*80)
    print("SHAPEFILE COLUMN INSPECTION")
    print("="*80)

    # Find all shapefile directories
    shapefile_dirs = [d for d in raw_data_dir.iterdir() if d.is_dir()]

    for dataset_dir in sorted(shapefile_dirs):
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset_dir.name}")
        print(f"{'='*80}")

        # Find all shapefiles
        shp_files = list(dataset_dir.glob("*.shp"))

        if not shp_files:
            print("  No shapefiles found")
            continue

        for shp_file in shp_files:
            print(f"\nFile: {shp_file.name}")
            print("-" * 80)

            try:
                # Read shapefile
                gdf = gpd.read_file(shp_file)

                print(f"Rows: {len(gdf):,}")
                print(f"CRS: {gdf.crs}")
                print(f"\nColumns ({len(gdf.columns)}):")

                # Get non-geometry columns
                non_geom_cols = [col for col in gdf.columns if col != 'geometry']

                # Print column info
                for col in non_geom_cols:
                    dtype = gdf[col].dtype
                    non_null = gdf[col].notna().sum()
                    percent_filled = 100 * non_null / len(gdf) if len(gdf) > 0 else 0

                    # Sample values (first non-null value)
                    sample = None
                    for val in gdf[col]:
                        if pd.notna(val):
                            sample = val
                            break

                    sample_str = str(sample)[:50] if sample is not None else "None"

                    print(f"  {col:<20} {str(dtype):<12} {percent_filled:>6.1f}% filled  Sample: {sample_str}")

                print(f"\nSample record (first row, non-geometry fields):")
                first_row = gdf.iloc[0]
                for col in non_geom_cols:
                    val = str(first_row[col])[:60]
                    print(f"  {col}: {val}")

            except Exception as e:
                print(f"  Error reading shapefile: {e}")

    print("\n" + "="*80)
    print("INSPECTION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    inspect_shapefiles()
