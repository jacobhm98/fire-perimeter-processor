# fire-perimeter-processor

This project takes a set of historical fire perimeters from the [NIFC open data](https://data-nifc.opendata.arcgis.com/) website, and processes them into a format more easily usable for machine learning applications.

For more detail about each of the datasets mentioned here, please refer to [WFIGS hub](https://wfigs-nifc.hub.arcgis.com/) and the [FAQ](data-nifc.opendata.arcgis.com/pages/faqs)

## Obtain the datasets

Create the needed directories:

```bash
mkdir -p data/raw
```

Download each of the following datasets in shapefile format from the https://data-nifc.opendata.arcgis.com website:

- WFIGS Interagency Fire Perimeters
- Historic Perimeters Combined 2000-2018 GeoMAC
- InterAgencyFirePerimeterHistory All Years View
- WFIGS 2025 Interagency Fire Perimeters to Date
- WFIGS Current Interagency Fire Perimeters

Put the .zip files into the ./data/raw directory

To extract the raw shapefiles:

```bash
./unzip_data.sh
```

## Process the datasets

Once the shapefiles are extracted, process them into a unified Parquet format:

```bash
# Install dependencies
uv sync

# Run the processing script
uv run process_fire_perimeters.py
```

This will:

- Read all shapefile datasets from `data/raw/`
- Normalize field names across different naming conventions
- Unify incident IDs using a 6-tier strategy (formal → IRWIN → local → FOR → GEO → Global IDs)
- Parse and normalize dates
- Filter out invalid data (pre-1900 dates, missing required fields, invalid geometries)
- Deduplicate across datasets
- Output to `data/processed/fire_perimeters.parquet`
- Generate a processing report at `data/processed/processing_report.json`

### Output Schema

The processed Parquet file contains:

- `incident_id` - Unified incident identifier
- `incident_name` - Fire name
- `fire_year` - Year of the fire
- `unit_id` - Responsible unit identifier
- `perimeter_date` - Perimeter date/time
- `geometry` - Polygon geometry (WGS84/EPSG:4326)
- `source_dataset` - Original dataset name
- `id_type` - Type of ID resolution ('formal', 'irwin', 'local', 'for_id', 'geo_id', 'global_id')

### Count number of usable ML samples

You can then run

```bash
uv run count_ml_pairs.py
```

in order to count how many possible (feature, label) pairs there are present in the parquet output.
