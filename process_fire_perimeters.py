#!/usr/bin/env python3
"""
Production Fire Perimeter Processor

Processes multiple wildfire perimeter shapefiles into a unified Parquet format.
Handles field name variations, ID unification, date normalization, and deduplication.

Usage:
    uv run process_fire_perimeters.py
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import pandas as pd

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


class Config:
    """Configuration for processing pipeline"""

    # Paths
    RAW_DATA_DIR = Path("data/raw")
    PROCESSED_DATA_DIR = Path("data/processed")
    OUTPUT_PARQUET = PROCESSED_DATA_DIR / "fire_perimeters.parquet"
    STATS_OUTPUT = PROCESSED_DATA_DIR / "processing_report.json"
    NO_ID_OUTPUT = PROCESSED_DATA_DIR / "no_id_records.csv"

    # Dataset configurations - all shapefiles now
    DATASETS = {
        "historic_geomac": {
            "pattern": "Historic_Geomac_*",
            "name": "Historic GeoMAC (2000-2018)",
        },
        "interagency_history": {
            "pattern": "InterAgencyFirePerimeterHistory_*",
            "name": "Interagency Fire Perimeter History",
        },
        "wfigs_recent": {
            "pattern": "WFIGS_Interagency_Perimeters_-8845918407708086874*",
            "name": "WFIGS Recent (2016-2025)",
        },
        "wfigs_ytd": {
            "pattern": "WFIGS_Interagency_Perimeters_YearToDate_*",
            "name": "WFIGS Year to Date",
        },
        "wfigs_current": {
            "pattern": "WFIGS_Interagency_Perimeters_Current_*",
            "name": "WFIGS Current",
        },
    }

    # Target CRS
    TARGET_CRS = "EPSG:4326"  # WGS84

    # Quality filters
    MIN_YEAR = 1900
    MAX_YEAR = datetime.now().year + 1
    REQUIRED_FIELDS = ["incident_name", "fire_year", "geometry"]


# ============================================================================
# FIELD NAME MAPPING
# ============================================================================


class FieldMapper:
    """Maps various field naming conventions to unified schema"""

    # Priority-ordered field names for each unified field
    INCIDENT_ID_FIELDS = [
        "uniquefireidentifier",
        "unqe_fire_",
        "uniquefire",
        "uniquefir",
        "attr_uniquefireidentifier",
        "attr_unqe_fire_",
        "UniqueFireIdentifier",
        "Unique_Fir",
    ]

    IRWIN_ID_FIELDS = [
        "irwinid",
        "irwin_id",
        "attr_irwinid",
        "attr_irwin_id",
        "IrwinID",
    ]

    INCIDENT_NAME_FIELDS = [
        "incidentname",
        "incident",
        "incidentna",
        "incidentn",
        "attr_incidentname",
        "poly_incidentname",
        "poly_incid",
        "IncidentNa",
        "Incident_N",
    ]

    FIRE_YEAR_FIELDS = [
        "fire_year",
        "fireyear",
        "year",
        "FireYear",
        "Fire_Year",
    ]

    UNIT_ID_FIELDS = [
        "unit_id",
        "unitid",
        "poo_resp_i",
        "poo_respon",
        "attr_unit_id",
        "UnitID",
        "Unit_ID",
    ]

    DATE_FIELDS = [
        "perimeterdatetime",  # Primary perimeter datetime
        "poly_polygondatetime",  # Polygon datetime
        "poly_polyg",  # Polygon datetime (truncated)
        "date_cur",  # Date current
    ]

    # Additional ID fields
    LOCAL_INCIDENT_FIELDS = [
        "localincid",
        "local_num",
        "attr_local",
        "localinci",
    ]

    FOR_ID_FIELDS = [
        "forid",
        "attr_forid",
        "poly_forid",
    ]

    GEO_ID_FIELDS = [
        "geo_id",
        "geoid",
    ]

    GLOBAL_ID_FIELDS = [
        "globalid",
        "global_id",
    ]

    STATE_FIELDS = [
        "state",
        "attr_poost",
        "attr_poo_st",
        "poostate",
    ]

    @staticmethod
    def normalize_field_name(field: str) -> str:
        """Normalize field name to lowercase with underscores"""
        return field.lower().strip().replace(" ", "_")

    @classmethod
    def find_field(cls, fields: List[str], candidates: List[str]) -> Optional[str]:
        """Find first matching field from candidates list"""
        normalized = {cls.normalize_field_name(f): f for f in fields}
        for candidate in candidates:
            normalized_candidate = cls.normalize_field_name(candidate)
            if normalized_candidate in normalized:
                return normalized[normalized_candidate]
        return None


# ============================================================================
# ID UNIFICATION
# ============================================================================


class IDUnifier:
    """Handles incident ID unification across datasets"""

    @staticmethod
    def is_valid_id(id_val: Any) -> bool:
        """Check if an ID value is valid (not None, empty, or placeholder)"""
        if not id_val:
            return False
        id_str = str(id_val).strip().lower()
        return id_str not in ["none", "nan", "", "null"]

    @classmethod
    def unify_id(cls, row: Dict[str, Any]) -> Tuple[Optional[str], str]:
        """
        Unify incident ID using 6-tier strategy
        Returns: (incident_id, id_type)
        id_type: 'formal', 'irwin', 'local', 'for_id', 'geo_id', 'global_id', or 'none'
        """
        # Tier 1: Try formal unique fire identifier
        incident_id = row.get("incident_id_raw")
        if cls.is_valid_id(incident_id):
            return str(incident_id).strip(), "formal"

        # Tier 2: Try IRWIN ID
        irwin_id = row.get("irwin_id_raw")
        if cls.is_valid_id(irwin_id):
            return str(irwin_id).strip(), "irwin"

        # Tier 3: Try local incident ID (combined with year/unit for uniqueness)
        local_id = row.get("local_incident_id")
        if cls.is_valid_id(local_id):
            year = row.get("fire_year")
            unit = row.get("unit_id")
            if year and unit:
                return f"{year}-{unit}-{local_id}", "local"

        # Tier 4: Try FOR ID
        for_id = row.get("for_id")
        if cls.is_valid_id(for_id):
            return str(for_id).strip(), "for_id"

        # Tier 5: Try GEO ID
        geo_id = row.get("geo_id")
        if cls.is_valid_id(geo_id):
            return str(geo_id).strip(), "geo_id"

        # Tier 6: Try Global ID
        global_id = row.get("global_id")
        if cls.is_valid_id(global_id):
            return str(global_id).strip(), "global_id"

        return None, "none"


# ============================================================================
# DATE PARSING
# ============================================================================


class DateParser:
    """Parses various date formats"""

    FORMATS = [
        "%Y%m%d%H%M%S",  # 20020201000000
        "%Y%m%d",  # 20020201
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
    ]

    @classmethod
    def parse_date(cls, date_str: Any) -> Optional[datetime]:
        """Parse date from various formats"""
        if pd.isna(date_str) or not date_str:
            return None

        date_str = str(date_str).strip()

        # Try pandas datetime parser first (handles RFC formats)
        try:
            return pd.to_datetime(date_str, errors="coerce")
        except:
            pass

        # Try specific formats
        for fmt in cls.FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        return None

    @staticmethod
    def is_valid_date(dt: datetime, min_year: int, max_year: int) -> bool:
        """Check if date is within valid range"""
        if not dt or pd.isna(dt):
            return False
        return min_year <= dt.year <= max_year


# ============================================================================
# DATASET READER
# ============================================================================


class DatasetReader:
    """Reads and normalizes shapefile datasets"""

    def __init__(self, config: Config):
        self.config = config
        self.field_mapper = FieldMapper()

    def read_shapefile(self, path: Path) -> gpd.GeoDataFrame:
        """Read and normalize shapefile"""
        logger.info(f"  Reading: {path.name}")

        try:
            gdf = gpd.read_file(path)
            logger.info(f"    Loaded {len(gdf)} features")
            return gdf
        except Exception as e:
            logger.error(f"    Error reading shapefile: {e}")
            return gpd.GeoDataFrame()

    def normalize_gdf(self, gdf: gpd.GeoDataFrame, source: str) -> gpd.GeoDataFrame:
        """Normalize GeoDataFrame to unified schema"""
        if gdf.empty:
            return gdf

        field_names = list(gdf.columns)

        # Extract primary ID fields
        incident_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.INCIDENT_ID_FIELDS
        )
        irwin_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.IRWIN_ID_FIELDS
        )

        gdf["incident_id_raw"] = gdf[incident_id_field] if incident_id_field else None
        gdf["irwin_id_raw"] = gdf[irwin_id_field] if irwin_id_field else None

        # Extract additional ID fields that could help with deduplication
        local_incident_field = self.field_mapper.find_field(
            field_names, self.field_mapper.LOCAL_INCIDENT_FIELDS
        )
        gdf["local_incident_id"] = (
            gdf[local_incident_field] if local_incident_field else None
        )

        for_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.FOR_ID_FIELDS
        )
        gdf["for_id"] = gdf[for_id_field] if for_id_field else None

        geo_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.GEO_ID_FIELDS
        )
        gdf["geo_id"] = gdf[geo_id_field] if geo_id_field else None

        global_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.GLOBAL_ID_FIELDS
        )
        gdf["global_id"] = gdf[global_id_field] if global_id_field else None

        # Extract core fields
        incident_name_field = self.field_mapper.find_field(
            field_names, self.field_mapper.INCIDENT_NAME_FIELDS
        )
        gdf["incident_name"] = gdf[incident_name_field] if incident_name_field else None

        fire_year_field = self.field_mapper.find_field(
            field_names, self.field_mapper.FIRE_YEAR_FIELDS
        )
        gdf["fire_year"] = gdf[fire_year_field] if fire_year_field else None

        unit_id_field = self.field_mapper.find_field(
            field_names, self.field_mapper.UNIT_ID_FIELDS
        )
        gdf["unit_id"] = gdf[unit_id_field] if unit_id_field else None

        state_field = self.field_mapper.find_field(
            field_names, self.field_mapper.STATE_FIELDS
        )
        gdf["state"] = gdf[state_field] if state_field else None

        # Extract perimeter date field
        date_field = self.field_mapper.find_field(
            field_names, self.field_mapper.DATE_FIELDS
        )
        gdf["perimeter_date_raw"] = gdf[date_field] if date_field else None

        gdf["source_dataset"] = source

        # Keep normalized columns plus geometry
        keep_cols = [
            "incident_id_raw",
            "irwin_id_raw",
            "local_incident_id",
            "for_id",
            "geo_id",
            "global_id",
            "incident_name",
            "fire_year",
            "unit_id",
            "state",
            "perimeter_date_raw",
            "source_dataset",
            "geometry",
        ]
        gdf = gdf[[col for col in keep_cols if col in gdf.columns]]

        return gdf

    def read_dataset(self, dataset_key: str) -> gpd.GeoDataFrame:
        """Read a dataset and return normalized GeoDataFrame"""
        dataset_config = self.config.DATASETS[dataset_key]
        dataset_name = dataset_config["name"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing: {dataset_name}")
        logger.info(f"{'=' * 60}")

        # Find dataset directories
        pattern = dataset_config["pattern"]
        matching_dirs = list(self.config.RAW_DATA_DIR.glob(pattern))

        if not matching_dirs:
            logger.warning(f"No directories found matching pattern: {pattern}")
            return gpd.GeoDataFrame()

        all_gdfs = []

        for dataset_dir in matching_dirs:
            if not dataset_dir.is_dir():
                continue

            logger.info(f"Searching in: {dataset_dir.name}")

            # Find all shapefiles in directory
            shp_files = list(dataset_dir.glob("*.shp"))

            if not shp_files:
                logger.warning(f"  No shapefiles found in {dataset_dir.name}")
                continue

            for shp_file in shp_files:
                gdf = self.read_shapefile(shp_file)
                if not gdf.empty:
                    # Convert to target CRS before normalizing
                    if gdf.crs is not None and str(gdf.crs) != self.config.TARGET_CRS:
                        logger.info(
                            f"    Converting from {gdf.crs} to {self.config.TARGET_CRS}"
                        )
                        gdf = gdf.to_crs(self.config.TARGET_CRS)
                    elif gdf.crs is None:
                        logger.info(f"    Setting CRS to {self.config.TARGET_CRS}")
                        gdf.set_crs(self.config.TARGET_CRS, inplace=True)

                    gdf = self.normalize_gdf(gdf, dataset_name)
                    all_gdfs.append(gdf)

        if not all_gdfs:
            logger.warning(f"No data extracted from {dataset_name}")
            return gpd.GeoDataFrame()

        # Combine all GeoDataFrames from this dataset
        combined = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        logger.info(f"Total features from {dataset_name}: {len(combined)}")

        return combined


# ============================================================================
# QUALITY FILTERING
# ============================================================================


class QualityFilter:
    """Applies quality filters to data"""

    def __init__(self, config: Config):
        self.config = config
        self.stats = {
            "total_input": 0,
            "missing_required_fields": 0,
            "invalid_dates": 0,
            "invalid_geometry": 0,
            "passed": 0,
        }

    def filter_gdf(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Apply quality filters to GeoDataFrame"""
        logger.info("\nApplying quality filters...")

        self.stats["total_input"] = len(gdf)
        initial_count = len(gdf)

        # Parse dates
        logger.info("  Parsing dates...")
        gdf["perimeter_date"] = gdf["perimeter_date_raw"].apply(DateParser.parse_date)

        # Filter invalid dates (if date exists)
        logger.info("  Filtering invalid dates...")
        date_mask = gdf["perimeter_date"].isna() | gdf["perimeter_date"].apply(
            lambda x: DateParser.is_valid_date(
                x, self.config.MIN_YEAR, self.config.MAX_YEAR
            )
        )
        invalid_dates = len(gdf) - date_mask.sum()
        gdf = gdf[date_mask]
        self.stats["invalid_dates"] = invalid_dates
        logger.info(f"    Removed {invalid_dates} records with invalid dates")

        # Filter missing required fields
        logger.info("  Checking required fields...")
        for field in self.config.REQUIRED_FIELDS:
            if field == "geometry":
                mask = gdf["geometry"].notna()
            else:
                mask = (
                    gdf[field].notna()
                    & (gdf[field] != "")
                    & (gdf[field].astype(str) != "None")
                )
            missing = len(gdf) - mask.sum()
            gdf = gdf[mask]
            if missing > 0:
                logger.info(f"    Removed {missing} records missing {field}")
                self.stats["missing_required_fields"] += missing

        # Validate geometries
        logger.info("  Validating geometries...")
        valid_geom = gdf["geometry"].apply(
            lambda g: g is not None and hasattr(g, "is_valid") and g.is_valid
        )
        invalid_geom = len(gdf) - valid_geom.sum()
        gdf = gdf[valid_geom]
        self.stats["invalid_geometry"] = invalid_geom
        if invalid_geom > 0:
            logger.info(f"    Removed {invalid_geom} records with invalid geometry")

        self.stats["passed"] = len(gdf)
        logger.info(
            f"\n  Quality filter results: {len(gdf):,} / {initial_count:,} passed ({100 * len(gdf) / initial_count:.1f}%)"
        )

        return gdf


# ============================================================================
# DEDUPLICATION
# ============================================================================


class Deduplicator:
    """Handles deduplication across datasets"""

    def __init__(self):
        self.stats = {"total_input": 0, "duplicates_removed": 0, "unique_output": 0}

    def deduplicate(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Remove duplicate perimeters across datasets"""
        logger.info("\nDeduplicating across datasets...")

        self.stats["total_input"] = len(gdf)
        initial_count = len(gdf)

        # Create deduplication key from multiple fields
        # Use incident_id, fire_year, and geometry centroid
        logger.info("  Creating deduplication keys...")
        gdf["_dedup_key"] = gdf.apply(
            lambda row: f"{row['incident_id']}_{row['fire_year']}_{row['geometry'].centroid.wkt if row['geometry'] else 'NOGEOM'}",
            axis=1,
        )

        # Sort by source priority (keep most complete records)
        source_priority = {
            "Interagency Fire Perimeter History": 1,
            "Historic GeoMAC (2000-2018)": 2,
            "WFIGS Recent (2016-2025)": 3,
            "WFIGS Year to Date": 4,
            "WFIGS Current": 5,
        }
        gdf["_priority"] = gdf["source_dataset"].map(source_priority).fillna(99)

        # Reset index to use as tiebreaker for stable sorting
        gdf = gdf.reset_index(drop=True)
        gdf = gdf.sort_values("_priority", kind="stable")

        # Keep first occurrence (highest priority)
        logger.info("  Removing duplicates...")
        gdf = gdf.drop_duplicates(subset="_dedup_key", keep="first")

        # Clean up temp columns
        gdf = gdf.drop(columns=["_dedup_key", "_priority"])

        self.stats["duplicates_removed"] = initial_count - len(gdf)
        self.stats["unique_output"] = len(gdf)

        logger.info(f"  Removed {self.stats['duplicates_removed']:,} duplicates")
        logger.info(f"  Unique records: {len(gdf):,}")

        return gdf


# ============================================================================
# MAIN PROCESSOR
# ============================================================================


class FirePerimeterProcessor:
    """Main processing pipeline"""

    def __init__(self, config: Config):
        self.config = config
        self.reader = DatasetReader(config)
        self.quality_filter = QualityFilter(config)
        self.deduplicator = Deduplicator()
        self.stats = {"datasets": {}, "overall": {}}

    def process_all_datasets(self) -> gpd.GeoDataFrame:
        """Process all datasets and combine"""
        logger.info("\n" + "=" * 60)
        logger.info("FIRE PERIMETER PROCESSING PIPELINE")
        logger.info("=" * 60)

        all_gdfs = []

        # Read all datasets
        for dataset_key in self.config.DATASETS.keys():
            gdf = self.reader.read_dataset(dataset_key)
            if not gdf.empty:
                all_gdfs.append(gdf)
                self.stats["datasets"][dataset_key] = {
                    "raw_count": len(gdf),
                    "source": self.config.DATASETS[dataset_key]["name"],
                }

        if not all_gdfs:
            logger.error("No data extracted from any dataset!")
            return gpd.GeoDataFrame()

        # Combine all datasets
        logger.info("\n" + "=" * 60)
        logger.info("COMBINING DATASETS")
        logger.info("=" * 60)
        combined_gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True))
        logger.info(f"Combined total: {len(combined_gdf):,} records")

        # Fill in missing fire_year from perimeter_date before ID unification
        logger.info("\nDeriving fire_year from perimeter_date where missing...")
        # Parse perimeter dates
        combined_gdf["_temp_date"] = combined_gdf["perimeter_date_raw"].apply(
            DateParser.parse_date
        )
        # Fill in fire_year from date year where fire_year is missing
        missing_year_mask = (
            combined_gdf["fire_year"].isna() & combined_gdf["_temp_date"].notna()
        )
        combined_gdf.loc[missing_year_mask, "fire_year"] = combined_gdf.loc[
            missing_year_mask, "_temp_date"
        ].dt.year
        filled_count = missing_year_mask.sum()
        logger.info(
            f"  Filled {filled_count:,} missing fire_year values from perimeter_date"
        )
        # Clean up temp column
        combined_gdf = combined_gdf.drop(columns=["_temp_date"])

        # Unify incident IDs
        logger.info("\nUnifying incident IDs...")
        id_results = combined_gdf.apply(
            lambda row: IDUnifier.unify_id(row.to_dict()),
            axis=1,
        )
        combined_gdf["incident_id"] = id_results.apply(lambda x: x[0])
        combined_gdf["id_type"] = id_results.apply(lambda x: x[1])

        # Log ID statistics
        id_counts = combined_gdf["id_type"].value_counts()
        logger.info("  ID type distribution:")
        for id_type, count in id_counts.items():
            logger.info(
                f"    {id_type}: {count:,} ({100 * count / len(combined_gdf):.1f}%)"
            )

        # Save records with no ID to error file
        no_id_records = combined_gdf[combined_gdf["id_type"] == "none"]
        if len(no_id_records) > 0:
            logger.info(
                f"\nSaving {len(no_id_records):,} records with no ID to {self.config.NO_ID_OUTPUT}"
            )
            # Create output directory
            self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
            # Save metadata only (exclude geometry for readability)
            metadata_cols = [col for col in no_id_records.columns if col != "geometry"]
            no_id_records[metadata_cols].to_csv(self.config.NO_ID_OUTPUT, index=False)
            logger.info(f"  Saved {len(no_id_records):,} no-ID records")

        # Filter out records with no ID
        logger.info("\nFiltering out records with no incident ID...")
        initial_count = len(combined_gdf)
        combined_gdf = combined_gdf[combined_gdf["id_type"] != "none"]
        logger.info(
            f"  Removed {initial_count - len(combined_gdf):,} records with no ID"
        )

        # Apply quality filtering
        combined_gdf = self.quality_filter.filter_gdf(combined_gdf)

        # Deduplicate
        combined_gdf = self.deduplicator.deduplicate(combined_gdf)

        # Ensure CRS
        if combined_gdf.crs is None:
            logger.info(f"\nSetting CRS to {self.config.TARGET_CRS}")
            combined_gdf.set_crs(self.config.TARGET_CRS, inplace=True)
        elif str(combined_gdf.crs) != self.config.TARGET_CRS:
            logger.info(
                f"\nReprojecting from {combined_gdf.crs} to {self.config.TARGET_CRS}"
            )
            combined_gdf = combined_gdf.to_crs(self.config.TARGET_CRS)

        # Select final columns
        final_columns = [
            "incident_id",
            "incident_name",
            "fire_year",
            "unit_id",
            "perimeter_date",
            "geometry",
            "source_dataset",
            "id_type",
        ]
        combined_gdf = combined_gdf[
            [col for col in final_columns if col in combined_gdf.columns]
        ]

        # Convert fire_year to numeric (for stats and Parquet compatibility)
        combined_gdf["fire_year"] = pd.to_numeric(
            combined_gdf["fire_year"], errors="coerce"
        )

        # Compile overall stats
        self.stats["overall"] = {
            "total_records": len(combined_gdf),
            "unique_incidents": combined_gdf["incident_id"].nunique(),
            "date_range": {
                "min": str(combined_gdf["perimeter_date"].min())
                if not combined_gdf["perimeter_date"].isna().all()
                else None,
                "max": str(combined_gdf["perimeter_date"].max())
                if not combined_gdf["perimeter_date"].isna().all()
                else None,
            },
            "year_range": {
                "min": int(combined_gdf["fire_year"].min())
                if not combined_gdf["fire_year"].isna().all()
                else None,
                "max": int(combined_gdf["fire_year"].max())
                if not combined_gdf["fire_year"].isna().all()
                else None,
            },
            "id_type_distribution": {k: int(v) for k, v in id_counts.items()},
            "quality_filter_stats": self.quality_filter.stats,
            "deduplication_stats": self.deduplicator.stats,
        }

        # Calculate perimeters per incident statistics
        perimeters_per_incident = combined_gdf.groupby("incident_id").size()
        multi_perimeter = perimeters_per_incident[perimeters_per_incident > 1]

        self.stats["overall"]["incident_stats"] = {
            "total_incidents": len(perimeters_per_incident),
            "single_perimeter_incidents": len(
                perimeters_per_incident[perimeters_per_incident == 1]
            ),
            "multi_perimeter_incidents": len(multi_perimeter),
            "avg_perimeters_per_incident": float(perimeters_per_incident.mean()),
            "max_perimeters": int(perimeters_per_incident.max()),
        }

        logger.info("\nIncident statistics:")
        logger.info(
            f"  Total unique incidents: {self.stats['overall']['incident_stats']['total_incidents']:,}"
        )
        logger.info(
            f"  Single perimeter: {self.stats['overall']['incident_stats']['single_perimeter_incidents']:,}"
        )
        logger.info(
            f"  Multi-perimeter: {self.stats['overall']['incident_stats']['multi_perimeter_incidents']:,}"
        )
        logger.info(
            f"  Max perimeters for one incident: {self.stats['overall']['incident_stats']['max_perimeters']:,}"
        )

        return combined_gdf

    def save_output(self, gdf: gpd.GeoDataFrame):
        """Save processed data and statistics"""
        logger.info("\n" + "=" * 60)
        logger.info("SAVING OUTPUT")
        logger.info("=" * 60)

        # Create output directory
        self.config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Save Parquet
        logger.info(f"Writing Parquet: {self.config.OUTPUT_PARQUET}")
        gdf.to_parquet(self.config.OUTPUT_PARQUET, index=False)
        file_size_mb = self.config.OUTPUT_PARQUET.stat().st_size / (1024 * 1024)
        logger.info(f"  File size: {file_size_mb:.1f} MB")

        # Save statistics
        logger.info(f"Writing statistics: {self.config.STATS_OUTPUT}")
        self.stats["processing_timestamp"] = datetime.now().isoformat()
        with open(self.config.STATS_OUTPUT, "w") as f:
            json.dump(self.stats, f, indent=2, default=str)

        logger.info("\n" + "=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total records: {len(gdf):,}")
        logger.info(f"Unique incidents: {gdf['incident_id'].nunique():,}")
        logger.info(f"Output: {self.config.OUTPUT_PARQUET}")
        logger.info(f"Report: {self.config.STATS_OUTPUT}")

    def run(self):
        """Run the complete processing pipeline"""
        start_time = datetime.now()

        try:
            gdf = self.process_all_datasets()
            if not gdf.empty:
                self.save_output(gdf)
            else:
                logger.error("Processing produced no output!")
                return 1
        except Exception as e:
            logger.error(f"Processing failed: {e}", exc_info=True)
            return 1

        elapsed = datetime.now() - start_time
        logger.info(f"\nTotal processing time: {elapsed}")
        return 0


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point"""
    config = Config()
    processor = FirePerimeterProcessor(config)
    return processor.run()


if __name__ == "__main__":
    exit(main())
