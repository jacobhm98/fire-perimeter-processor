# fire-perimeter-processor

This project takes a set of historical fire perimeters from the [NIFC open data](https://data-nifc.opendata.arcgis.com/) website, and processes them into a format more easily usable for machine learning applications.

For more detail about each of the datasets mentioned here, please refer to [WFIGS hub](https://wfigs-nifc.hub.arcgis.com/) and the [FAQ](data-nifc.opendata.arcgis.com/pages/faqs)

## Obtain the datasets

Download each of the following datasets in shapefile format from the https://data-nifc.opendata.arcgis.com website:

- WFIGS Interagency Fire Perimeters
- Historic Perimeters Combined 2000-2018 GeoMAC
- InterAgencyFirePerimeterHistory All Years View
- WFIGS 2025 Interagency Fire Perimeters to Date
- WFIGS Current Interagency Fire Perimeters

Put the .zip files into the /data/raw directory, create the dirs if they don't already exist.

To extract the raw shapefiles:

```bash
./unzip_data.sh
```

