COLUMN_METADATA = {
    # Soil Properties
    "Depth": {
        "description": "Soil sampling depth (e.g. '0-20 cm')",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "POINTID": {
        "description": "Unique identifier for sampling point",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "pH_CaCl₂": {
        "description": "Soil pH measured in CaCl₂ solution",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "pH_H₂O": {
        "description": "Soil pH measured in water solution",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "EC": {
        "description": "Electrical conductivity (dS/m)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "OC": {
        "description": "Organic carbon content (g/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "CaCO₃": {
        "description": "Calcium carbonate content (g/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "P": {
        "description": "Phosphorus content (mg/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "N": {
        "description": "Nitrogen content (mg/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "K": {
        "description": "Potassium content (mg/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "OC (20–30 cm)": {
        "description": "Organic carbon at 20-30cm depth (g/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "CaCO₃ (20–30 cm)": {
        "description": "Calcium carbonate at 20-30cm depth (g/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "Ox_Al": {
        "description": "Oxalate-extractable aluminum (mmol/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    "Ox_Fe": {
        "description": "Oxalate-extractable iron (mmol/kg)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    
    # Location Metadata
    "NUTS_0": {
        "description": "Country code (ISO 2-letter)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "NUTS_1": {
        "description": "Regional code (NUTS level 1)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "NUTS_2": {
        "description": "Sub-regional code (NUTS level 2)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "NUTS_3": {
        "description": "Local administrative code (NUTS level 3)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "TH_LAT": {
        "description": "Theoretical latitude (approximate)",
        "dtype": "float64",
        "dataset": ["csv"]
    },
    "TH_LONG": {
        "description": "Theoretical longitude (approximate)",
        "dtype": "float64",
        "dataset": ["csv"]
    },
    
    # Temporal and Spatial
    "SURVEY_DATE": {
        "description": "Date of sample collection (YYYY-MM-DD)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "Elev": {
        "description": "Elevation above sea level (meters)",
        "dtype": "float64",
        "dataset": ["csv", "shapefile"]
    },
    
    # Land Characteristics
    "LC": {
        "description": "Land cover code (EUROSTAT classification)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "LU": {
        "description": "Land use code (EUROSTAT classification)",
        "dtype": "object",
        "dataset": ["csv", "shapefile"]
    },
    "LC₀_Desc": {
        "description": "Land cover description (Level 0)",
        "dtype": "category",
        "dataset": ["csv", "shapefile"]
    },
    "LC₁_Desc": {
        "description": "Land cover description (Level 1)",
        "dtype": "category",
        "dataset": ["csv", "shapefile"]
    },
    "LU₁_Desc": {
        "description": "Land use description (Level 1)",
        "dtype": "category",
        "dataset": ["csv", "shapefile"]
    },
    
    # Spatial Data
    "geometry": {
        "description": "Precise spatial coordinates (Point geometry)",
        "dtype": "geometry",
        "dataset": ["shapefile"],
        "crs": "EPSG:3035"  # Updated CRS
    }
}

DATASET_WARNINGS = {
    "csv": [
        "Always use pandas to load the dataset"
        "LC₀_Desc uses subscript ₀ (not regular 0)",
        "Do not check for missing values - there aren't any"
        "Never create methods - just a set of commands, that are executed immediately"
    ],
    "shapefile": [
        "Never merge the European coastline shapefile and the geo_dataframe shapefile",
        "Always use geopandas to load the shapefiles separately"
        "Never create methods - just a set of commands, that are executed immediately",
        "Always plot the coastline in lightgrey",
        "Always plot soil points with marker type '.' and marker size 5",
        "Save plots as PNG using matplotlib.pyplot",
        "Never merge shapefiles with other datasets",
    ]
}