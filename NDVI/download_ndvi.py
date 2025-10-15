#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:36:12 2024

@author: melis
"""
import os
import requests
from bs4 import BeautifulSoup


#%% 2017
# Base URL of the NDVI data
base_url = "https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/"

# Function to download data for a specific year
def download_ndvi_data(year):
    year_url = f"{base_url}{year}/"
    download_directory = f"ndvi_data_{year}"
    os.makedirs(download_directory, exist_ok=True)

    response = requests.get(year_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            file_name = link.get('href')
            if file_name.endswith('.nc'):  # Only download .nc files
                file_url = f"{year_url}{file_name}"
                print(f"Downloading {file_url}")
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    file_path = os.path.join(download_directory, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(file_response.content)
                else:
                    print(f"Failed to download {file_name}")
        print(f"Download for {year} completed.")
    else:
        print(f"Failed to access {year_url}")

# Specify the year to download
year = 2018  # Change this to the desired year, e.g., 2018, 2019, etc.
#%%
# download_ndvi_data(year)

download_ndvi_data(2019)
download_ndvi_data(2020)
download_ndvi_data(2021)
download_ndvi_data(2022)
download_ndvi_data(2023)
download_ndvi_data(2024)


#%% Get the .xml data format for metadata

from netCDF4 import Dataset
import xml.etree.ElementTree as ET

def extract_nc_metadata_to_xml(nc_path, xml_out_path):
    ds = Dataset(nc_path, 'r')
    root = ET.Element("NetCDF_Metadata")

    # Global attributes
    global_attrs = ET.SubElement(root, "GlobalAttributes")
    for attr in ds.ncattrs():
        ET.SubElement(global_attrs, attr).text = str(getattr(ds, attr))

    # Dimensions
    dims = ET.SubElement(root, "Dimensions")
    for name, dim in ds.dimensions.items():
        dim_elem = ET.SubElement(dims, name)
        dim_elem.set("size", str(len(dim)))
        dim_elem.set("isunlimited", str(dim.isunlimited()))

    # Variables
    vars_elem = ET.SubElement(root, "Variables")
    for name, var in ds.variables.items():
        var_elem = ET.SubElement(vars_elem, name)
        var_elem.set("datatype", str(var.dtype))
        var_elem.set("dimensions", str(var.dimensions))
        for attr in var.ncattrs():
            ET.SubElement(var_elem, attr).text = str(getattr(var, attr))

    ds.close()

    # Save to XML
    tree = ET.ElementTree(root)
    tree.write(xml_out_path, encoding='utf-8', xml_declaration=True)

# Example usage
extract_nc_metadata_to_xml("/Users/melis/Desktop/NDVI_Download_09_22_2025/ndvi_data_2017/VIIRS-Land_v001_NPP13C1_S-NPP_20170101_c20240314150625.nc", "metadata.xml")

#%% convert .xml to .doc

from lxml import etree
from pathlib import Path

input_path = Path("/Users/melis/Desktop/NDVI_Download_09_22_2025/metadata.xml")
output_path = input_path.with_suffix(".txt")

# Parse with lxml and recover from errors
with open(input_path, 'rb') as f:
    parser = etree.XMLParser(recover=True)
    tree = etree.parse(f, parser)

# Pretty print to text
with open(output_path, "w", encoding="utf-8") as f:
    f.write(etree.tostring(tree, pretty_print=True, encoding='unicode'))

print(f"✅ Saved cleaned XML as: {output_path}")



#%% 2018
# Base URL of the NDVI data
base_url = "https://www.ncei.noaa.gov/data/land-normalized-difference-vegetation-index/access/"

# Function to download data for a specific year
def download_ndvi_data(year):
    year_url = f"{base_url}{year}/"
    download_directory = f"ndvi_data_{year}"
    os.makedirs(download_directory, exist_ok=True)

    response = requests.get(year_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        links = soup.find_all('a')

        for link in links:
            file_name = link.get('href')
            if file_name.endswith('.nc'):  # Only download .nc files
                file_url = f"{year_url}{file_name}"
                print(f"Downloading {file_url}")
                file_response = requests.get(file_url)
                if file_response.status_code == 200:
                    file_path = os.path.join(download_directory, file_name)
                    with open(file_path, 'wb') as file:
                        file.write(file_response.content)
                else:
                    print(f"Failed to download {file_name}")
        print(f"Download for {year} completed.")
    else:
        print(f"Failed to access {year_url}")

# Specify the year to download
year = 2018  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)

#%%

# Specify the year to download
year = 2019  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)



# Specify the year to download
year = 2020  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)



# Specify the year to download
year = 2021  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)



# Specify the year to download
year = 2022  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)



# Specify the year to download
year = 2023  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)



# Specify the year to download
year = 2024  # Change this to the desired year, e.g., 2018, 2019, etc.
download_ndvi_data(year)
