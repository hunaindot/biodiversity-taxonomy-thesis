import os
import zipfile
import pandas as pd
import sqlite3
import xml.etree.ElementTree as ET
import sys

def extract_dwc_archive(zip_path, extract_to):
    if not os.path.exists(zip_path):
        print("GBIF taxonomy backbone zip file not found.")
        print("Please download it manually from:")
        print("URL: https://www.gbif.org/dataset/d7dddbf4-2cf0-4f39-9b2a-bb099caae36c#dataDescription")
        print("Direct download link: https://hosted-datasets.gbif.org/datasets/backbone/current/backbone.zip")
        print("Save the file to: gbif_taxonomy/source_files/backbone.zip")
        sys.exit(1)

    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"Extraction folder '{extract_to}' already exists and is not empty. Skipping extraction.")
        return extract_to
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted Darwin Core archive to: {extract_to}")
    return extract_to

def parse_meta_xml(meta_path):
    tree = ET.parse(meta_path)
    root = tree.getroot()
    core = root.find('{http://rs.tdwg.org/dwc/text/}core')

    location_element = core.find('{http://rs.tdwg.org/dwc/text/}files/{http://rs.tdwg.org/dwc/text/}location')
    file_location = location_element.text.strip()

    fields = core.findall('{http://rs.tdwg.org/dwc/text/}field')
    field_names = [field.attrib['term'].split('/')[-1] for field in fields]

    id_field = core.find('{http://rs.tdwg.org/dwc/text/}id')
    if id_field is not None:
        id_index = int(id_field.attrib['index'])
        field_names.insert(id_index, 'taxonID')  # Ensure ID column is preserved

    return file_location, field_names

def load_taxon_data(folder_path):
    meta_path = os.path.join(folder_path, 'meta.xml')
    data_file, columns = parse_meta_xml(meta_path)
    file_path = os.path.join(folder_path, data_file)

    try:
        df = pd.read_csv(
            file_path,
            sep='\t',
            dtype=str,
            low_memory=False,
            on_bad_lines='skip'
        )
        print(f"Loaded {len(df)} rows from {data_file}")
    except Exception as e:
        print(f"Failed to load data: {e}")
        raise
    return df

def save_to_sqlite(df, db_path, table_name="taxon"):
    if os.path.exists(db_path):
        print(f"Database already exists at '{db_path}'. Skipping save.")
        return
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved {len(df)} records to SQLite database: {db_path}")

def load_taxonomy(
        zip_path = "gbif_taxonomy/source_files/backbone.zip", 
        extract_to = "gbif_taxonomy/extracted", 
        db_path= "gbif_taxonomy/database/taxon.db"
        ):
    extracted_folder = extract_dwc_archive(zip_path, extract_to)
    df = load_taxon_data(extracted_folder)
    save_to_sqlite(df, db_path)
    print(f"SQLite database is available at '{db_path}' for use.")