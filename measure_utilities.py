from pathlib import Path

import numpy as np
import pandas as pd


def measure_spherical_distance(point_a, point_b):
    # Convert decimal degrees to radians
    lat_a, lon_a = np.radians(point_a)
    lat_b, lon_b = np.radians(point_b)
    # Haversine formula
    d_lon, d_lat = lon_b - lon_a, lat_b - lat_a
    distance_km = 2 * np.arcsin(
        np.sqrt(np.sin(d_lat / 2) ** 2 + np.cos(lat_a) * np.cos(lat_b) * np.sin(d_lon / 2) ** 2)) * 6371
    return np.round(distance_km, 2)


def calculate_process(df: pd.DataFrame, filename: str,
                      addr: str = "Address", lat: str = "lat", lon: str = "lng") -> list:
    filepath = str(Path("input/external_data/", filename))
    df_external = pd.read_csv(filepath)
    df_external = df_external[[addr, lat, lon]]
    df_external.drop_duplicates(inplace=True)
    df_external[addr] = df_external[addr].apply(lambda x: x.replace('臺', '台'))

    distances_lst = []
    for row in df.itertuples():
        point_a = (getattr(row, lat), getattr(row, lon))
        distances = []
        df_core = df_external[df_external[addr] == getattr(row, addr)]
        for _, row_ext in df_core.iterrows():
            point_b = (row_ext[lat], row_ext[lon])
            dist = measure_spherical_distance(point_a, point_b)
            distances.append(dist)
        distances_lst.append(distances)

    return distances_lst


def process_facility(df: pd.DataFrame, facility: str):
    facility_distance = calculate_process(df, f'{facility}.csv')
    df[f'{facility}_distance'] = facility_distance
    df[f'num_{facility}'] = df[f'{facility}_distance'].apply(len)
    df[f'has_{facility}'] = df[f'num_{facility}'].apply(lambda x: x > 0)


def measure_utilities(df: pd.DataFrame) -> pd.DataFrame:
    facilities = [
        'ATM',
        'ConvenienceStore',
        'BusStop',
        'MetroStation',
        'TrainStation',
        'BicycleStation',
        'University',
        'SeniorHighSchool',
        'JuniorHighSchool',
        'ElementarySchool',
        'PostOffice',
        'FinancialInstitution',
        'MedicalInstitution',
    ]
    for facility in facilities:
        process_facility(df, facility)
    return df
