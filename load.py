import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data_dir = "path of directory"

# Load metadata
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

# Function to read a single .svc file
def read_svc_file(svc_path):
    return pd.read_csv(
        svc_path, sep="\s+", header=None, skiprows=1,
        names=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure']
    )

# Function to load all .svc files into a dictionary of DataFrames
def load_svc_data(data_dir):
    svc_data = {}
    for collection in ["Collection1", "Collection2"]:
        user_dir = os.path.join(data_dir, collection)
        for user in os.listdir(user_dir):
            session_dir = os.path.join(user_dir, user, "session00001")
            if not os.path.exists(session_dir):
                continue
            svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
            for svc_file in svc_files:
                svc_path = os.path.join(session_dir, svc_file)
                df = read_svc_file(svc_path)
                key = f"{collection}/{user}/{svc_file}"
                svc_data[key] = df
    return svc_data
