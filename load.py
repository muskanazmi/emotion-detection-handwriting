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


def load_and_merge_svc_with_metadata(data_dir, metadata_df):
    user_counter = 0
    dfs = []

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
                df.columns = ['x_position', 'y_position', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure']

                # Extract the file number from the filename
                match = re.search(r'\d+', svc_file)
                if not match:
                    continue
                file_number = int(match.group())

                # Filter metadata row
                matched_row = metadata_df[metadata_df['File Number user'] == file_number]
                if matched_row.empty:
                    continue

                # Add metadata to df
                df['Database Collectors'] = matched_row['Database Collectors'].iloc[0]
                df['File Number user'] = file_number
                df['Directory'] = matched_row['Directory'].iloc[0]
                df['depression'] = matched_row['depression'].iloc[0]
                df['anxiety'] = matched_row['anxiety'].iloc[0]
                df['stress'] = matched_row['stress'].iloc[0]

                dfs.append(df)

    # Combine all individual user data
    data = pd.concat(dfs, ignore_index=True)

    # Merge again (optional if already merged above)
    merged_df = pd.merge(
        data, metadata_df,
        on=['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'],
        how='left'
    )

    # Save as CSV
    merged_path = os.path.join(data_dir, "merged_data.csv")
    merged_df.to_csv(merged_path, index=False)

    return merged_df
