# Importing data

import pandas as pd
import os
import io
import re

# Set the data directory path

data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"


# Load the Excel file containing the metadata
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

def read_svc_file(svc_path):
    df = pd.read_csv(svc_path, sep="\s+", header=None, skiprows=1, names=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
    return df


# Loop through each user directory in Collection1 and Collection2
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            df = pd.DataFrame(df, columns=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
             
            
            print(df.head())
            print(df.tail())


# Normalizing the dataset

from sklearn.preprocessing import MinMaxScaler
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            
            # Normalize the azimuth and altitude columns using the maximum values as normalization factor
            
            max_azimuth = df['azimuth_angle'].max()
            max_altitude = df['altitude_angle'].max()
            df['azimuth_angle'] = df['azimuth_angle'] / max_azimuth* 360
            df['altitude_angle'] = df['altitude_angle'] / max_altitude* 360
            scaler = MinMaxScaler()
            df['pressure'] = scaler.fit_transform(df[['pressure']])
            
            print(df.head())
            print(df.tail())

# Checking null values

# Check for null values in metadata_df
print(metadata_df.isnull().any())

# Check for null values in df
print(df.isnull().any())


# Check for null values in df
print(df.isnull().any())


# Data_file

import pandas as pd
import os
import re
# Set the data directory path
#data_dir = "C:\\Users\\lubna\\Downloads\\DataEmothaw\\DataEmothaw\\"
data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"


# Load the Excel file containing the metadata
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

def read_svc_file(svc_path):
    df = pd.read_csv(svc_path, sep="\s+", header=None,skiprows=1, names=['x_position', 'y_position', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
    return df

# Initialize a counter for the number of users
user_counter = 0

# Initialize an empty list to hold the dataframes for each user
dfs = []

# Loop through each user directory in Collection1 and Collection2
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and add metadata information
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            df = pd.DataFrame(df, columns=['x_position', 'y_position', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
            # Extract metadata information
            file_number = int(re.search(r'\d+', svc_file).group())
            database_collectors= metadata_df.loc[metadata_df['File Number user'] == file_number, 'Database Collectors'].iloc[0]
            directory = metadata_df.loc[metadata_df['File Number user'] == file_number, 'Directory'].iloc[0]
            depression = metadata_df.loc[metadata_df['File Number user'] == file_number, 'depression'].iloc[0]
            anxiety = metadata_df.loc[metadata_df['File Number user'] == file_number, 'anxiety'].iloc[0]
            stress = metadata_df.loc[metadata_df['File Number user'] == file_number, 'stress'].iloc[0]
            # Add metadata information to dataframe
            df['Database Collectors'] = database_collectors
            df['File Number user'] = file_number
            df['Directory'] = directory
            df['depression'] = depression
            df['anxiety'] = anxiety
            df['stress'] = stress
            # Add dataframe to list
            dfs.append(df)

# Concatenate all dataframes into one
data = pd.concat(dfs, ignore_index=True)

# Merge with metadata dataframe
merged_df = pd.merge(data, metadata_df, on=['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'])

# Save merged dataframe as a CSV file
merged_df.to_csv(os.path.join(data_dir, "merged_data.csv"), index=False)



print(merged_df)

# Analyzing data

import matplotlib.pyplot as plt
# Initialize a counter for the number of users
user_counter = 0

# Loop through each user directory in Collection1 and Collection2
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        # Increment the user counter
        user_counter += 1
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            df = pd.DataFrame(df, columns=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])

            # Filter data by pen status and get statistics
            on_paper = df[df['pen_status'] == 1]
            in_air = df[df['pen_status'] == 0]
            on_paper_stats = on_paper.describe()
            in_air_stats = in_air.describe()

            # Print statistics
            print(f'Pen status of {svc_file}:')
            print(f'On paper:\n{on_paper_stats}')
            print(f'In air:\n{in_air_stats}')
           # Plot data
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(on_paper['x_pos'], on_paper['y_pos'], label='On paper', c='blue')
            ax.plot(in_air['x_pos'], in_air['y_pos'], label='In air', c='red')
            ax.legend()

            # Adjust x and y axis limits for u00001s00001_hw00007 file
            if svc_file == 'u00001s00001_hw00007.svc':
                ax.set_xlim([min(df['x_pos'])-100, max(df['x_pos'])+100])
                ax.set_ylim([min(df['y_pos'])-200, max(df['y_pos'])+200])
            else:
                ax.set_xlim([min(df['x_pos'])-10, max(df['x_pos'])+10])
                ax.set_ylim([min(df['y_pos'])-10, max(df['y_pos'])+10])
            ax.set_aspect('equal')
            plt.savefig(os.path.join(session_dir, f'{svc_file[:-4]}_penstatus.png'))
            plt.show()

# Print the total number of users
print(f"Total number of users: {user_counter}")


def compute_velocity(x_pos, y_pos, time_stamp):
    dt = diff_with_zero_first_element(time_stamp)
    dx = diff_with_zero_first_element(x_pos)
    dy = diff_with_zero_first_element(y_pos)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = np.divide(np.sqrt(dx**2 + dy**2), dt)
    v[np.isnan(v)] = 0  # Replace NaNs with zeros
    return v

def compute_acceleration(v, time_stamp):
    dt = diff_with_zero_first_element(time_stamp)
    dv = diff_with_zero_first_element(v)
    with np.errstate(divide='ignore', invalid='ignore'):
        a = np.divide(dv, dt)
    a[np.isnan(a)] = 0  # Replace NaNs with zeros
    return a
# Loop through each user directory in Collection1 and Collection2
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            
            # Compute pen_down column and stroke_id based on pen_down
            # Add a new column to indicate when the pen is down
            df['pen_down'] = df['pen_status'].apply(lambda x: True if x == 1 else False)
            # Compute stroke_id based on the pen_down column
            df['stroke_id'] = (df['pen_down'].diff() != 0).cumsum()
            # Compute time in seconds
            df['time'] = df['time_stamp'] / 1000

            df['x_pos'] = df['altitude_angle'] * np.sin(df['azimuth_angle'] * np.pi / 180)
            df['y_pos'] = df['altitude_angle'] * np.cos(df['azimuth_angle'] * np.pi / 180)
            df['velocity'] = compute_velocity(df['x_pos'], df['y_pos'], df['time'])
            df['acceleration'] = compute_acceleration(df['velocity'], df['time'])
            
            print(df.head())
            print(df.tail())

print(df.isnull().any())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            
            # Compute pen_down column and stroke_id based on pen_down
            # Add a new column to indicate when the pen is down
            df['pen_down'] = df['pen_status'].apply(lambda x: True if x == 0 else False)
            # Compute stroke_id based on the pen_down column
            df['stroke_id'] = (df['pen_down'].diff() != 0).cumsum()
            # Compute time in seconds
            df['time'] = df['time_stamp'] / 1000

            # Compute x and y positions, velocity, and acceleration
            df['x_pos'] = df['altitude_angle'] * np.sin(df['azimuth_angle'] * np.pi / 180)
            df['y_pos'] = df['altitude_angle'] * np.cos(df['azimuth_angle'] * np.pi / 180)
            df['velocity'] = compute_velocity(df['x_pos'], df['y_pos'], df['time'])
            df['acceleration'] = compute_acceleration(df['velocity'], df['time'])

            # Visualizations
            # Line plots to visualize the trajectory of the pen movements
            plt.plot(df['x_pos'], df['y_pos'])
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.title('Pen Trajectory')
            plt.show()

            # Histograms to show the distribution of velocity and acceleration
            fig, axs = plt.subplots(1, 2, figsize=(10, 4))
            axs[0].hist(df['velocity'], bins=50)
            axs[0].set_xlabel('Velocity')
            axs[0].set_ylabel('Count')
            axs[0].set_title('Velocity Distribution')
            axs[1].hist(df['acceleration'], bins=50)
            axs[1].set_xlabel('Acceleration')
            axs[1].set_ylabel('Count')
            
        stroke_lengths = df.groupby('stroke_id')['x_pos'].count()
        plt.boxplot(stroke_lengths)
        plt.ylabel('Stroke Length')
        plt.title('Distribution of Stroke Lengths')
        plt.show()

        # Summary statistics
        # Mean, median, and standard deviation of x and y positions
        x_mean = df['x_pos'].mean()
        x_median = df['x_pos'].median()
        x_std = df['x_pos'].std()
        y_mean = df['y_pos'].mean()
        y_median = df['y_pos'].median()
        y_std = df['y_pos'].std()
        print("X Position Mean:", x_mean)
        print("X Position Median:", x_median)
        print("X Position Standard Deviation:", x_std)
        print("Y Position Mean:", y_mean)
        print("Y Position Median:", y_median)
        print("Y Position Standard Deviation:", y_std)

        # Mean, median, and standard deviation of velocity and acceleration
        v_mean = df['velocity'].mean()
        v_median = df['velocity'].median()
        v_std = df['velocity'].std()
        a_mean = df['acceleration'].mean()
        a_median = df['acceleration'].median()
        a_std = df['acceleration'].std()
        print("Velocity Mean:", v_mean)
        print("Velocity Median:", v_median)
        print("Velocity Standard Deviation:", v_std)
        print("Acceleration Mean:", a_mean)
        print("Acceleration Median:", a_median)
        print("Acceleration Standard Deviation:", a_std)

        # Total number of strokes, total number of pen-down events, and total duration of writing
        num_strokes = df['stroke_id'].nunique()
        num_pen_down_events = df['pen_down'].sum()
        duration = df['time'].max() - df['time'].min()
        print("Number of Strokes:", num_strokes)
        print("Number of Pen-Down Events:", num_pen_down_events)
        print("Duration of Writing (seconds):", duration)


# plotting dass score

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

# Set the data directory path
data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"

# Load the Excel file containing the metadata
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

# Define function to read SVC file and handle decoding errors
def read_svc_file(svc_path):
    df = pd.read_csv(svc_path, sep="\s+", header=None, names=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
    df.drop(0, inplace=True)
    return df

# Define labels and ranges for depression, anxiety, and stress
depression_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
depression_ranges = [(0, 9), (10, 13), (14, 20), (21, 27), (28, np.inf)]

anxiety_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
anxiety_ranges = [(0, 7), (8, 9), (10, 14), (15, 19), (20, np.inf)]

stress_labels = ['Normal', 'Mild', 'Moderate', 'Severe', 'Extremely Severe']
stress_ranges = [(0, 7), (15, 18), (19, 25), (26, 33), (34, np.inf)]

# Initialize counters for depression, anxiety, and stress labels
depression_counts = [0] * len(depression_labels)
anxiety_counts = [0] * len(anxiety_labels)
stress_counts = [0] * len(stress_labels)

# Initialize dictionary to store user file counts and corresponding scores
# Loop through each collection and user directory and load 7 task files for each user
user_scores = {}
for collection in ["Collection1", "Collection2"]:
    collection_dir = os.path.join(data_dir, collection)
    for user in os.listdir(collection_dir):
        if not user.endswith(".xls"):
            user_dir = os.path.join(collection_dir, user)
            file_count = 0
            depression_scores = []
            anxiety_scores = []
            stress_scores = []
            for i in range(1, 8):
                task_dir = os.path.join(user_dir, f"session0000{i}")
                if os.path.exists(task_dir):
                    svc_files = [f for f in os.listdir(task_dir) if f.endswith(".svc")]
                    if len(svc_files) > 0:
                        file_count += len(svc_files)
                        for svc_file in svc_files:
                            # Load each SVC file as a dataframe and perform analysis
                            svc_path = os.path.join(task_dir, svc_file)
                            df = read_svc_file(svc_path)
                            # Get depression, anxiety, and stress values for the user from metadata_df
                            metadata_row = metadata_df.loc[metadata_df["File Number user"] == int(user[4:])]
                            depression = metadata_row["depression"].values[0]
                            anxiety = metadata_row["anxiety"].values[0]
                            stress = metadata_row["stress"].values[0]
                            # Add scores to corresponding lists
                            depression_scores.append(depression)
                            anxiety_scores.append(anxiety)
                            stress_scores.append(stress)
                        
            # Store user file count and corresponding scores in dictionary
            user_scores[user] = {'file_count': file_count, 'depression_scores': depression_scores, 
                                 'anxiety_scores': anxiety_scores, 'stress_scores': stress_scores}
        
# Plot histograms for depression, anxiety, and stress scores
fig, axs = plt.subplots(1, 3, figsize=(20, 5))
axs[0].hist([score for user in user_scores for score in user_scores[user]['depression_scores']], bins=20)
axs[0].set_title('Depression Scores')
axs[1].hist([score for user in user_scores for score in user_scores[user]['anxiety_scores']], bins=20)
axs[1].set_title('Anxiety Scores')
axs[2].hist([score for user in user_scores for score in user_scores[user]['stress_scores']], bins=20)
axs[2].set_title('Stress Scores')
plt.show()






# Feature Engineering

import numpy as np
import pandas as pd
import os

def compute_features(df):
    # Compute pen_down column and stroke_id based on pen_down
    # Add a new column to indicate when the pen is down
    df['pen_down'] = df['pen_status'].apply(lambda x: True if x == 1 else False)
    # Compute stroke_id based on the pen_down column
    df['stroke_id'] = (df['pen_down'].diff() != 0).cumsum()
    # Compute time in seconds
    df['time'] = df['time_stamp'] / 1000


    # Compute timing-based features
    #air_time = df[df['pen_down'] == False]['time'].max() - df[df['pen_down'] == False]['time'].min()
    #paper_time = df[df['pen_down'] == True]['time'].max() - df[df['pen_down'] == True]['time'].min()
    #paper_time = df.loc[df['pen_status'] == 1, 'time_stamp'].diff().loc[df['pen_status'] == 1].sum() / 1000
    paper_time = df[df['pen_down'] == True].groupby('stroke_id')['time'].apply(lambda x: x.max() - x.min()).sum()
    total_time = df['time'].max() - df['time'].min()
    air_time = (total_time - paper_time)
    # Compute ductus-based feature
    num_on_paper_strokes = df[df['pen_down'] == True]['stroke_id'].nunique()

    return paper_time, total_time, num_on_paper_strokes, air_time

for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and compute features
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
            paper_time, total_time, num_on_paper_strokes, air_time = compute_features(df)

            # Print the features for each SVC file
            print("SVC File:", svc_file)
            print("F1: Time spent in-air while completing the task = {:.2f} seconds".format(air_time))
            print("F2: Time spent on-paper while completing the task = {:.2f} seconds".format(paper_time))
            print("F3: Time to complete the whole task = {:.2f} seconds".format(total_time))
            print("F4: Number of on-paper strokes while completing the task = {}".format(num_on_paper_strokes))


# creating merged file and target columns

import csv
import numpy as np
import pandas as pd
import os

def compute_features(df):
    # Compute pen_down column and stroke_id based on pen_down
    # Add a new column to indicate when the pen is down
    df['pen_down'] = df['pen_status'].apply(lambda x: True if x == 1 else False)
    # Compute stroke_id based on the pen_down column
    df['stroke_id'] = (df['pen_down'].diff() != 0).cumsum()
    # Compute time in seconds
    df['time'] = df['time_stamp'] / 1000


     
    paper_time = df[df['pen_down'] == True].groupby('stroke_id')['time'].apply(lambda x: x.max() - x.min()).sum()
    total_time = df['time'].max() - df['time'].min()
    air_time = (total_time - paper_time)
    # Compute ductus-based feature
    num_on_paper_strokes = df[df['pen_down'] == True]['stroke_id'].nunique()

    return paper_time, total_time, num_on_paper_strokes, air_time

# Create a CSV file and write the header row
with open("features.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["SVC File", "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes"])

    for collection in ["Collection1", "Collection2"]:
        user_dir = os.path.join(data_dir, collection)
        for user in os.listdir(user_dir):
            session_dir = os.path.join(user_dir, user, "session00001")
            svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc") and f not in ["u00002s00001_hw00004.svc", "u00002s00001_hw00005.svc"]]
            for svc_file in svc_files:
                # Load each SVC file as a dataframe and compute features
                svc_path = os.path.join(session_dir, svc_file)
                df = read_svc_file(svc_path)
                paper_time, total_time, num_on_paper_strokes, air_time = compute_features(df)

                # Write the features to the CSV file
                writer.writerow([svc_file, air_time, paper_time, total_time, num_on_paper_strokes])


import pandas as pd

# Load the CSV file into a pandas dataframe
features_df = pd.read_csv('features.csv')

# Print the entire dataframe
print(features_df)

# Or print only the first few rows
print(features_df.head())


import pandas as pd
import os

# Read in the feature dataframe
feature_df = pd.read_csv('features.csv')

# Read in the metadata dataframe
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

# Create a dictionary to map file number to depression, anxiety, and stress scores
metadata_dict = {}

for index, row in metadata_df.iterrows():
    file_number = row['File Number user']
    depression = row['depression']
    anxiety = row['anxiety']
    stress = row['stress']
    
    if file_number not in metadata_dict:
        metadata_dict[file_number] = {'depression': depression, 'anxiety': anxiety, 'stress': stress}

# Assign the depression, anxiety, and stress values to each of the 7 tasks for each user in the feature dataframe
depression_list = []
anxiety_list = []
stress_list = []

for index, row in feature_df.iterrows():
    file_number = int(re.search(r'\d+', row['SVC File']).group())
    metadata_row = metadata_dict[file_number]
    
    depression_list.append(metadata_row['depression'])
    anxiety_list.append(metadata_row['anxiety'])
    stress_list.append(metadata_row['stress'])

# Create new dataframe with 7 copies of each row in the feature dataframe
Merged_df = pd.concat([feature_df], ignore_index=True)

# Add the depression, anxiety, and stress columns to the merged dataframe
Merged_df['depression'] = depression_list 
Merged_df['anxiety'] = anxiety_list 
Merged_df['stress'] = stress_list 

# Write the merged dataframe to a new csv file
Merged_df.to_csv('merged_features_metadata.csv', index=False)
print(Merged_df)



# Read in the merged dataset
merged_df = pd.read_csv('merged_features_metadata.csv')

# Concatenate the merged dataset with itself
d_df = pd.concat([merged_df, merged_df], ignore_index=True)

# Write the doubled dataset to a new csv file
d_df.to_csv('d_features_metadata.csv', index=False)

# Print the first few rows of the doubled dataset
print(d_df.head())


# Correlation Plot

import matplotlib.pyplot as plt

# Calculate the correlation matrix between all columns
corr_matrix = Merged_df.corr()

# Select the correlation values for the "depression" column
depression_corr = corr_matrix["depression"].drop("depression")
depression_corr = corr_matrix["depression"].drop(['depression','stress', 'anxiety'])

# Plot a bar graph of the correlation values
plt.bar(x=depression_corr.index, height=depression_corr.values)
plt.xticks(rotation=90)
#plt.xlabel("Feature")
plt.xlabel([ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes"])
plt.ylabel("Correlation with Depression")
plt.title("Correlation between Depression and Features")
plt.show()

import matplotlib.pyplot as plt

# Calculate the correlation matrix between all columns
corr_matrix = Merged_df.corr()

# Select the correlation values for the "depression" column
stress_corr = corr_matrix["stress"].drop("stress")
stress_corr = corr_matrix["stress"].drop(['depression','stress', 'anxiety'])

# Plot a bar graph of the correlation values
plt.bar(x=stress_corr.index, height=stress_corr.values)
plt.xticks(rotation=90)
#plt.xlabel("Feature")
plt.xlabel([ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes"])
plt.ylabel("Correlation with stress")
plt.title("Correlation between stress and Features")
plt.show()

import matplotlib.pyplot as plt

# Calculate the correlation matrix between all columns
corr_matrix = Merged_df.corr()

# Select the correlation values for the "depression" column
anxiety_corr = corr_matrix["anxiety"].drop("anxiety")
anxiety_corr = corr_matrix["anxiety"].drop(['depression','stress', 'anxiety'])

# Plot a bar graph of the correlation values
plt.bar(x=anxiety_corr.index, height=anxiety_corr.values)
plt.xticks(rotation=90)
#plt.xlabel("Feature")
plt.xlabel([ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes"])
plt.ylabel("Correlation with anxiety")
plt.title("Correlation between anxiety and Features")
plt.show()

# Univariant analysis

import pandas as pd
import numpy as np
import seaborn as sns

# Load data
#data = pd.read_csv("C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw\\Merged_data.csv")
data= pd.read_csv('doubled_features_metadata.csv')
# Extract target variable
target = data["depression"]

# Compute correlation matrix between each feature and target
corr_matrix = data.corrwith(target).drop(['depression', 'anxiety','stress'])

# Sort features by correlation with target
sorted_features = corr_matrix.sort_values(ascending=False)

# Plot correlation heatmap
sns.heatmap(data[sorted_features.index].corr(), cmap="coolwarm", annot=True)

# Print top 10 positively correlated features
print(sorted_features[:7])

# Print top 10 negatively correlated features
print(sorted_features[-7:])


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('d_features_metadata.csv')

# Separate features and target variables
X = df[[ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes" ]]
#X = df.drop[[ 'depression', 'stress', 'anxiety']]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y = df['depression']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=5, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict the emotions on the test data
y_pred = clf.predict(X_test)

print(np.array(X_train))
print(np.array(y_train))
 
# Print classification report and confusion matrix
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

print(confusion_matrix(y_test, y_pred))
print('Accuracy for depression:', accuracy_score(y_test, y_pred))
train_acc = clf.score(X_train, y_train)
#print('Training Accuracy for depression:', train_acc)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('d_features_metadata.csv')

# Separate features and target variables
X = df[[ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes" ]]
#X = df.drop[[ 'depression', 'stress', 'anxiety']]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y = df['stress']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=8, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict the emotions on the test data
y_pred = clf.predict(X_test)

print(np.array(X_train))
print(np.array(y_train))
 
# Print classification report and confusion matrix
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

print(confusion_matrix(y_test, y_pred))
print('Accuracy for stress:', accuracy_score(y_test, y_pred))
train_acc = clf.score(X_train, y_train)
#print('Training Accuracy for stress:', train_acc)


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

#from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('d_features_metadata.csv')

# Separate features and target variables
X = df[[ "Air Time (seconds)", "Paper Time (seconds)", "Total Time (seconds)", "Number of on-paper strokes" ]]
#X = df.drop[[ 'depression', 'stress', 'anxiety']]
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y = df['anxiety']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Create a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=7, random_state=42)

# Train the model on the training data
clf.fit(X_train, y_train)

# Predict the emotions on the test data
y_pred = clf.predict(X_test)

print(np.array(X_train))
print(np.array(y_train))
 
# Print classification report and confusion matrix
#print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=1))

print(confusion_matrix(y_test, y_pred))
print('Accuracy for anxiety:', accuracy_score(y_test, y_pred))
train_acc = clf.score(X_train, y_train)
#print('Training Accuracy for anxiety:', train_acc)


# Dimensionality Reduction using LDA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Load the merged data as a pandas dataframe
#data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"
merged_df = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Define the feature and target variables
X = merged_df.drop(['depression', 'anxiety', 'stress','Database Collectors','File Number user',  'Directory' ], axis=1)
y = merged_df['depression']  # or 'anxiety' or 'stress'

# Instantiate an LDA object and fit the data
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Add the LDA components to the dataframe
merged_df['LDA1'] = X_lda[:, 0]
merged_df['LDA2'] = X_lda[:, 1]

# Print the updated dataframe
print(merged_df.head())
print(merged_df.tail())


import matplotlib.pyplot as plt

# Create a scatter plot of the LDA components
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA Scatter Plot')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the feature and target variables using the LDA components
X = merged_df[['LDA1', 'LDA2']]
y = merged_df['depression']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a random forest classifier and fit the training data
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

# Predict the depression values for the testing data
y_pred = rfc.predict(X_test)

# Evaluate the performance of the classifier
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

# Predict the depression values for the training data
y_train_pred = rfc.predict(X_train)

# Evaluate the performance of the classifier on the training data
print('Training accuracy:', accuracy_score(y_train, y_train_pred))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Load the merged data as a pandas dataframe
#data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"
merged_df = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Define the feature and target variables
X = merged_df.drop(['depression', 'anxiety', 'stress','Database Collectors','File Number user',  'Directory' ], axis=1)
y = merged_df['stress']  # or 'anxiety' or 'stress'

# Instantiate an LDA object and fit the data
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Add the LDA components to the dataframe
merged_df['LDA1'] = X_lda[:, 0]
merged_df['LDA2'] = X_lda[:, 1]

# Print the updated dataframe
print(merged_df.head())
print(merged_df.tail())


import matplotlib.pyplot as plt

# Create a scatter plot of the LDA components
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA Scatter Plot')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the feature and target variables using the LDA components
X = merged_df[['LDA1', 'LDA2']]
y = merged_df['stress']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a random forest classifier and fit the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict the depression values for the testing data
y_pred = rfc.predict(X_test)

# Evaluate the performance of the classifier
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

# Predict the depression values for the training data
y_train_pred = rfc.predict(X_train)

# Evaluate the performance of the classifier on the training data
print('Training accuracy:', accuracy_score(y_train, y_train_pred))



from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Load the merged data as a pandas dataframe
#data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"
merged_df = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Define the feature and target variables
X = merged_df.drop(['depression', 'anxiety', 'stress','Database Collectors','File Number user',  'Directory' ], axis=1)
y = merged_df['anxiety']  # or 'anxiety' or 'stress'

# Instantiate an LDA object and fit the data
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

# Add the LDA components to the dataframe
merged_df['LDA1'] = X_lda[:, 0]
merged_df['LDA2'] = X_lda[:, 1]

# Print the updated dataframe
print(merged_df.head())
print(merged_df.tail())


import matplotlib.pyplot as plt

# Create a scatter plot of the LDA components
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y)
plt.xlabel('LDA 1')
plt.ylabel('LDA 2')
plt.title('LDA Scatter Plot')
plt.show()


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Define the feature and target variables using the LDA components
X = merged_df[['LDA1', 'LDA2']]
y = merged_df['anxiety']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate a random forest classifier and fit the training data
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Predict the depression values for the testing data
y_pred = rfc.predict(X_test)

# Evaluate the performance of the classifier
from sklearn.metrics import accuracy_score
print('Accuracy:', accuracy_score(y_test, y_pred))

# Predict the depression values for the training data
y_train_pred = rfc.predict(X_train)

# Evaluate the performance of the classifier on the training data
print('Training accuracy:', accuracy_score(y_train, y_train_pred))



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#data = pd.read_csv('merged_data.csv')
data = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))
X = data.drop(['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'], axis=1)
y = data[['depression', 'anxiety', 'stress']]
X = (X - np.mean(X)) / np.std(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], y['depression'], c=y['depression'], cmap=plt.cm.jet)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('Depression')
plt.colorbar(scatter)
plt.show()


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the merged data
data = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Prepare the data
X = data.drop(['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'], axis=1)
y = data[['depression', 'anxiety', 'stress']]
X = (X - np.mean(X)) / np.std(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Prepare the target variable for depression classification
depression_labels = np.where(y['depression'] > 9,1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, depression_labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Calculate the percentage of depressed and non-depressed subjects in the test set
num_depressed = np.sum(y_test)
num_non_depressed = len(y_test) - num_depressed
pct_depressed = (num_depressed / len(y_test)) * 100
pct_non_depressed = 100 - pct_depressed
print("Percentage of depressed subjects: {:.2f}%".format(pct_depressed))
print("Percentage of non-depressed subjects: {:.2f}%".format(pct_non_depressed))


import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the merged data
data = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Prepare the data
X = data.drop(['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'], axis=1)
y = data[['depression', 'anxiety', 'stress']]
X = (X - np.mean(X)) / np.std(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Prepare the target variable for depression classification
depression_labels = np.where(y['stress'] > 14, 1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, depression_labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Calculate the percentage of depressed and non-depressed subjects in the test set
num_depressed = np.sum(y_test)
num_non_depressed = len(y_test) - num_depressed
pct_depressed = (num_depressed / len(y_test)) * 100
pct_non_depressed = 100 - pct_depressed
print("Percentage of stressed subjects: {:.2f}%".format(pct_depressed))
print("Percentage of non-stressed subjects: {:.2f}%".format(pct_non_depressed))


import pandas as pd
import numpy as np
import os
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import os
import io
import re

# Set the data directory path
#data_dir = "C:\\Users\\lubna\\Downloads\\DataEmothaw\\DataEmothaw\\"
data_dir = "C:\\Users\\Dell\\Downloads\\AllData\\DataEmothaw"


# Load the Excel file containing the metadata
metadata_df = pd.read_excel(os.path.join(data_dir, "DASS_scores.xls"))

def read_svc_file(svc_path):
    df = pd.read_csv(svc_path, sep="\s+", header=None, skiprows=1, names=['x_pos', 'y_pos', 'time_stamp', 'pen_status', 'azimuth_angle', 'altitude_angle', 'pressure'])
    return df


# Loop through each user directory in Collection1 and Collection2
for collection in ["Collection1", "Collection2"]:
    user_dir = os.path.join(data_dir, collection)
    for user in os.listdir(user_dir):
        session_dir = os.path.join(user_dir, user, "session00001")
        svc_files = [f for f in os.listdir(session_dir) if f.endswith(".svc")]
        for svc_file in svc_files:
            # Load each SVC file as a dataframe and perform analysis
            svc_path = os.path.join(session_dir, svc_file)
            df = read_svc_file(svc_path)
# Load the merged data
data = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Prepare the data
X = data.drop(['Database Collectors', 'File Number user', 'Directory', 'depression', 'anxiety', 'stress'], axis=1)
y = data[['depression', 'anxiety', 'stress']]
X = (X - np.mean(X)) / np.std(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Prepare the target variable for depression classification
depression_labels = np.where(y['anxiety'] > 20,1, 0)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, depression_labels, test_size=0.2, random_state=42)

# Train a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rfc.predict(X_test)

# Evaluate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

# Calculate the percentage of depressed and non-depressed subjects in the test set
num_depressed = np.sum(y_test)
num_non_depressed = len(y_test) - num_depressed
pct_depressed = (num_depressed / len(y_test)) * 100
pct_non_depressed = 100 - pct_depressed
print("Percentage of anxious subjects: {:.2f}%".format(pct_depressed))
print("Percentage of non-anxious subjects: {:.2f}%".format(pct_non_depressed))


from sklearn.ensemble import RandomForestClassifier

# Load the merged data as a pandas dataframe
merged_df = pd.read_csv(os.path.join(data_dir, "merged_data.csv"))

# Define the feature and target variables
X = merged_df.drop(['depression', 'anxiety', 'stress','Database Collectors','File Number user',  'Directory' ,'Suject'], axis=1)
y = merged_df[['depression', 'anxiety', 'stress']]

# Build a random forest model for each emotion
for emotion in ['depression', 'anxiety', 'stress']:
    # Instantiate a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42)

    # Fit the random forest model to the data
    rf.fit(X, y[emotion])

    # Print the feature importances
    print("Feature importances for", emotion)
    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print(importances.head(10))


