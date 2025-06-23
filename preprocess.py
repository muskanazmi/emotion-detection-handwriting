def normalize_svc_data(svc_data):
    normalized_data = {}
    scaler = MinMaxScaler()

    for key, df in svc_data.items():
        df = df.copy()
        max_azimuth = df['azimuth_angle'].max() or 1  # avoid divide-by-zero
        max_altitude = df['altitude_angle'].max() or 1
        df['azimuth_angle'] = (df['azimuth_angle'] / max_azimuth) * 360
        df['altitude_angle'] = (df['altitude_angle'] / max_altitude) * 360
        df['pressure'] = scaler.fit_transform(df[['pressure']])
        normalized_data[key] = df

    return normalized_data
