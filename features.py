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

def compute_features(df):
    # Compute pen_down column and stroke_id based on pen_down
    # Add a new column to indicate when the pen is down
    df['pen_down'] = df['pen_status'].apply(lambda x: True if x == 1 else False)
    # Compute stroke_id based on the pen_down column
    df['stroke_id'] = (df['pen_down'].diff() != 0).cumsum()
    # Compute time in seconds
    df['time'] = df['time_stamp'] / 1000

    # Compute timing-based features
  
    paper_time = df[df['pen_down'] == True].groupby('stroke_id')['time'].apply(lambda x: x.max() - x.min()).sum()
    total_time = df['time'].max() - df['time'].min()
    air_time = (total_time - paper_time)
    # Compute ductus-based feature
    num_on_paper_strokes = df[df['pen_down'] == True]['stroke_id'].nunique()

    return paper_time, total_time, num_on_paper_strokes, air_time
