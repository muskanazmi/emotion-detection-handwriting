def plot_velocity_acceleration_histograms(df, svc_file, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(df['velocity'], bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Velocity Distribution')
    axs[0].set_xlabel('Velocity')
    axs[0].set_ylabel('Count')

    axs[1].hist(df['acceleration'], bins=50, color='salmon', edgecolor='black')
    axs[1].set_title('Acceleration Distribution')
    axs[1].set_xlabel('Acceleration')
    axs[1].set_ylabel('Count')

    plt.suptitle(f"Histograms - {svc_file}")
    if save_path:
        plt.savefig(os.path.join(save_path, f'{svc_file[:-4]}_histograms.png'))
    plt.show()

def plot_velocity_acceleration_histograms(df, svc_file, save_path=None):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].hist(df['velocity'], bins=50, color='skyblue', edgecolor='black')
    axs[0].set_title('Velocity Distribution')
    axs[0].set_xlabel('Velocity')
    axs[0].set_ylabel('Count')

    axs[1].hist(df['acceleration'], bins=50, color='salmon', edgecolor='black')
    axs[1].set_title('Acceleration Distribution')
    axs[1].set_xlabel('Acceleration')
    axs[1].set_ylabel('Count')

    plt.suptitle(f"Histograms - {svc_file}")
    if save_path:
        plt.savefig(os.path.join(save_path, f'{svc_file[:-4]}_histograms.png'))
    plt.show()

def plot_stroke_length_boxplot(df, svc_file, save_path=None):
    stroke_lengths = df.groupby('stroke_id')['x_pos'].count()
    plt.figure(figsize=(6, 5))
    plt.boxplot(stroke_lengths)
    plt.title(f'Stroke Lengths - {svc_file}')
    plt.ylabel('Stroke Length')
    if save_path:
        plt.savefig(os.path.join(save_path, f'{svc_file[:-4]}_boxplot.png'))
    plt.show()

def print_summary_statistics(df):
    def stat_report(col):
        return {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std()
        }

    print("X Position:", stat_report('x_pos'))
    print("Y Position:", stat_report('y_pos'))
    print("Velocity:", stat_report('velocity'))
    print("Acceleration:", stat_report('acceleration'))

    num_strokes = df['stroke_id'].nunique()
    num_pen_down_events = df['pen_down'].sum()
    duration = df['time'].max() - df['time'].min()

    print(f"Number of Strokes: {num_strokes}")
    print(f"Pen-Down Events: {num_pen_down_events}")
    print(f"Duration: {duration:.2f} seconds")

def analyze_svc_file(df, svc_file, save_path=None):
    plot_pen_trajectory(df, svc_file, save_path)
    plot_velocity_acceleration_histograms(df, svc_file, save_path)
    plot_stroke_length_boxplot(df, svc_file, save_path)
    print_summary_statistics(df)

def plot_dass_score_histograms(user_scores):
    depression_all = [score for user in user_scores for score in user_scores[user]['depression_scores']]
    anxiety_all = [score for user in user_scores for score in user_scores[user]['anxiety_scores']]
    stress_all = [score for user in user_scores for score in user_scores[user]['stress_scores']]

    fig, axs = plt.subplots(1, 3, figsize=(20, 5))
    
    axs[0].hist(depression_all, bins=20, color='lightblue', edgecolor='black')
    axs[0].set_title('Depression Scores')
    axs[0].set_xlabel('Score')
    axs[0].set_ylabel('Count')

    axs[1].hist(anxiety_all, bins=20, color='lightgreen', edgecolor='black')
    axs[1].set_title('Anxiety Scores')
    axs[1].set_xlabel('Score')
    axs[1].set_ylabel('Count')

    axs[2].hist(stress_all, bins=20, color='salmon', edgecolor='black')
    axs[2].set_title('Stress Scores')
    axs[2].set_xlabel('Score')
    axs[2].set_ylabel('Count')

    plt.tight_layout()
    plt.show()

