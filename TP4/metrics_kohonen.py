import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# for filtered data
def filtered_metrics():
    # Data Loading
    file_id = '1alJ5jLaPY8mfmb0MLoh5-bwwrH45pp2A'
    direct_link = f'https://drive.google.com/uc?id={file_id}'
    movie_data = pd.read_csv(direct_link, delimiter=';')

    # Define the genres of interest
    selected_genres = ['Action', 'Comedy', 'Drama']

    # Filter the dataset to include only movies with the selected genres
    filtered_movie_data = movie_data[movie_data['genres'].isin(selected_genres)]

    # Check the first few rows of the filtered dataset
    filtered_movie_data.head()

    # Identifying Numerical Columns
    numerical_cols = movie_data.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing Pipeline for Numerical Features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Splitting the filtered data into training and test sets
    train_data_filtered, test_data_filtered = train_test_split(filtered_movie_data, test_size=0.1, random_state=42)

    # Fit and transform the preprocessing pipeline to the filtered training data
    processed_train_data_filtered = numerical_pipeline.fit_transform(train_data_filtered[numerical_cols])

    # Transform the filtered test data with the same pipeline
    processed_test_data_filtered = numerical_pipeline.transform(test_data_filtered[numerical_cols])

    # Convert the processed data back to DataFrames
    processed_train_df_filtered = pd.DataFrame(processed_train_data_filtered, columns=numerical_cols)
    processed_test_df_filtered = pd.DataFrame(processed_test_data_filtered, columns=numerical_cols)

    # Initialize the SOM with the filtered data's dimensions
    num_features_filtered = processed_train_df_filtered.shape[1]
    som_filtered = SOM(height=10, width=10, input_dim=num_features_filtered)

    """ # Train the SOM
    print("Training the SOM on filtered data...")
    som_filtered.train(processed_train_df_filtered.to_numpy(), num_epochs=100, learning_rate=0.5, radius=3)

    # Optionally, save the trained SOM model
    with open('trained_som_filtered_5x5.pkl', 'wb') as file:
        pickle.dump(som_filtered, file) """

    # Load the trained SOM model
    with open('trained_som.pkl', 'rb') as file:
        som = pickle.load(file)

    # Generating and Visualizing U-Matrix
    u_matrix = som.create_u_matrix()
    plt.imshow(u_matrix, cmap='gray')
    plt.colorbar()
    plt.title('U-Matrix')
    plt.show()

    plt.figure(figsize=(10, 10))
    plt.imshow(u_matrix, cmap='coolwarm', interpolation='bilinear')  # 'coolwarm', 'viridis', 'plasma', etc.
    plt.colorbar()
    plt.title('U-Matrix')
    plt.show()

    # Function to find BMU for each movie
    def find_bmu_for_each_movie(som, data, genres):
        bmus = []
        for i, input_vec in enumerate(data):
            bmu_idx = som.find_bmu(input_vec)
            bmus.append((bmu_idx, genres.iloc[i]))
        return bmus

    # Function to plot genre distribution on SOM
    def plot_genre_distribution_on_som(som, bmus, title='SOM Genre Distribution'):
        plt.figure(figsize=(10, 10))
        plt.title(title)
        for bmu, genre in bmus:
            x, y = bmu
            if 'Action' in genre:
                color = 'r'
            elif 'Comedy' in genre:
                color = 'g'
            elif 'Drama' in genre:
                color = 'b'
            else:
                color = 'k'  # for other genres
            plt.plot(x, y, 'o', color=color, markersize=4)
        plt.xlim([0, som.width])
        plt.ylim([0, som.height])
        plt.xlabel('SOM X')
        plt.ylabel('SOM Y')
        plt.show()

    # Extract genres from the training data
    train_genres = train_data_filtered['genres']  # Adjust this if the column name for genres is different

    # Find BMUs for each movie
    bmus = find_bmu_for_each_movie(som, processed_train_df_filtered.to_numpy(), train_genres)

    def create_density_map(som, bmus, genre, som_dimensions):
        density_map = np.zeros(som_dimensions)
        for bmu, movie_genre in bmus:
            if genre in movie_genre:
                density_map[bmu] += 1
        return density_map

    # SOM dimensions
    som_dimensions = (som.height, som.width)

    # Create density maps for each genre
    density_map_action = create_density_map(som, bmus, 'Action', som_dimensions)
    density_map_comedy = create_density_map(som, bmus, 'Comedy', som_dimensions)
    density_map_drama = create_density_map(som, bmus, 'Drama', som_dimensions)

    from scipy.ndimage import gaussian_filter

    # Apply Gaussian filter for smoothing
    smoothed_density_action = gaussian_filter(density_map_action, sigma=1)
    smoothed_density_comedy = gaussian_filter(density_map_comedy, sigma=1)
    smoothed_density_drama = gaussian_filter(density_map_drama, sigma=1)

    plt.figure(figsize=(15, 5))

    # Plotting Action genre density map
    plt.subplot(1, 3, 1)
    plt.imshow(smoothed_density_action, cmap='Reds', interpolation='none')
    plt.title('Action Genre Density')
    plt.colorbar()

    # Plotting Comedy genre density map
    plt.subplot(1, 3, 2)
    plt.imshow(smoothed_density_comedy, cmap='Greens', interpolation='none')
    plt.title('Comedy Genre Density')
    plt.colorbar()

    # Plotting Drama genre density map
    plt.subplot(1, 3, 3)
    plt.imshow(smoothed_density_drama, cmap='Blues', interpolation='none')
    plt.title('Drama Genre Density')
    plt.colorbar()

    plt.show()

# for unfiltered data
def unfiltered_metrics():
    # Data Loading
    file_id = '1alJ5jLaPY8mfmb0MLoh5-bwwrH45pp2A'
    direct_link = f'https://drive.google.com/uc?id={file_id}'
    movie_data = pd.read_csv(direct_link, delimiter=';')

    # Identifying Numerical Columns
    numerical_cols = movie_data.select_dtypes(include=['int64', 'float64']).columns

    # Preprocessing Pipeline for Numerical Features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Splitting the data into training and test sets
    train_data, test_data = train_test_split(movie_data, test_size=0.5, random_state=42)

    # Fit and transform the preprocessing pipeline to the training data
    processed_train_data = numerical_pipeline.fit_transform(train_data[numerical_cols])

    # Transform the test data with the same pipeline
    processed_test_data = numerical_pipeline.transform(test_data[numerical_cols])

    # Convert the processed data back to DataFrames
    processed_train_df = pd.DataFrame(processed_train_data, columns=numerical_cols)
    processed_test_df = pd.DataFrame(processed_test_data, columns=numerical_cols)

    # Initialize the SOM with the data's dimensions
    num_features = processed_train_df.shape[1]
    som = SOM(height=10, width=10, input_dim=num_features)

    # Train the SOM on the entire data
    print("Training the SOM...")
    som.train(processed_train_df.to_numpy(), num_epochs=20, learning_rate=0.5, radius=3)

    # Save the trained SOM model (optional)
    with open('trained_som_10x10.pkl', 'wb') as file:
        pickle.dump(som, file)

    """ # Load the trained SOM model
    with open('trained_som.pkl', 'rb') as file:
        som = pickle.load(file) """

    # Generating and Visualizing U-Matrix
    u_matrix = som.create_u_matrix()
    plt.imshow(u_matrix, cmap='gray')
    plt.colorbar()
    plt.title('U-Matrix')
    plt.show()