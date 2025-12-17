# This file contains helper functions used in the main notebook.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

def clean_data(df):
    
    #Cleans the input DataFrame by handling missing values and duplicates.
    
    #Parameters:
    #df (pd.DataFrame): The input pandas DataFrame to be cleaned.
    
    #Returns:
    #pd.DataFrame: A new DataFrame with duplicates removed and missing values reported.
                  # This function primarily reports missing values; further steps (like dropna)
                  # are done after encoding for a fully clean, numerical dataset.
    
    print("Cleaning dataset...")
    
    # Display missing values before any action
    missing_values = df.isnull().sum()
    print(f"Missing values per column:\n{missing_values[missing_values > 0]}")

    # Handle duplicates
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    duplicates_removed = initial_rows - df.shape[0]
    print(f"Removed {duplicates_removed} duplicate rows")

    return df.copy()

def encode_features(df):
    
    #Encodes categorical (non-numeric) features in the DataFrame into numerical representations.
    #Machine learning models require numerical input, so this step is crucial for text-based categories.

    #Parameters:
    #df (pd.DataFrame): The DataFrame containing features, some of which may be categorical.

    #Returns:
    #pd.DataFrame: A new DataFrame with all specified categorical features encoded.
    
    print("\nEncoding categorical features...")
    df_encoded = df.copy()
    
    # List of categorical columns to encode.
    categorical_cols = ['GADE', 'Game', 'earnings', 'whyplay', 'streams', 'Gender', 'Work', 'Playstyle']

    for col in categorical_cols:
        if col in df_encoded.columns and df_encoded[col].dtype == 'object': # Check if column exists and is object type (e.g., string)
            # Use LabelEncoder to convert categories into numerical labels (0, 1, 2, ...)
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str)) # Convert to string to handle potential NaN values cleanly
            print(f"  Encoded column: {col}")
        elif col in df_encoded.columns and df_encoded[col].dtype == 'float64':
             # For 'GADE' if it's already float (e.g., from initial integer encoding) fill NaN with mode and convert to int
            df_encoded[col] = df_encoded[col].fillna(df_encoded[col].mode()[0])
            df_encoded[col] = df_encoded[col].astype(int)
            print(f"  Filled NaN and converted {col} to integer.")

    return df_encoded

def split_data(X, y, test_size=0.2, random_state=42):
    
    #Splits the dataset into training and testing sets.
    #The training set is used to train the machine learning models, and the testing set
    #is used to evaluate their performance on unseen data.

    #Parameters:
    #X (pd.DataFrame): The feature matrix (input variables).
    #y (pd.Series or np.array): The target variable (output variable).
    #test_size (float): The proportion of the dataset to include in the test split (e.g., 0.2 for 20%).
    #random_state (int): Controls the shuffling applied to the data before splitting.
    #Ensures reproducibility of the split.

    #Returns:
    #tuple: X_train, X_test, y_train, y_test - the split datasets.
    
    print("\nSplitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"  Training set size: {len(X_train)} samples")
    print(f"  Testing set size: {len(X_test)} samples")
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test):
    
    #Scales numerical features using StandardScaler. This transforms features to have
    #zero mean and unit variance, which is important for many machine learning algorithms
    #(e.g., Linear Regression) that are sensitive to the scale of input features.

    #Parameters:
    #X_train (pd.DataFrame): The training set features.
    #X_test (pd.DataFrame): The testing set features.

    #Returns:
    #tuple: X_train_scaled, X_test_scaled, scaler - the scaled datasets and the fitted StandardScaler object.
    
    print("\nScaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("  Features scaled successfully.")
    return X_train_scaled, X_test_scaled, scaler

def evaluate_regression_model(model, X_train, X_test, y_train, y_test, model_name):
    
    #Evaluates a regression model using R-squared (R²), Mean Squared Error (MSE),
    #and Mean Absolute Error (MAE) for both training and test datasets.

    #Parameters:
    #model: The trained regression model.
    #X_train (np.array): Scaled training features.
    #X_test (np.array): Scaled test features.
    #y_train (np.array): Actual training target values.
    #y_test (np.array): Actual test target values.
    #model_name (str): Name of the model for printing results.

    #Returns:
    #dict: A dictionary containing R², MSE, and MAE for both train and test sets.
    
    print(f"\n==================================================")
    print(f"Evaluation for {model_name}")
    print(f"==================================================")

    # Make predictions on training and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Calculate R-squared (R²)
    # R² measures the proportion of the variance in the dependent variable that is predictable from the independent variables.
    # A higher R² indicates a better fit.
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Calculate Mean Squared Error (MSE)
    # MSE measures the average of the squares of the errors. It's sensitive to outliers.
    # A lower MSE indicates better accuracy.
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    # Calculate Mean Absolute Error (MAE)
    # MAE measures the average of the absolute differences between predictions and actual observations.
    # It is less sensitive to outliers than MSE.
    # A lower MAE indicates better accuracy.
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    print(f"Train R²: {train_r2:.4f}")
    print(f"Test R²:  {test_r2:.4f}")
    print(f"Train MSE: {train_mse:.4f}")
    print(f"Test MSE:  {test_mse:.4f}")
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Test MAE:  {test_mae:.4f}")

    return {'train_r2': train_r2, 'test_r2': test_r2,
            'train_mse': train_mse, 'test_mse': test_mse,
            'train_mae': train_mae, 'test_mae': test_mae}

def plot_r2(all_results):
    #Plots R² scores for each model separately, comparing targets.

    # Prepare data for plotting by iterating through all experiment results.
    plot_data = []
    for target_name, target_data in all_results.items():
        for model_name, model_data in target_data['results'].items():
            plot_data.append({
                'Target': target_name,
                'Model': model_name,
                'Train R²': model_data['metrics']['train_r2'], # R² score on the training data.
                'Test R²': model_data['metrics']['test_r2'] # R² score on the unseen test data.
            })

    # Convert the list of dictionaries into a pandas DataFrame for easier manipulation and plotting.
    df = pd.DataFrame(plot_data)

    # Get unique model names and target variable names from the DataFrame.
    models = df['Model'].unique()
    targets = df['Target'].unique()

    # Define specific colours for each target variable for consistent visualisation.
    bar_colours = ['red', 'green', 'blue'] 
    # Create a mapping from target name to its assigned colour.
    target_colour_map = {target: colour for target, colour in zip(targets, bar_colours)}

    # Calculate the global minimum and maximum R² scores across all models and targets.
    # This is done to set a consistent y-axis range for all plots, making comparisons easier.
    all_test_r2_scores = df['Test R²'].tolist()
    min_r2 = min(0.0, min(all_test_r2_scores)) - 0.05 # Ensures the y-axis starts slightly below the lowest score, or at 0 if all scores are positive.
    max_r2 = max(all_test_r2_scores) + 0.05 # Extends the y-axis slightly above the highest score to accommodate labels.

    # Iterate through each model to create a separate plot for it.
    for current_model in models:
        # Create a new figure and a set of subplots for each model.
        fig, ax = plt.subplots(figsize=(8, 5))
        # Filter the DataFrame to include only data for the current model.
        model_df = df[df['Model'] == current_model]

        # Check if there is data for the current model before attempting to plot.
        if not model_df.empty:
            x = np.arange(len(targets)) # Position for bars on the x-axis.
            width = 1.0  # Set bar width to 1.0 to remove spaces between bars.

            # Extract the 'Test R²' scores for the current model across all target variables.
            test_r2 = [model_df[model_df['Target'] == target]['Test R²'].values[0] 
                       if target in model_df['Target'].values else 0 
                       for target in targets]
            
            # Assign colours to the bars based on the target_color_map.
            colours_for_bars = [target_colour_map[target] for target in targets]

            # Create the bar chart.
            bars = ax.bar(x, test_r2, width, color=colours_for_bars, alpha=0.8)

            # Set labels and title for the plot.
            ax.set_xlabel('Target')
            ax.set_ylabel('R² Score (Test)')
            ax.set_title(f'Test R² Scores for {current_model} by Target')
            ax.set_xticks(x) # Set x-axis ticks to correspond to the number of targets.
            ax.set_xticklabels(targets, rotation=45, ha='right') # Label x-axis with target names, rotated for readability.
            ax.grid(True, alpha=0.3) # Add a grid for better readability.
            ax.axhline(y=0, color='black', linewidth=0.5) # Add a horizontal line at y=0.
            ax.set_ylim(min_r2, max_r2) # Apply the consistent y-axis limit.

            # Add R² values as text labels on top of each bar.
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + 0.005, f'{height:.3f}', # Position the text slightly above the bar.
                        ha='center', va='bottom', fontsize=8) # Center and align the text.

            plt.tight_layout() # Adjust plot to prevent labels from overlapping.
            plt.show() # Display the plot.
        else:
            print(f"No data found for model: {current_model}")

    return df

def plot_mae(all_results):
    #Plots MAE scores for each model separately, comparing targets.

    # Prepare data for plotting by extracting MAE metrics from all_results.
    plot_data = []
    for target_name, target_data in all_results.items():
        for model_name, model_data in target_data['results'].items():
            plot_data.append({
                'Target': target_name,
                'Model': model_name,
                'Train MAE': model_data['metrics']['train_mae'], # MAE score on the training data.
                'Test MAE': model_data['metrics']['test_mae'] # MAE score on the unseen test data.
            })

    # Convert the list of dictionaries into a pandas DataFrame.
    df = pd.DataFrame(plot_data)

    # Get unique model names and target variable names.
    models = df['Model'].unique()
    targets = df['Target'].unique()

    # Define specific colours for each target variable for consistent visualisation.
    bar_colours = ['red', 'green', 'blue']
    # Create a mapping from target name to its assigned colour.
    target_colour_map = {target: colour for target, colour in zip(targets, bar_colours)}

    # Calculate the global minimum and maximum MAE scores across all models and targets.
    # This is done to set a consistent y-axis range for all plots.
    all_test_mae_scores = df['Test MAE'].tolist()
    min_mae = min(0.0, min(all_test_mae_scores)) - 0.1 # Ensures the y-axis starts slightly below the lowest score, or at 0.
    max_mae = max(all_test_mae_scores) + 0.1 # Extends the y-axis slightly above the highest score for labels.

    # Iterate through each model to create a separate plot.
    for current_model in models:
        # Create a new figure and subplots.
        fig, ax = plt.subplots(figsize=(8, 5))
        # Filter data for the current model.
        model_df = df[df['Model'] == current_model]

        # Plot only if data exists for the model.
        if not model_df.empty:
            x = np.arange(len(targets)) # Position for bars on the x-axis.
            width = 1.0  # Set bar width to 1.0 to remove spaces between bars.

            # Extract 'Test MAE' scores for the current model across all target variables.
            test_mae = [model_df[model_df['Target'] == target]['Test MAE'].values[0]
                       if target in model_df['Target'].values else 0
                       for target in targets]
            
            # Assign colours to the bars.
            colours_for_bars = [target_colour_map[target] for target in targets]

            # Create the bar chart.
            bars = ax.bar(x, test_mae, width, color=colours_for_bars, alpha=0.8)

            # Set labels and title.
            ax.set_xlabel('Target')
            ax.set_ylabel('MAE Score (Test)')
            ax.set_title(f'Test MAE Scores for {current_model} by Target')
            ax.set_xticks(x) # Set x-axis ticks.
            ax.set_xticklabels(targets, rotation=45, ha='right') # Label x-axis with target names.
            ax.grid(True, alpha=0.3) # Add a grid.
            ax.axhline(y=0, color='black', linewidth=0.5) # Add a horizontal line at y=0.
            ax.set_ylim(min_mae, max_mae) # Apply the consistent y-axis limit.

            # Add MAE values as text labels on top of each bar.
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2,
                        height + 0.005, f'{height:.3f}', 
                        ha='center', va='bottom', fontsize=8)

            plt.tight_layout() # Adjust layout.
            plt.show() # Display the plot.
        else:
            print(f"No data found for model: {current_model}")

    return df

def plot_r2_values(y_test, y_pred, target_columns):
      
    #Calculates and visualises R-squared scores for each target variable for a single model.  
    #Uses the same style as plot_r2() function.

    #R-squared (R²) is a statistical measure that shows how well a model's predictions match the actual data.
    #A score of 1 means perfect predictions, 0 means no better than just guessing the average, 
    #and negative values mean the model is performing worse than a simple average.

    #Parameters:
        #y_test (pd.DataFrame): The actual true values from your test dataset.
        #y_pred (np.ndarray): The values your model predicted for the test data.
        #target_columns (list): A list of the names of the target variables you're predicting.
                               
      

    # First, we calculate the R² score for each target variable separately
    # We create an empty dictionary to store the scores
    r2_scores = {}
    
    # Print a header so we can see the results clearly
    print("\n--- R-squared Scores ---")
    
    # Loop through each target variable
    for i, target_col in enumerate(target_columns):
        # Calculate R² for this specific target
        # y_test.iloc[:, i] gets all rows for column i (the current target)
        # y_pred[:, i] gets all predictions for the same target
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        
        # Store the score in our dictionary with the target name as the key
        r2_scores[target_col] = r2
        
        # Print the result in a readable format
        print(f"R-squared for {target_col}: {r2:.4f}")

    # Now we create a list of the target names from our dictionary
    # This will be used for labelling the bars on our graph
    targets = list(r2_scores.keys())

    # We choose specific colours for each bar so they're easy to distinguish
    # These colours will be used consistently across all our plots
    bar_colours = ['red', 'green', 'blue']

    # We need to set the y-axis limits for our graph
    # This makes sure all bars fit nicely and we can compare them easily
    all_r2_scores = list(r2_scores.values())
    
    # Calculate the minimum value for the y-axis
    # We use min(0.0, ...) because R² can be negative (bad models)
    # We subtract 0.05 to give a little space below the lowest bar
    min_r2 = min(0.0, min(all_r2_scores)) - 0.05
    
    # Calculate the maximum value for the y-axis
    # We add 0.05 to give space for the value labels above the bars
    max_r2 = max(all_r2_scores) + 0.05

    # Create a new figure for our plot
    # figsize=(8, 5) means 8 inches wide by 5 inches tall
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set up the positions for our bars on the x-axis
    # np.arange creates evenly spaced numbers: [0, 1, 2, ...]
    x = np.arange(len(targets))
    
    # Set the bar width to 1.0 - this removes any gaps between bars
    # Makes the bars look like they're touching each other
    width = 1.0

    # Create the actual bar chart
    # x: where each bar goes on the x-axis
    # all_r2_scores: the height of each bar (the R² values)
    # width: how wide each bar is
    # colour: what colour each bar should be
    # alpha=0.8: opacity
    bars = ax.bar(x, all_r2_scores, width, color=bar_colours, alpha=0.8)

    # Now we add labels and titles to make the graph clear
    ax.set_xlabel('Target')  # Label for the x-axis
    ax.set_ylabel('R² Score')  # Label for the y-axis
    ax.set_title('R² Scores by Target Variable')  # Main title of the graph
    
    # Set where the ticks should go on the x-axis
    ax.set_xticks(x)
    
    # Add the target names below each bar
    # rotation=45: tilt the labels 45 degrees so they don't overlap
    # ha='right': right-align the labels for better readability
    ax.set_xticklabels(targets, rotation=45, ha='right')
    
    # Add a grid to make it easier to read the values
    # alpha=0.3 makes the grid lines faint so they don't distract from the bars
    ax.grid(True, alpha=0.3)

    
    # Set the limits of the y-axis using our calculated min and max values
    ax.set_ylim(min_r2, max_r2)

    # Add the actual R² values as text on top of each bar
    # This makes it easy to see the exact numbers
    for bar in bars:
        # Get the height of this bar (the R² value)
        height = bar.get_height()
        
        # Add text at the top centre of the bar
        # bar.get_x() + bar.get_width() / 2: calculates the centre of the bar
        # height + 0.005: position slightly above the top of the bar
        # f'{height:.3f}': format the number to 3 decimal places
        # ha='center': centre the text horizontally
        # va='bottom': align the bottom of the text to the position
        # fontsize=8: make the text a bit smaller so it fits nicely
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.005, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    
    # Finally, display the plot
    plt.show()

def plot_mae_values(y_test, y_pred, target_columns):
    
    #Calculates and visualises Mean Absolute Error (MAE) scores for each target variable.
    #Uses the same style as plot_mae() function.

    #Mean Absolute Error (MAE) measures how far off your predictions are from the actual values.
    #It calculates the average size of errors, ignoring whether they're positive or negative.
    #A lower MAE means your model's predictions are closer to the true values.
    
    #Parameters:
        #y_test (pd.DataFrame): The actual true values from your test dataset.
        #y_pred (np.ndarray): The values your model predicted for the test data.
        #target_columns (list): A list of the names of the target variables you're predicting.

    # First, we calculate the MAE score for each target variable separately
    # We create an empty dictionary to store the scores
    mae_scores = {}
    
    # Print a header so we can see the results clearly
    print("\n--- Mean Absolute Error Scores ---")
    
    # Loop through each target variable
    for i, target_col in enumerate(target_columns):
        # Calculate MAE for this specific target
        # y_test.iloc[:, i] gets all rows for column i (the current target)
        # y_pred[:, i] gets all predictions for the same target
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        
        # Store the score in our dictionary with the target name as the key
        mae_scores[target_col] = mae
        
        # Print the result in a readable format
        print(f"MAE for {target_col}: {mae:.4f}")

    # Now we create a list of the target names from our dictionary
    # This will be used for labelling the bars on our graph
    targets = list(mae_scores.keys())

    # We choose specific colours for each bar so they're easy to distinguish
    # These colours will be used consistently across all our plots
    # Red, green, and blue bars will match the R² plot colours
    bar_colours = ['red', 'green', 'blue']

    # We need to set the y-axis limits for our graph
    # This makes sure all bars fit nicely and we can compare them easily
    all_mae_scores = list(mae_scores.values())
    
    # Calculate the minimum value for the y-axis
    # We use min(0.0, ...) because MAE should never be negative (errors are always positive)
    # We subtract 0.1 to give a little space below the lowest bar
    min_mae = min(0.0, min(all_mae_scores)) - 0.1
    
    # Calculate the maximum value for the y-axis
    # We add 0.1 to give space for the value labels above the bars
    max_mae = max(all_mae_scores) + 0.1

    # Create a new figure for our plot
    # figsize=(8, 5) means 8 inches wide by 5 inches tall
    fig, ax = plt.subplots(figsize=(8, 5))

    # Set up the positions for our bars on the x-axis
    # np.arange creates evenly spaced numbers: [0, 1, 2, ...]
    x = np.arange(len(targets))
    
    # Set the bar width to 1.0 - this removes any gaps between bars
    # Makes the bars look like they're touching each other
    width = 1.0

    # Create the actual bar chart
    # x: where each bar goes on the x-axis
    # all_mae_scores: the height of each bar (the MAE values)
    # width: how wide each bar is
    # colour: what colour each bar should be
    # alpha=0.8: makes the bars slightly transparent (80% opaque)
    bars = ax.bar(x, all_mae_scores, width, color=bar_colours, alpha=0.8)

    # Now we add labels and titles to make the graph clear
    ax.set_xlabel('Target')  # Label for the x-axis
    ax.set_ylabel('MAE Score')  # Label for the y-axis
    ax.set_title('MAE Scores by Target Variable')  # Main title of the graph
    
    # Set where the ticks (markers) should go on the x-axis
    ax.set_xticks(x)
    
    # Add the target names below each bar
    # rotation=45: tilt the labels 45 degrees so they don't overlap
    # ha='right': right-align the labels for better readability
    ax.set_xticklabels(targets, rotation=45, ha='right')
    
    # Add a grid to make it easier to read the values
    # alpha=0.3 makes the grid lines faint so they don't distract from the bars
    ax.grid(True, alpha=0.3)
    
    # Add a horizontal line at y=0 for reference
    # This helps see the baseline (though MAE should never be below 0)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Set the limits of the y-axis using our calculated min and max values
    ax.set_ylim(min_mae, max_mae)

    # Add the actual MAE values as text on top of each bar
    # This makes it easy to see the exact numbers
    for bar in bars:
        # Get the height of this bar (the MAE value)
        height = bar.get_height()
        
        # Add text at the top centre of the bar
        # bar.get_x() + bar.get_width() / 2: calculates the centre of the bar
        # height + 0.005: position slightly above the top of the bar
        # f'{height:.3f}': format the number to 3 decimal places
        # ha='center': centre the text horizontally
        # va='bottom': align the bottom of the text to the position
        # fontsize=8: make the text a bit smaller so it fits nicely
        ax.text(bar.get_x() + bar.get_width() / 2,
                height + 0.005, f'{height:.3f}',
                ha='center', va='bottom', fontsize=8)

    # Adjust the layout to make sure everything fits without overlapping
    plt.tight_layout()
    
    # Finally, display the plot
    plt.show()

