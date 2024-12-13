
#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def analyze_csv(file_path):
    df = pd.read_csv(file_path)
    analysis = {
        "num_rows": len(df),
        "num_columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict(),
    }
    return df, analysis

def create_visualizations(df, dataset_name):
    if not os.path.exists(dataset_name):
        os.makedirs(dataset_name)

    # Correlation heatmap
    if df.select_dtypes(include=['float64', 'int64']).shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
        plt.title(f'{dataset_name} - Correlation Heatmap')
        plt.savefig(f'{dataset_name}/correlation_heatmap.png')
        plt.close()

    # Pair plot
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_columns) > 1:
        sns.pairplot(df[numerical_columns])
        plt.savefig(f'{dataset_name}/pair_plot.png')
        plt.close()

    # Count plot for categorical data
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        plt.figure(figsize=(10, 6))
        sns.countplot(x=df[categorical_columns[0]])
        plt.title(f'{dataset_name} - Count Plot ({categorical_columns[0]})')
        plt.xticks(rotation=45)
        plt.savefig(f'{dataset_name}/count_plot.png')
        plt.close()

def analyze_and_create_visualizations(csv_file):
    # Extract dataset name (without extension)
    dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
    
    # Perform analysis and create visualizations
    print(f"Analyzing {csv_file}...")
    df, analysis = analyze_csv(csv_file)
    print(f"Creating visualizations for {dataset_name}...")
    create_visualizations(df, dataset_name)
    print(f"Analysis complete for {dataset_name}. Visualizations saved.")

# Directly passing the CSV files for analysis
goodreads_file = "goodreads.csv"
happiness_file = "happiness.csv"
media_file = "media.csv"

analyze_and_create_visualizations(goodreads_file)
analyze_and_create_visualizations(happiness_file)
analyze_and_create_visualizations(media_file)
