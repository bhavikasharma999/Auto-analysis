import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Set OpenAI API configurations
API_BASE = "https://aiproxy.sanand.workers.dev/openai/v1"
API_KEY = os.environ.get("AIPROXY_TOKEN")
MODEL = "gpt-4o-mini"

# Validate OpenAI API Key
if not API_KEY:
    print("Error: AIPROXY_TOKEN is not set. Ensure the token is loaded in the environment.")
    sys.exit(1)

# Function to call the LLM
def get_llm_response(prompt):
    try:
        print("Sending request to LLM...")
        headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are an AI analyst."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }
        response = requests.post(f"{API_BASE}/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return "No analysis provided by LLM."
    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Request error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"



# Validate command-line arguments
if len(sys.argv) != 2:
    print("Usage: python autolysis.py <dataset.csv>")
    sys.exit(1)

# Load the dataset
csv_file = sys.argv[1]
if not os.path.exists(csv_file):
    print(f"File {csv_file} not found.")
    sys.exit(1)

data = pd.read_csv(csv_file, encoding='ISO-8859-1')

# Generate summary statistics
summary_stats = data.describe(include="all").transpose()
missing_values = data.isnull().sum()
correlation_matrix = data.corr(numeric_only=True)

# Define the output directory based on the dataset name

# Define the output directory based on the dataset name
dataset_name = os.path.splitext(os.path.basename(csv_file))[0]
output_dir = os.path.join(dataset_name)


# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Save correlation matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix")
plt.savefig(os.path.join(output_dir, "correlation_matrix.png"))
plt.close()

# Generate distribution plots for numerical columns
for column in data.select_dtypes(include=[np.number]).columns:
    plt.figure(figsize=(8, 6))
    sns.histplot(data[column].dropna(), kde=True, bins=30, color="blue")
    plt.title(f"Distribution of {column}")
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.savefig(os.path.join(output_dir, f"{column}_distribution.png"))
    plt.close()

# Generic analysis insights
generic_analysis = (
    "- Summary statistics provide an overview of the dataset, including mean, median, and standard deviation.\n"
    "- Missing values are identified and counted to assess data quality.\n"
    "- Correlation matrices reveal relationships between numerical variables.\n"
    "- Distribution plots help identify outliers and data distribution.\n"
    "- Outliers can be visually inspected from the distribution plots and addressed.\n"
    "- Hierarchical patterns or clustering opportunities could be explored for grouping data points.\n"
    "- Additional analyses can involve advanced clustering techniques such as KMeans or DBSCAN."
)

# Prepare data for LLM
sample_data = data.head(5).to_dict(orient="records")
llm_prompt = f"""
You are an data analyst. Here's the dataset summary:
- Columns: {list(data.columns)}
- Data types: {data.dtypes.to_dict()}
- Missing values: {missing_values.to_dict()}
- Sample data: {sample_data}

Perform the following:
1. Analyze the data and provide insights.
2. Suggest any interesting trends, outliers, or relationships.
3. also look if there is possiblities for {generic_analysis} or any other analysis, if yes please do and give insights.

Be concise and professional.
"""

# Call the LLM for analysis
llm_analysis = get_llm_response(llm_prompt)

# Provide fallback insights if LLM call fails
if "failed" in llm_analysis.lower() or "error" in llm_analysis.lower():
    llm_analysis = (
        "The data contains various columns with relationships worth exploring. "
        "Use the correlation matrix and distribution plots to identify key patterns, "
        "trends, or anomalies in the dataset. Ensure to address missing values for better insights."
    )

# Append generic analysis insights
llm_analysis += "\n\n" + generic_analysis

# Generate README.md content
markdown_content = f"""
# Automated Analysis Report

## Dataset Overview
- **Rows**: {data.shape[0]}
- **Columns**: {data.shape[1]}
- **Missing Values**:
{missing_values.to_string()}

## Key Insights
{llm_analysis}

## Visualizations
### Correlation Matrix
![Correlation Matrix](correlation_matrix.png)

### Distributions
"""

# Add distribution plots to the README content
for column in data.select_dtypes(include=[np.number]).columns:
    markdown_content += f"![{column} Distribution]({column}_distribution.png)\n"

# Save README.md inside the output directory
readme_path = os.path.join(output_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(markdown_content)

print(f"Analysis complete. Outputs saved in {output_dir}/README.md and PNG files.")

