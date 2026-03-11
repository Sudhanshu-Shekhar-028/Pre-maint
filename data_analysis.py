import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_analysis():
    # File paths
    input_file = 'dataset/cleaned_dataset.csv'
    plots_dir = 'plots'
    
    # Ensure plots directory exists
    os.makedirs(plots_dir, exist_ok=True)
    
    try:
        print(f"Loading cleaned dataset from {input_file}...")
        df = pd.read_csv(input_file)
        
        # 1. Dataset summary statistics
        print("\n--- Summary Statistics ---")
        summary_stats = df.describe()
        print(summary_stats)
        
        # Save summary statistics to a text file for reference
        with open(os.path.join(plots_dir, 'summary_statistics.txt'), 'w') as f:
            f.write("Dataset Summary Statistics\n")
            f.write("=" * 30 + "\n\n")
            f.write(summary_stats.to_string())
            
        print("\nSaving plots to 'plots' directory...")
            
        # 2. Machine failure distribution plot
        plt.figure(figsize=(8, 6))
        # Handle column naming variations
        target_col = 'Machine_failure' if 'Machine_failure' in df.columns else 'Machine failure'
        
        sns.countplot(data=df, x=target_col, hue=target_col, palette='Set2', legend=False)
        plt.title('Distribution of Machine Failures')
        plt.xlabel('Machine Failure (0 = No, 1 = Yes)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(plots_dir, 'machine_failure_distribution.png'))
        plt.close()
        
        # 3. Correlation matrix
        # Select only numerical columns for correlation calculation
        numeric_df = df.select_dtypes(include=['number'])
        
        plt.figure(figsize=(12, 10))
        # Compute correlation map
        corr = numeric_df.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', 
                    vmin=-1, vmax=1, square=True, linewidths=.5)
        plt.title('Correlation Matrix of Features')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
        plt.close()
        
        # 4. Feature distribution plots
        # Separate plots for the most important continuous variables
        continuous_features = [
            'Air temperature [K]', 
            'Process temperature [K]', 
            'Rotational speed [rpm]', 
            'Torque [Nm]', 
            'Tool wear [min]'
        ]
        
        # Verify the columns exist before plotting
        features_to_plot = [f for f in continuous_features if f in df.columns]
        
        if features_to_plot:
            fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(10, 4 * len(features_to_plot)))
            
            # Handle case where there's only one feature to plot (so axes is not an array)
            if len(features_to_plot) == 1:
                axes = [axes]
                
            for i, col in enumerate(features_to_plot):
                sns.histplot(data=df, x=col, kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_ylabel('Frequency')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_distributions.png'))
            plt.close()
            
            # Optional: Boxplots grouped by machine failure
            fig, axes = plt.subplots(len(features_to_plot), 1, figsize=(10, 4 * len(features_to_plot)))
            if len(features_to_plot) == 1:
                axes = [axes]
                
            for i, col in enumerate(features_to_plot):
                sns.boxplot(data=df, x=target_col, y=col, ax=axes[i], hue=target_col, palette='Set3', legend=False)
                axes[i].set_title(f'{col} by Machine Failure')
                axes[i].set_xlabel('Machine Failure')
                
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'feature_boxplots_by_failure.png'))
            plt.close()
            
        print("\nAnalysis complete! Check the 'plots' folder for the visualizations.")
        
    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
        print("Please run preprocessing.py first to generate the cleaned dataset.")
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    perform_analysis()
