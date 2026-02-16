import os
import matplotlib.pyplot as plt
import pandas as pd

def analyze_dataset(dataset_path):
    """
    Analyzes the image dataset distribution and displays a bar chart of class counts.
    """
    data = []
    
    # Check if dataset path exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset path '{dataset_path}' not found.")
        return

    # Iterate through splits (train, test, validation)
    splits = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for split in splits:
        split_path = os.path.join(dataset_path, split)
        classes = [c for c in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, c))]
        
        for cls in classes:
            class_path = os.path.join(split_path, cls)
            # Count images (adjust extensions as needed)
            count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            data.append({'Split': split, 'Class': cls, 'Count': count})

    if not data:
        print("No image data found in the specified dataset directory.")
        return

    df = pd.DataFrame(data)
    
    # Print summary table
    print("\nDataset Distribution Summary:")
    print(df.pivot(index='Class', columns='Split', values='Count').fillna(0).astype(int))
    
    # Visualization
    plot_distribution(df)

def plot_distribution(df):
    """
    Plots a grouped bar chart of class counts across splits.
    """
    # Pivot for plotting
    plot_df = df.pivot(index='Class', columns='Split', values='Count').fillna(0)
    
    # Set aesthetics
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(15, 8))
    
    plot_df.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_title('Vegetable Dataset Distribution', fontsize=16, fontweight='bold')
    ax.set_xlabel('Vegetable Classes', fontsize=12)
    ax.set_ylabel('Number of Images', fontsize=12)
    ax.legend(title='Split', fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    output_filename = 'dataset_distribution.png'
    plt.savefig(output_filename)
    print(f"\nBar chart saved as '{output_filename}'")
    
    # Show the plot
    plt.show()

if __name__ == "__main__":
    # Path to the dataset directory
    DATASET_DIR = "dataset"
    
    # Run analysis
    analyze_dataset(DATASET_DIR)
