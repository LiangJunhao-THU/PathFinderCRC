import os
import pandas as pd
from survival_analysis import perform_cox_analysis, plot_significant_km_curve, plot_forest_plot

def main():
    # Set paths
    output_dir = '/mnt/bulk-saturn/junhao/pathfinder_cre/reproduce'
    final_data_path = os.path.join(output_dir, 'new_TCGA_CRC_clinic.csv')
    
    # Load processed data
    print("Loading processed data...")
    final_df = pd.read_csv(final_data_path)
    
    # Perform survival analysis
    print("Performing survival analysis...")
    cox_model, analysis_df = perform_cox_analysis(final_df)
    
    # Generate plots
    print("Generating plots...")
    cox_summary = plot_forest_plot(cox_model, os.path.join(output_dir, 'multivariate_forest_plot.png'))
    plot_significant_km_curve(final_df, analysis_df, cox_summary, 
                            os.path.join(output_dir, 'significant_var_survival.png'))
    
    # Print detailed results
    print("\nCox Regression Results:")
    print(cox_summary)
    
    print("\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 