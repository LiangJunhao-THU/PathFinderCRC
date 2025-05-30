import os
import pandas as pd
import numpy as np
from tissue_analysis import str_fraction, mus_fraction, tum_fraction

def calculate_tissue_fractions(npy_dir, clinical_df, filename_to_patient_id_df, output_path, scores_to_score_func={
    'str_fraction': str_fraction,
    'mus_fraction': mus_fraction,
    'tum_fraction': tum_fraction
}):
    """Calculate tissue fractions and merge with clinical data"""
    print("Calculating tissue fractions...")
    results = []
    total_files = len([f for f in os.listdir(npy_dir) if f.endswith('.npy')])
    processed_files = 0
    
    for filename in os.listdir(npy_dir):
        if filename.endswith('.npy'):
            patient_id = filename_to_patient_id_df[filename_to_patient_id_df['FILENAME'] == filename]['PATIENT'].values[0]
            npy_path = os.path.join(npy_dir, filename)
            
            # Calculate fractions
            results.append({
                'PATIENT': patient_id,
                **{k: v(npy_path) for k, v in scores_to_score_func.items()}
            })
            
            # Update progress
            processed_files += 1
            if processed_files % 10 == 0:
                print(f"Processed {processed_files}/{total_files} files...")
    
    print("Converting to DataFrame and merging with clinical data...")
    # Convert to DataFrame and merge
    results_df = pd.DataFrame(results)
    final_df = pd.merge(
        clinical_df,
        results_df,
        on='PATIENT',
        how='right'
    )
    
    # Save results
    print(f"Saving results to {output_path}")
    final_df.to_csv(output_path, index=False)
    print("Data generation completed successfully!")
    
    return final_df

def main():
    # Set paths
    base_dir = '/mnt/bulk-saturn/junhao/pathfinder_cre/reproduce'
    info_path = os.path.join(base_dir, 'TCGA_CRC_info.csv')
    filename_to_patient_id_path = os.path.join(base_dir, 'TCGA_CRC_files.csv')
    npy_dir = os.path.join(base_dir, 'TCGA_CRC')
    output_path = os.path.join(base_dir, 'new_TCGA_CRC_clinic.csv')

    # Load clinical data
    print("Loading clinical data...")
    clinical_df = pd.read_csv(info_path)
    filename_to_patient_id_df = pd.read_csv(filename_to_patient_id_path)
    
    # Calculate fractions and generate combined dataset
    final_df = calculate_tissue_fractions(npy_dir, clinical_df, filename_to_patient_id_df, output_path)
    
    # Print summary statistics
    print("\nSummary of tissue fractions:")
    for col in ['str_fraction', 'mus_fraction', 'tum_fraction']:
        print(f"\n{col}:")
        print(final_df[col].describe())

if __name__ == "__main__":
    main() 
