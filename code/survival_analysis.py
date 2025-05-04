import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

def perform_cox_analysis(df, duration_col="DSS.time", event_col="DSS", score_cols=("str_fraction", "mus_fraction", "tum_fraction"), binary_cols=("MSI",)):
    """Perform Cox proportional hazards regression analysis"""
    # Prepare data
    analysis_df = df[[duration_col, event_col, *score_cols, *binary_cols]].copy()
    
    # Convert time to months
    analysis_df[duration_col] = analysis_df[duration_col] / 30.44
    
    # Binarize continuous variables
    for col in score_cols:
        median = analysis_df[col].median()
        analysis_df[f'{col}_binary'] = (analysis_df[col] > median).astype(int)
    
    # Prepare final analysis data
    final_analysis_df = analysis_df[[duration_col, event_col, *[f"{col}_binary" for col in score_cols], *binary_cols]]
    final_analysis_df = final_analysis_df.dropna()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(final_analysis_df, duration_col=duration_col, event_col=event_col)
    
    return cph, final_analysis_df

def plot_forest_plot(cph, output_path):
    """Create forest plot for multivariate Cox analysis"""
    plt.figure(figsize=(10, 6))
    
    # Get summary data
    summary = cph.summary
    hazard_ratios = summary['exp(coef)']
    conf_intervals = summary[['exp(coef) lower 95%', 'exp(coef) upper 95%']]
    
    # Plot
    y_pos = range(len(hazard_ratios))
    plt.errorbar(
        hazard_ratios, 
        y_pos,
        xerr=[
            hazard_ratios - conf_intervals['exp(coef) lower 95%'],
            conf_intervals['exp(coef) upper 95%'] - hazard_ratios
        ],
        fmt='o',
        capsize=5,
        color='darkblue',
        markersize=8
    )
    
    # Add reference line
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    
    # Set y-axis labels
    plt.yticks(y_pos, hazard_ratios.index)
    
    # Add title and labels
    plt.title('Hazard Ratios with 95% Confidence Intervals', fontsize=12)
    plt.xlabel('Hazard Ratio (log scale)', fontsize=10)
    plt.ylabel('Variables')
    
    # Use log scale
    plt.xscale('log')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add p-values
    for i, (var, row) in enumerate(summary.iterrows()):
        plt.text(
            plt.xlim()[1] * 1.1,  # Position after the error bars
            i,
            f"p = {row['p']:.3f}",
            va='center'
        )
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return summary

def plot_significant_km_curve(df, analysis_df, cox_summary, output_path):
    """Plot Kaplan-Meier survival curve for most significant variable"""
    # Find most significant variable
    most_sig_var = cox_summary['p'].idxmin()
    
    kmf = KaplanMeierFitter()
    plt.figure(figsize=(10, 6))
    
    # High group
    kmf.fit(
        durations=analysis_df[analysis_df[most_sig_var] == 1]['DSS.time'],
        event_observed=analysis_df[analysis_df[most_sig_var] == 1]['DSS'],
        label='High'
    )
    kmf.plot(ci_show=True)
    
    # Low group
    kmf.fit(
        durations=analysis_df[analysis_df[most_sig_var] == 0]['DSS.time'],
        event_observed=analysis_df[analysis_df[most_sig_var] == 0]['DSS'],
        label='Low'
    )
    kmf.plot(ci_show=True)
    
    # Perform logrank test
    high_group = analysis_df[analysis_df[most_sig_var] == 1]
    low_group = analysis_df[analysis_df[most_sig_var] == 0]
    results = logrank_test(
        high_group['DSS.time'],
        low_group['DSS.time'],
        high_group['DSS'],
        low_group['DSS']
    )
    
    # Add plot elements
    var_name = most_sig_var.replace('_binary', '')
    plt.title(f'Disease Specific Survival by {var_name}', fontsize=12)
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.grid(True, alpha=0.3)
    plt.text(0.7, 0.2, f'Log-rank p = {results.p_value:.3e}',
             transform=plt.gca().transAxes)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return results
