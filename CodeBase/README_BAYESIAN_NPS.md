# Bayesian NPS Analysis with Cultural Bias Adjustment

## Overview

This implementation provides a comprehensive Bayesian hierarchical modeling framework for adjusting Net Promoter Score (NPS) calculations to account for cultural biases using Hofstede cultural dimensions.

## Features

- **Separate Bayesian Models**: One model per sentiment signal (Raw Stars, BERT, Gemma-27B, Qwen-32B, Qwen-14B)
- **Cultural Bias Adjustment**: Uses Hofstede indices (PDI, IDV, MAS, UAI, LTO, IVR) to adjust sentiment scores
- **Country-level Random Effects**: Captures country-specific cultural biases
- **Comprehensive Evaluation**: Multiple metrics including country-wise dispersion, instability reduction, correlation analysis, and agreement scores
- **CRS-B Calculation**: Composite Reliability Score (Bayesian) for method comparison
- **Validation Methods**: Prior sensitivity analysis, Posterior Predictive Checks (PPC), and Leave-One-Country-Out (LOCO) validation
- **Automated Reporting**: Generates detailed text reports and visualizations

## Requirements

### Python Packages

```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn jax numpyro
```

### GPU Support (Optional)

For GPU acceleration (A6000 compatible):
- CUDA-compatible JAX installation
- Set `CUDA_VISIBLE_DEVICES` environment variable if needed

## Data Requirements

### Main Dataset (`nps_dataset.csv`)

The dataset should contain 30,000 reviews with the following columns:

**Required Columns:**
- `review_id`: Unique review identifier
- `country`: Country name (will be matched with Hofstede data)
- `stars`: Star rating (1-5)
- `sentiment_score_bert_base`: BERT sentiment score (1-5)
- `sentiment_score_gemma27b`: Gemma-27B sentiment score (1-5)
- `sentiment_score_qwen25_32b`: Qwen-32B sentiment score (1-5)
- `sentiment_score_qwen25_14b`: Qwen-14B sentiment score (1-5)

**Optional Columns (if available):**
- `nps_category_stars`: NPS category (Promoter/Passive/Detractor)
- `nps_category_bert_base`: NPS category from BERT
- `nps_category_gemma27b`: NPS category from Gemma
- `nps_category_qwen25_32b`: NPS category from Qwen-32B
- `nps_category_qwen25_14b`: NPS category from Qwen-14B

### Hofstede Data (`hofstede_country_scores.csv`)

Should contain columns:
- `country`: Country name
- `pdi`: Power Distance Index
- `idv`: Individualism
- `mas`: Masculinity
- `uai`: Uncertainty Avoidance
- `lto`: Long-term Orientation
- `ivr`: Indulgence

## Usage

### 1. Update Configuration

Open `bayesian_nps_analysis.ipynb` and update the data file path in the configuration cell:

```python
CONFIG = {
    'data_file': 'path/to/your/nps_dataset.csv',  # Update this
    'hofstede_file': 'hofstede_country_scores.csv',
    # ... other settings
}
```

### 2. Run the Notebook

Execute all cells in `bayesian_nps_analysis.ipynb`. The notebook will:

1. Load and preprocess data
2. Calculate initial NPS scores
3. Fit Bayesian models for each sentiment signal
4. Calculate adjusted NPS scores
5. Compute evaluation metrics
6. Calculate CRS-B scores
7. Perform validation analyses
8. Generate visualizations
9. Create comprehensive report

### 3. Output Files

All outputs are saved in a timestamped directory: `bayesian_nps_results_YYYYMMDD_HHMMSS/`

**Generated Files:**
- `bayesian_nps_report_YYYYMMDD_HHMMSS.txt`: Comprehensive text report
- `metrics_table_YYYYMMDD_HHMMSS.csv`: Metrics comparison table
- `sentiment_distributions_YYYYMMDD_HHMMSS.png`: Distribution plots
- `nps_comparison_YYYYMMDD_HHMMSS.png`: Before/after NPS comparison
- `crs_comparison_YYYYMMDD_HHMMSS.png`: CRS-B scores comparison

## Configuration Options

### Model Parameters

```python
CONFIG = {
    'num_samples': 2000,      # MCMC samples
    'num_warmup': 1000,       # Warmup samples
    'num_chains': 4,          # Number of chains
    'prior_type': 'regularized',  # 'weakly_informative' or 'regularized'
    'min_country_reviews': 10,   # Minimum reviews per country
}
```

### Prior Types

- **`weakly_informative`**: Broader priors, less regularization
- **`regularized`**: Tighter priors, more regularization (default)

## Model Structure

The Bayesian hierarchical model is defined as:

```
Observed_Sentiment = True_Latent_Sentiment + Model_Bias + Cultural_Bias + error
```

Where:
- **True_Latent_Sentiment**: Baseline sentiment (global intercept)
- **Model_Bias**: Systematic bias in the measurement method
- **Cultural_Bias**: Hofstede dimension effects + country random effects
- **error**: Measurement error

## Evaluation Metrics

1. **Country-wise NPS Dispersion**: Standard deviation of NPS across countries
2. **Instability Reduction**: Reduction in NPS variability after adjustment
3. **Hofstede Correlation**: Correlation between adjusted NPS and cultural dimensions
4. **Agreement Score**: Inter-model agreement (Cohen's kappa)
5. **Within-Country Consistency**: Consistency of scores within countries
6. **Distribution Alignment**: Alignment with star ratings

## CRS-B (Composite Reliability Score - Bayesian)

CRS-B combines multiple reliability metrics:

```
CRS-B = w1*(1 - σ_country) + w2*(1 - Δ_pre/post) + w3*(1 - |ρ|) + w4*Agreement
```

Where:
- `σ_country`: Country-wise NPS dispersion
- `Δ_pre/post`: Instability reduction
- `ρ`: Correlation with Hofstede dimensions
- `Agreement`: Inter-model agreement

Higher CRS-B scores indicate better reliability.

## Validation Methods

1. **Prior Sensitivity Analysis**: Tests model robustness to prior specifications
2. **Posterior Predictive Checks (PPC)**: Validates model fit by comparing observed vs predicted distributions
3. **Leave-One-Country-Out (LOCO)**: Assesses cross-country generalization

## Troubleshooting

### Data File Not Found

If you get a "Data file not found" error:
1. Check that the file path in `CONFIG['data_file']` is correct
2. The code will try multiple common paths automatically
3. Use absolute path if needed

### GPU Issues

If GPU is not available:
- The code will automatically fall back to CPU
- Set `CONFIG['use_gpu'] = False` to force CPU mode

### Memory Issues

For large datasets:
- Reduce `num_samples` and `num_warmup` in CONFIG
- Process signals one at a time
- Use fewer chains (`num_chains = 2`)

### Missing Sentiment Scores

The code handles missing values by:
- Filling with median values
- Skipping signals that are completely missing
- Continuing with available signals

## Output Interpretation

### Report Sections

1. **Experiment Summary**: Overview, best method, key findings
2. **Experiment Details**: Hypotheses and results
3. **Model Details**: Architecture and configuration
4. **Evaluation Metrics**: Comprehensive comparison table
5. **CRS-B Details**: Component breakdown
6. **Validation Results**: Prior sensitivity, PPC, LOCO
7. **Conclusions**: Final observations and recommendations

### Metrics Table

The metrics table compares all methods on:
- Country dispersion (before/after)
- Instability reduction
- Hofstede correlation
- Agreement scores
- Within-country consistency
- CRS-B scores

## Best Practices

1. **Data Quality**: Ensure country names match between dataset and Hofstede file
2. **Sample Size**: Ensure sufficient reviews per country (minimum 10)
3. **Prior Selection**: Use `regularized` priors for better generalization
4. **Validation**: Always check PPC and LOCO results
5. **Interpretation**: Consider CRS-B scores in context of all metrics

## Citation

If you use this implementation in research, please cite appropriately and acknowledge the use of:
- NumPyro for Bayesian inference
- Hofstede cultural dimensions
- JAX for computational backend

## License

This implementation is provided for research purposes.

## Contact

For questions or issues, please refer to the project documentation or create an issue in the repository.

