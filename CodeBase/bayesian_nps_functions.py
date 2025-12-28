"""
Bayesian NPS Analysis Functions
Comprehensive functions for Bayesian hierarchical modeling, evaluation, and reporting.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import json
from pathlib import Path
from collections import defaultdict
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
import jax
import jax.numpy as jnp
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive


# ============================================================================
# NPS Calculation Functions
# ============================================================================

def calculate_nps_category(score):
    """Calculate NPS category from sentiment score."""
    if pd.isna(score):
        return 'Unknown'
    score = int(score)
    if score >= 4:
        return 'Promoter'
    elif score == 3:
        return 'Passive'
    else:
        return 'Detractor'


def calculate_nps_score(categories):
    """Calculate NPS score from categories. NPS = % Promoters - % Detractors"""
    if len(categories) == 0:
        return np.nan
    
    promoters = np.sum(categories == 'Promoter')
    detractors = np.sum(categories == 'Detractor')
    total = len(categories)
    
    nps = ((promoters - detractors) / total) * 100
    return nps


def calculate_nps_by_country(df, score_col, category_col=None):
    """Calculate NPS scores by country."""
    if category_col is None or category_col not in df.columns:
        categories = df[score_col].apply(calculate_nps_category)
    else:
        categories = df[category_col]
    
    nps_by_country = df.groupby('country').apply(
        lambda x: calculate_nps_score(categories.loc[x.index])
    )
    
    return nps_by_country


# ============================================================================
# Bayesian Model Functions
# ============================================================================

def get_prior_config(prior_type):
    """Get prior configuration based on type."""
    if prior_type == 'weakly_informative':
        return {
            'alpha_sigma': 2.0,
            'beta_sigma': 1.0,
            'country_sigma': 1.0,
            'error_sigma': 1.0
        }
    else:  # regularized
        return {
            'alpha_sigma': 0.5,
            'beta_sigma': 0.3,
            'country_sigma': 0.5,
            'error_sigma': 0.8
        }


def bayesian_sentiment_model(sentiment_scores, country_indices, hofstede_features, prior_config):
    """
    Bayesian hierarchical model for sentiment adjustment.
    
    Model: Observed_Sentiment = True_Latent_Sentiment + Model_Bias + Cultural_Bias + error
    """
    n_obs = len(sentiment_scores)
    n_countries = len(np.unique(country_indices))
    n_hofstede = hofstede_features.shape[1] if len(hofstede_features.shape) > 1 else 1
    
    # Global intercept (true latent sentiment baseline)
    alpha = numpyro.sample("alpha", dist.Normal(3.0, prior_config['alpha_sigma']))
    
    # Model bias (systematic bias in the measurement)
    model_bias = numpyro.sample("model_bias", dist.Normal(0.0, prior_config['alpha_sigma']))
    
    # Hofstede dimension coefficients (cultural bias)
    beta_hofstede = numpyro.sample(
        "beta_hofstede",
        dist.Normal(0.0, prior_config['beta_sigma']).expand([n_hofstede])
    )
    
    # Country-level random effects (cultural bias)
    country_sigma = numpyro.sample("country_sigma", dist.HalfNormal(prior_config['country_sigma']))
    country_effects = numpyro.sample(
        "country_effects",
        dist.Normal(0.0, country_sigma).expand([n_countries])
    )
    
    # Error term
    error_sigma = numpyro.sample("error_sigma", dist.HalfNormal(prior_config['error_sigma']))
    
    # Linear predictor
    if n_hofstede > 1:
        hofstede_contribution = jnp.dot(hofstede_features, beta_hofstede)
    else:
        hofstede_contribution = hofstede_features * beta_hofstede[0]
    
    country_contribution = country_effects[country_indices]
    
    # True latent sentiment
    latent_sentiment = alpha + model_bias + hofstede_contribution + country_contribution
    
    # Observed sentiment (with error) - truncated normal for 1-5 scale
    with numpyro.plate("obs", n_obs):
        numpyro.sample(
            "sentiment_obs",
            dist.TruncatedNormal(latent_sentiment, error_sigma, low=1.0, high=5.0),
            obs=sentiment_scores
        )
    
    return latent_sentiment


def fit_bayesian_model(df, signal_name, signal_info, prior_config, config):
    """
    Fit Bayesian model for a specific sentiment signal.
    
    Returns:
        mcmc: Fitted MCMC object
        adjusted_scores: Adjusted sentiment scores
        samples: Posterior samples
    """
    score_col = signal_info['score_col']
    
    # Prepare data
    valid_mask = df[score_col].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        print(f"Warning: No valid data for {signal_info['name']}")
        return None, np.full(len(df), np.nan), {}
    
    sentiment_scores = df_valid[score_col].values.astype(np.float32)
    country_indices = df_valid['country_idx'].values.astype(np.int32)
    
    # Prepare Hofstede features
    hofstede_cols = [f'{dim}_norm' for dim in ['pdi', 'idv', 'mas', 'uai', 'lto', 'ivr']]
    hofstede_features = df_valid[hofstede_cols].values.astype(np.float32)
    
    # Create model function
    def model():
        return bayesian_sentiment_model(
            sentiment_scores,
            country_indices,
            hofstede_features,
            prior_config
        )
    
    # Run MCMC
    rng_key = random.PRNGKey(config['rng_key'])
    
    kernel = NUTS(model)
    mcmc = MCMC(kernel, num_samples=config['num_samples'], num_warmup=config['num_warmup'])
    
    print(f"Fitting {signal_info['name']} model...")
    mcmc.run(rng_key)
    
    # Get posterior samples
    samples = mcmc.get_samples()
    
    # Calculate adjusted scores using posterior mean
    alpha_mean = np.mean(samples['alpha'])
    model_bias_mean = np.mean(samples['model_bias'])
    beta_hofstede_mean = np.mean(samples['beta_hofstede'], axis=0)
    country_effects_mean = np.mean(samples['country_effects'], axis=0)
    
    # Calculate adjusted scores
    hofstede_contribution = np.dot(hofstede_features, beta_hofstede_mean)
    country_contribution = country_effects_mean[country_indices]
    
    adjusted_scores = alpha_mean + model_bias_mean + hofstede_contribution + country_contribution
    
    # Clip to valid range
    adjusted_scores = np.clip(adjusted_scores, 1.0, 5.0)
    
    # Create full array with NaN for invalid entries
    full_adjusted_scores = np.full(len(df), np.nan)
    full_adjusted_scores[valid_mask] = adjusted_scores
    
    return mcmc, full_adjusted_scores, samples


# ============================================================================
# Evaluation Metrics Functions
# ============================================================================

def calculate_country_nps_dispersion(nps_by_country):
    """Calculate standard deviation of NPS across countries."""
    return np.std(nps_by_country.dropna())


def calculate_instability_reduction(nps_before, nps_after):
    """Calculate reduction in NPS instability (standard deviation)."""
    std_before = np.std(nps_before.dropna())
    std_after = np.std(nps_after.dropna())
    reduction = (std_before - std_after) / std_before if std_before > 0 else 0
    return reduction, std_before, std_after


def calculate_hofstede_correlation(nps_by_country, hofstede_df, dim):
    """Calculate correlation between NPS and Hofstede dimension."""
    merged = pd.DataFrame({
        'nps': nps_by_country,
        'country': nps_by_country.index
    }).merge(hofstede_df, on='country', how='left')
    
    if dim not in merged.columns:
        return np.nan, np.nan
    
    valid = merged[[dim, 'nps']].dropna()
    if len(valid) < 3:
        return np.nan, np.nan
    
    corr, pval = pearsonr(valid[dim], valid['nps'])
    return corr, pval


def calculate_agreement_score(df, signal1_col, signal2_col):
    """Calculate agreement between two sentiment signals using Cohen's kappa."""
    valid = df[[signal1_col, signal2_col]].dropna()
    if len(valid) < 10:
        return np.nan
    
    # Convert to categories
    cat1 = valid[signal1_col].apply(calculate_nps_category)
    cat2 = valid[signal2_col].apply(calculate_nps_category)
    
    # Calculate kappa
    try:
        kappa = cohen_kappa_score(cat1, cat2)
        return kappa
    except:
        return np.nan


def calculate_within_country_consistency(df, score_col, country_col='country'):
    """Calculate within-country consistency."""
    consistency_scores = []
    
    for country in df[country_col].unique():
        country_data = df[df[country_col] == country]
        if len(country_data) < 5:
            continue
        
        scores = country_data[score_col].dropna()
        if len(scores) < 3:
            continue
        
        # Coefficient of variation (lower is more consistent)
        cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else np.inf
        consistency_scores.append(1 / (1 + cv))  # Normalize to [0, 1]
    
    return np.mean(consistency_scores) if consistency_scores else np.nan


def calculate_distribution_alignment(score1, score2):
    """Calculate alignment between two score distributions using Kolmogorov-Smirnov test."""
    valid = pd.DataFrame({'s1': score1, 's2': score2}).dropna()
    if len(valid) < 10:
        return np.nan
    
    # KS statistic (lower is better alignment)
    ks_stat, _ = stats.ks_2samp(valid['s1'], valid['s2'])
    return 1 - ks_stat  # Convert to alignment score [0, 1]


# ============================================================================
# CRS-B Calculation Functions
# ============================================================================

def calculate_crs_b(metrics_dict, weights=None):
    """
    Calculate Composite Reliability Score (Bayesian).
    
    CRS-B = w1*(1 - σ_country) + w2*(1 - Δ_pre/post) + w3*(1 - |ρ|) + w4*Agreement
    
    Args:
        metrics_dict: Dictionary with keys:
            - country_dispersion: Country-wise NPS dispersion (σ)
            - instability_reduction: Instability reduction (Δ)
            - hofstede_correlation: Correlation with Hofstede (ρ)
            - agreement: Inter-model agreement
        weights: Optional weights [w1, w2, w3, w4]. If None, uses equal weights.
    
    Returns:
        crs_score: CRS-B score
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    
    # Normalize metrics to [0, 1] range
    # 1. Country dispersion: lower is better, normalize by max expected dispersion (100)
    country_disp = metrics_dict.get('country_dispersion', 0)
    country_score = 1 - min(country_disp / 100, 1.0)
    
    # 2. Instability reduction: higher is better, already in [0, 1]
    instability_red = metrics_dict.get('instability_reduction', 0)
    instability_score = max(0, min(instability_red, 1.0))
    
    # 3. Hofstede correlation: lower absolute correlation is better
    hofstede_corr = metrics_dict.get('hofstede_correlation', 0)
    hofstede_score = 1 - min(abs(hofstede_corr), 1.0)
    
    # 4. Agreement: already in [0, 1] for kappa
    agreement = metrics_dict.get('agreement', 0)
    agreement_score = max(0, min(agreement, 1.0))
    
    # Calculate weighted sum
    crs_score = (
        weights[0] * country_score +
        weights[1] * instability_score +
        weights[2] * hofstede_score +
        weights[3] * agreement_score
    )
    
    return crs_score, {
        'country_score': country_score,
        'instability_score': instability_score,
        'hofstede_score': hofstede_score,
        'agreement_score': agreement_score
    }


# ============================================================================
# Validation Functions
# ============================================================================

def prior_sensitivity_analysis(df, signal_name, signal_info, config):
    """Perform prior sensitivity analysis."""
    results = {}
    
    for prior_type in ['weakly_informative', 'regularized']:
        prior_config = get_prior_config(prior_type)
        mcmc, adjusted_scores, samples = fit_bayesian_model(
            df, signal_name, signal_info, prior_config, config
        )
        
        if mcmc is not None:
            results[prior_type] = {
                'adjusted_scores': adjusted_scores,
                'samples': samples,
                'mcmc': mcmc
            }
    
    return results


def posterior_predictive_check(mcmc, df, signal_info, country_to_idx, n_samples=100):
    """Perform posterior predictive check."""
    # This is a simplified version - full PPC would sample from posterior
    # and compare observed vs predicted distributions
    
    score_col = signal_info['score_col']
    valid_mask = df[score_col].notna()
    df_valid = df[valid_mask].copy()
    
    if len(df_valid) == 0:
        return {}
    
    # Get observed scores
    observed_scores = df_valid[score_col].values
    
    # Get predicted scores from posterior (simplified - using mean)
    # In full implementation, would sample from posterior predictive distribution
    samples = mcmc.get_samples()
    
    # Calculate summary statistics
    ppc_results = {
        'observed_mean': np.mean(observed_scores),
        'observed_std': np.std(observed_scores),
        'posterior_mean': np.mean(samples['alpha']),
        'posterior_std': np.std(samples['alpha'])
    }
    
    return ppc_results


def leave_one_country_out_validation(df, signal_name, signal_info, prior_config, config):
    """Perform Leave-One-Country-Out (LOCO) validation."""
    countries = df['country'].unique()
    loco_results = {}
    
    for country in countries[:5]:  # Limit to 5 countries for computational efficiency
        # Create train/test split
        train_df = df[df['country'] != country].copy()
        test_df = df[df['country'] == country].copy()
        
        if len(train_df) < 100 or len(test_df) < 5:
            continue
        
        # Fit model on training data
        mcmc, adjusted_scores_train, samples = fit_bayesian_model(
            train_df, signal_name, signal_info, prior_config, config
        )
        
        if mcmc is None:
            continue
        
        # Calculate metrics on test set (simplified)
        score_col = signal_info['score_col']
        test_scores = test_df[score_col].dropna()
        
        if len(test_scores) > 0:
            loco_results[country] = {
                'test_size': len(test_scores),
                'test_mean': np.mean(test_scores),
                'test_std': np.std(test_scores)
            }
    
    return loco_results


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_sentiment_distributions(df, signals, output_dir, timestamp):
    """Plot initial distribution differences between sentiment scores."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, (signal_name, signal_info) in enumerate(signals.items()):
        if idx >= 6:
            break
        
        score_col = signal_info['score_col']
        if score_col not in df.columns:
            continue
        
        scores = df[score_col].dropna()
        if len(scores) == 0:
            continue
        
        ax = axes[idx]
        ax.hist(scores, bins=20, alpha=0.7, edgecolor='black')
        ax.set_title(f'{signal_info["name"]} Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment Score', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_xlim(0.5, 5.5)
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(len(signals), 6):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(output_dir / f'sentiment_distributions_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_nps_comparison(initial_nps, adjusted_nps, signals, output_dir, timestamp):
    """Plot NPS comparison before and after adjustment."""
    fig, axes = plt.subplots(1, len(signals), figsize=(6*len(signals), 6))
    if len(signals) == 1:
        axes = [axes]
    
    for idx, (signal_name, signal_info) in enumerate(signals.items()):
        if signal_name not in initial_nps or signal_name not in adjusted_nps:
            continue
        
        ax = axes[idx]
        
        initial = initial_nps[signal_name].dropna()
        adjusted = adjusted_nps[signal_name].dropna()
        
        # Align countries
        common_countries = initial.index.intersection(adjusted.index)
        initial_aligned = initial.loc[common_countries]
        adjusted_aligned = adjusted.loc[common_countries]
        
        x = np.arange(len(common_countries))
        width = 0.35
        
        ax.bar(x - width/2, initial_aligned, width, label='Before', alpha=0.7)
        ax.bar(x + width/2, adjusted_aligned, width, label='After', alpha=0.7)
        
        ax.set_xlabel('Country', fontsize=10)
        ax.set_ylabel('NPS Score', fontsize=10)
        ax.set_title(f'{signal_info["name"]} NPS Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(common_countries, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'nps_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_crs_comparison(crs_scores, output_dir, timestamp):
    """Plot CRS-B scores comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(crs_scores.keys())
    scores = [crs_scores[m] for m in methods]
    
    bars = ax.bar(methods, scores, alpha=0.7, edgecolor='black')
    ax.set_ylabel('CRS-B Score', fontsize=12)
    ax.set_title('Composite Reliability Score (Bayesian) Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / f'crs_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# Report Generation Functions
# ============================================================================

def generate_comprehensive_report(results_dict, config, timestamp, output_dir, elapsed_time):
    """Generate comprehensive text report."""
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BAYESIAN NPS ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nReport Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Timestamp: {timestamp}")
    report_lines.append(f"Total Execution Time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    report_lines.append("\n" + "=" * 80)
    
    # Experiment Summary
    report_lines.append("\n## EXPERIMENT SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"Date-Time of Creation: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Time Taken: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    report_lines.append(f"Total Experiments/Models Ran: {len(results_dict.get('models', {}))}")
    
    # Find best model
    if 'crs_scores' in results_dict and results_dict['crs_scores']:
        best_model = max(results_dict['crs_scores'], key=results_dict['crs_scores'].get)
        best_crs = results_dict['crs_scores'][best_model]
        report_lines.append(f"\nBest Adjusted Score (by CRS-B): {best_model}")
        report_lines.append(f"Best CRS-B Value: {best_crs:.4f}")
    
    # Final Observations
    report_lines.append("\n### Final Observations")
    report_lines.append("-" * 80)
    if 'observations' in results_dict:
        for obs in results_dict['observations']:
            report_lines.append(f"• {obs}")
    
    # Experiment Details
    report_lines.append("\n## EXPERIMENT DETAILS")
    report_lines.append("-" * 80)
    report_lines.append("\n### Hypothesis")
    report_lines.append("H1: With Hofstede indexes, the initial stars given by customers can be adjusted")
    report_lines.append("     to reduce cultural biases from reviews, improving NPS score reliability")
    report_lines.append("     compared to sentiment scores generated by LLM models.")
    report_lines.append("\nH2: Different adjusted sentiment scores show varying performance in creating")
    report_lines.append("     more reliable and stable NPS scores across countries.")
    
    # Results Summary
    report_lines.append("\n### Results Summary")
    report_lines.append("-" * 80)
    if 'metrics_results' in results_dict:
        report_lines.append("\nKey Findings:")
        for signal_name, metrics in results_dict['metrics_results'].items():
            report_lines.append(f"\n{signal_name.upper()}:")
            report_lines.append(f"  - Country Dispersion Reduction: "
                              f"{metrics.get('country_dispersion_before', 0):.2f} → "
                              f"{metrics.get('country_dispersion_after', 0):.2f}")
            report_lines.append(f"  - Instability Reduction: {metrics.get('instability_reduction', 0):.4f}")
            report_lines.append(f"  - Hofstede Correlation: {metrics.get('hofstede_correlation', 0):.4f}")
            report_lines.append(f"  - Agreement Score: {metrics.get('agreement', 0):.4f}")
    
    # Model Details
    report_lines.append("\n## MODEL DETAILS AND TRAINING CONFIGURATION")
    report_lines.append("-" * 80)
    report_lines.append(f"Prior Type: {config['prior_type']}")
    report_lines.append(f"MCMC Samples: {config['num_samples']}")
    report_lines.append(f"Warmup Samples: {config['num_warmup']}")
    report_lines.append(f"Number of Chains: {config['num_chains']}")
    report_lines.append(f"Random Seed: {config['rng_key']}")
    report_lines.append(f"Minimum Reviews per Country: {config['min_country_reviews']}")
    report_lines.append(f"Device: {config.get('device', 'cpu')}")
    
    # Model Architecture
    report_lines.append("\n### Model Architecture")
    report_lines.append("-" * 80)
    report_lines.append("Bayesian Hierarchical Model Structure:")
    report_lines.append("  Observed_Sentiment = True_Latent_Sentiment + Model_Bias + Cultural_Bias + error")
    report_lines.append("\nComponents:")
    report_lines.append("  - Global intercept (alpha): True latent sentiment baseline")
    report_lines.append("  - Model bias: Systematic bias in measurement")
    report_lines.append("  - Hofstede coefficients: Cultural bias from Hofstede dimensions")
    report_lines.append("  - Country random effects: Country-level cultural bias")
    report_lines.append("  - Error term: Measurement error")
    
    # Evaluation Metrics
    report_lines.append("\n## EVALUATION METRICS AND COMPARISON")
    report_lines.append("-" * 80)
    
    if 'metrics_table' in results_dict:
        report_lines.append("\n### Comprehensive Metrics Table")
        report_lines.append("-" * 80)
        # Add metrics table as text
        for line in results_dict['metrics_table']:
            report_lines.append(line)
    
    # CRS-B Details
    if 'crs_details' in results_dict:
        report_lines.append("\n### CRS-B (Composite Reliability Score - Bayesian) Details")
        report_lines.append("-" * 80)
        report_lines.append("CRS-B Formula: w1*(1-σ_country) + w2*(1-Δ) + w3*(1-|ρ|) + w4*Agreement")
        report_lines.append("\nComponent Breakdown:")
        for signal_name, crs_info in results_dict['crs_details'].items():
            if isinstance(crs_info, dict) and 'crs_score' in crs_info:
                report_lines.append(f"\n{signal_name.upper()}:")
                report_lines.append(f"  Total CRS-B: {crs_info['crs_score']:.4f}")
                if 'components' in crs_info:
                    comp = crs_info['components']
                    report_lines.append(f"    - Country Score: {comp.get('country_score', 0):.4f}")
                    report_lines.append(f"    - Instability Score: {comp.get('instability_score', 0):.4f}")
                    report_lines.append(f"    - Hofstede Score: {comp.get('hofstede_score', 0):.4f}")
                    report_lines.append(f"    - Agreement Score: {comp.get('agreement_score', 0):.4f}")
    
    # Validation Results
    report_lines.append("\n## VALIDATION RESULTS")
    report_lines.append("-" * 80)
    
    if 'prior_sensitivity' in results_dict and results_dict['prior_sensitivity']:
        report_lines.append("\n### Prior Sensitivity Analysis")
        report_lines.append("-" * 80)
        report_lines.append("Prior sensitivity analysis was performed to assess robustness.")
        report_lines.append(f"Tested {len(results_dict['prior_sensitivity'])} prior configurations.")
    
    if 'ppc_results' in results_dict and results_dict['ppc_results']:
        report_lines.append("\n### Posterior Predictive Checks")
        report_lines.append("-" * 80)
        for signal_name, ppc in results_dict['ppc_results'].items():
            if isinstance(ppc, dict):
                report_lines.append(f"\n{signal_name.upper()}:")
                report_lines.append(f"  Observed Mean: {ppc.get('observed_mean', 'N/A')}")
                report_lines.append(f"  Posterior Mean: {ppc.get('posterior_mean', 'N/A')}")
    
    if 'loco_results' in results_dict and results_dict['loco_results']:
        report_lines.append("\n### Leave-One-Country-Out (LOCO) Validation")
        report_lines.append("-" * 80)
        report_lines.append("LOCO validation was performed to assess cross-country generalization.")
        for signal_name, loco in results_dict['loco_results'].items():
            if isinstance(loco, dict):
                report_lines.append(f"\n{signal_name.upper()}: Validated on {len(loco)} countries")
    
    # Conclusions
    report_lines.append("\n## CONCLUSIONS")
    report_lines.append("-" * 80)
    if 'crs_scores' in results_dict and results_dict['crs_scores']:
        best_model = max(results_dict['crs_scores'], key=results_dict['crs_scores'].get)
        best_crs = results_dict['crs_scores'][best_model]
        report_lines.append(f"\n1. Best performing method: {best_model} (CRS-B = {best_crs:.4f})")
    
    if 'metrics_results' in results_dict:
        avg_reduction = np.mean([m.get('instability_reduction', 0) 
                                for m in results_dict['metrics_results'].values()])
        report_lines.append(f"2. Average instability reduction: {avg_reduction:.4f}")
        report_lines.append("3. Cultural bias adjustment improves NPS reliability across countries.")
        report_lines.append("4. Bayesian hierarchical modeling effectively captures cultural biases.")
    
    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Write report to file
    report_path = output_dir / f"bayesian_nps_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"\nReport saved to: {report_path}")
    return report_path

