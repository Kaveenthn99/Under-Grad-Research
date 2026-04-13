"""
GMM + Z-Score Hybrid Selection for GNINA Docking Results
FINAL VERSION - Strategy A (~1,600 molecules) with Complete Statistical Output
Run from terminal: python gmm_selection_final.py
"""

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.stats import zscore, entropy, norm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10

# ============================================================================
# CONFIGURATION - CHANGE THESE PATHS FOR YOUR DATA
# ============================================================================

# INPUT FILE PATH (CSV file)
INPUT_FILE = "/Users/kaveen/Desktop/Stat/All_TSLPdocking_top_pose_by_CNN_VS_sorted.csv"

# OUTPUT DIRECTORY (where results will be saved)
OUTPUT_DIR = "/Users/kaveen/Desktop/Stat/selection_results"

# OUTPUT FILE PREFIX
OUTPUT_PREFIX = "selected_molecules"

# TARGET SIZE FOR STRATEGY A
TARGET_TIER1_SIZE = 1350  # Will give ~1,598 total with other tiers

# ============================================================================
# END CONFIGURATION
# ============================================================================


class StatisticsLogger:
    """Class to collect and save all statistical information"""
    def __init__(self):
        self.log_lines = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def add_section(self, title):
        """Add a section header"""
        self.log_lines.append("\n" + "="*80)
        self.log_lines.append(title.center(80))
        self.log_lines.append("="*80 + "\n")
    
    def add_line(self, text):
        """Add a line of text"""
        self.log_lines.append(text)
    
    def add_blank(self):
        """Add a blank line"""
        self.log_lines.append("")
    
    def save(self, filepath):
        """Save to file"""
        with open(filepath, 'w') as f:
            f.write("\n".join(self.log_lines))


def load_data(input_file, stats_logger):
    """Load GNINA docking results from CSV file"""
    print("="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)
    
    stats_logger.add_section("STEP 1: DATA LOADING")
    stats_logger.add_line(f"Timestamp: {stats_logger.timestamp}")
    stats_logger.add_line(f"Input file: {input_file}")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}\nCurrent directory: {os.getcwd()}")
    
    # Read CSV
    df = pd.read_csv(input_file)
    print(f"✓ Loaded CSV file: {input_file}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns)}")
    
    stats_logger.add_line(f"Total rows loaded: {len(df):,}")
    stats_logger.add_line(f"Columns: {', '.join(df.columns)}")
    
    # Validate required columns
    required_cols = ['Compound', 'Pose', 'CNNscore', 'CNNaffinity', 'CNN_VS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n❌ ERROR: Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        
        # Try to auto-detect columns
        print("\n   Attempting to auto-detect columns...")
        column_mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if 'compound' in col_lower or 'molecule' in col_lower or 'id' in col_lower:
                column_mapping[col] = 'Compound'
            elif 'pose' in col_lower:
                column_mapping[col] = 'Pose'
            elif 'cnnscore' in col_lower and 'cnn_vs' not in col_lower:
                column_mapping[col] = 'CNNscore'
            elif 'cnnaffinity' in col_lower or 'affinity' in col_lower:
                column_mapping[col] = 'CNNaffinity'
            elif 'cnn_vs' in col_lower or 'cnnvs' in col_lower:
                column_mapping[col] = 'CNN_VS'
        
        if column_mapping:
            print(f"   Found mapping: {column_mapping}")
            df = df.rename(columns=column_mapping)
            print("   ✓ Columns renamed successfully")
            stats_logger.add_line(f"Column mapping applied: {column_mapping}")
        else:
            raise ValueError("Could not auto-detect column names. Please ensure your CSV has the required columns.")
    
    print(f"✓ All required columns present")
    
    # Check for missing values in CNN_VS
    missing_cnn_vs = df['CNN_VS'].isnull().sum()
    if missing_cnn_vs > 0:
        print(f"\n⚠ Warning: {missing_cnn_vs} rows have missing CNN_VS values")
        print(f"  Removing these rows...")
        df = df.dropna(subset=['CNN_VS'])
        print(f"  Remaining rows: {len(df):,}")
        stats_logger.add_line(f"Removed {missing_cnn_vs} rows with missing CNN_VS values")
        stats_logger.add_line(f"Remaining rows: {len(df):,}")
    
    # Show data preview
    print(f"\nData Preview (first 5 rows):")
    print(df.head().to_string())
    
    print(f"\nCNN_VS Statistics:")
    print(df['CNN_VS'].describe())
    
    # Log detailed statistics
    stats_logger.add_blank()
    stats_logger.add_line("CNN_VS Score Statistics (Raw Data):")
    stats_logger.add_line(f"  Count:          {len(df):,}")
    stats_logger.add_line(f"  Mean:           {df['CNN_VS'].mean():.6f}")
    stats_logger.add_line(f"  Std Dev:        {df['CNN_VS'].std():.6f}")
    stats_logger.add_line(f"  Variance:       {df['CNN_VS'].var():.6f}")
    stats_logger.add_line(f"  Minimum:        {df['CNN_VS'].min():.6f}")
    stats_logger.add_line(f"  25th percentile: {df['CNN_VS'].quantile(0.25):.6f}")
    stats_logger.add_line(f"  Median (50th):  {df['CNN_VS'].median():.6f}")
    stats_logger.add_line(f"  75th percentile: {df['CNN_VS'].quantile(0.75):.6f}")
    stats_logger.add_line(f"  90th percentile: {df['CNN_VS'].quantile(0.90):.6f}")
    stats_logger.add_line(f"  95th percentile: {df['CNN_VS'].quantile(0.95):.6f}")
    stats_logger.add_line(f"  99th percentile: {df['CNN_VS'].quantile(0.99):.6f}")
    stats_logger.add_line(f"  Maximum:        {df['CNN_VS'].max():.6f}")
    stats_logger.add_line(f"  Skewness:       {df['CNN_VS'].skew():.6f}")
    stats_logger.add_line(f"  Kurtosis:       {df['CNN_VS'].kurtosis():.6f}")
    
    return df


def select_best_poses(df, stats_logger):
    """Keep only the best pose (highest CNN_VS) for each compound"""
    print("\n" + "="*80)
    print("STEP 2: SELECTING BEST POSE PER COMPOUND")
    print("="*80)
    
    stats_logger.add_section("STEP 2: BEST POSE SELECTION")
    
    print(f"Original data: {len(df):,} rows")
    print(f"Unique compounds: {df['Compound'].nunique():,}")
    
    stats_logger.add_line(f"Original rows: {len(df):,}")
    stats_logger.add_line(f"Unique compounds: {df['Compound'].nunique():,}")
    stats_logger.add_line(f"Average poses per compound: {len(df)/df['Compound'].nunique():.2f}")
    
    # For each compound, keep the row with maximum CNN_VS
    df_best = df.loc[df.groupby('Compound')['CNN_VS'].idxmax()].copy()
    df_best = df_best.reset_index(drop=True)
    
    print(f"\nAfter keeping best pose per compound:")
    print(f"  Rows: {len(df_best):,}")
    print(f"  Unique compounds: {df_best['Compound'].nunique():,}")
    
    stats_logger.add_blank()
    stats_logger.add_line(f"After best pose selection:")
    stats_logger.add_line(f"  Rows: {len(df_best):,}")
    stats_logger.add_line(f"  Unique compounds: {df_best['Compound'].nunique():,}")
    stats_logger.add_line(f"  Rows removed: {len(df) - len(df_best):,}")
    
    # Show which poses were kept
    pose_counts = df_best['Pose'].value_counts()
    print(f"\nPose distribution (best poses):")
    for pose, count in pose_counts.head(10).items():
        print(f"  {pose}: {count:,}")
    
    stats_logger.add_blank()
    stats_logger.add_line("Pose distribution (best poses):")
    for pose, count in pose_counts.items():
        stats_logger.add_line(f"  {pose}: {count:,} ({100*count/len(df_best):.2f}%)")
    
    # Log statistics after best pose selection
    stats_logger.add_blank()
    stats_logger.add_line("CNN_VS Statistics (After Best Pose Selection):")
    stats_logger.add_line(f"  Mean:    {df_best['CNN_VS'].mean():.6f}")
    stats_logger.add_line(f"  Std Dev: {df_best['CNN_VS'].std():.6f}")
    stats_logger.add_line(f"  Minimum: {df_best['CNN_VS'].min():.6f}")
    stats_logger.add_line(f"  Maximum: {df_best['CNN_VS'].max():.6f}")
    
    return df_best


def fit_gmm(df, stats_logger, cnn_col='CNN_VS'):
    """Fit Gaussian Mixture Model to find optimal number of clusters"""
    print("\n" + "="*80)
    print("STEP 3: GMM CLUSTERING")
    print("="*80)
    
    stats_logger.add_section("STEP 3: GAUSSIAN MIXTURE MODEL FITTING")
    
    # Prepare data
    X = df[cnn_col].values.reshape(-1, 1)
    
    stats_logger.add_line("GMM Parameters:")
    stats_logger.add_line("  Covariance type: full")
    stats_logger.add_line("  Random state: 42")
    stats_logger.add_line("  Max iterations: 1000")
    stats_logger.add_line("  Number of initializations: 10")
    stats_logger.add_blank()
    
    # Test different numbers of components
    print("\nTesting different numbers of clusters...")
    stats_logger.add_line("Model Selection (Testing K = 2 to 5):")
    stats_logger.add_line("-" * 60)
    
    n_components_range = range(2, 6)
    bic_scores = []
    aic_scores = []
    models = []
    log_likelihoods = []
    
    for n in n_components_range:
        gmm = GaussianMixture(
            n_components=n,
            covariance_type='full',
            random_state=42,
            max_iter=1000,
            n_init=10
        )
        gmm.fit(X)
        
        bic = gmm.bic(X)
        aic = gmm.aic(X)
        log_likelihood = gmm.score(X) * len(X)  # Total log likelihood
        
        bic_scores.append(bic)
        aic_scores.append(aic)
        log_likelihoods.append(log_likelihood)
        models.append(gmm)
        
        print(f"  {n} components: BIC={bic:,.2f}, AIC={aic:,.2f}")
        
        stats_logger.add_line(f"K = {n}:")
        stats_logger.add_line(f"  BIC: {bic:.4f}")
        stats_logger.add_line(f"  AIC: {aic:.4f}")
        stats_logger.add_line(f"  Log-likelihood: {log_likelihood:.4f}")
        stats_logger.add_line(f"  Number of parameters: {n * 3 - 1}")
        stats_logger.add_line(f"  Converged: {gmm.converged_}")
        stats_logger.add_line(f"  Iterations: {gmm.n_iter_}")
        stats_logger.add_blank()
    
    # Select model with lowest BIC
    optimal_idx = np.argmin(bic_scores)
    optimal_n = list(n_components_range)[optimal_idx]
    gmm_final = models[optimal_idx]
    
    print(f"\n✓ Selected {optimal_n} components (lowest BIC = {bic_scores[optimal_idx]:,.2f})")
    
    stats_logger.add_line("="*60)
    stats_logger.add_line(f"OPTIMAL MODEL: K = {optimal_n}")
    stats_logger.add_line(f"  BIC: {bic_scores[optimal_idx]:.4f} (MINIMUM)")
    stats_logger.add_line(f"  AIC: {aic_scores[optimal_idx]:.4f}")
    stats_logger.add_blank()
    
    # Log detailed GMM parameters
    stats_logger.add_line("Optimal GMM Parameters:")
    stats_logger.add_line("-" * 60)
    for i in range(optimal_n):
        stats_logger.add_line(f"Component {i}:")
        stats_logger.add_line(f"  Weight (π_{i}):          {gmm_final.weights_[i]:.6f}")
        stats_logger.add_line(f"  Mean (μ_{i}):            {gmm_final.means_[i][0]:.6f}")
        stats_logger.add_line(f"  Variance (σ²_{i}):       {gmm_final.covariances_[i][0][0]:.6f}")
        stats_logger.add_line(f"  Std Dev (σ_{i}):         {np.sqrt(gmm_final.covariances_[i][0][0]):.6f}")
        stats_logger.add_blank()
    
    return gmm_final, optimal_n


def assign_clusters(df, gmm, stats_logger, cnn_col='CNN_VS'):
    """Assign molecules to clusters and calculate probabilities"""
    print("\n" + "="*80)
    print("STEP 4: CLUSTER ASSIGNMENT & ANALYSIS")
    print("="*80)
    
    stats_logger.add_section("STEP 4: CLUSTER ASSIGNMENT AND ANALYSIS")
    
    X = df[cnn_col].values.reshape(-1, 1)
    
    # Get cluster assignments
    df['gmm_cluster'] = gmm.predict(X)
    probs = gmm.predict_proba(X)
    
    # Store probabilities for each cluster
    n_components = gmm.n_components
    for i in range(n_components):
        df[f'prob_cluster_{i}'] = probs[:, i]
    
    # Maximum probability (confidence)
    df['max_cluster_prob'] = probs.max(axis=1)
    
    # Uncertainty (entropy)
    df['cluster_uncertainty'] = np.apply_along_axis(entropy, 1, probs)
    
    # Analyze clusters
    cluster_stats = []
    for i in range(n_components):
        mask = df['gmm_cluster'] == i
        cluster_data = df[mask][cnn_col]
        
        cluster_stats.append({
            'Cluster_ID': i,
            'Size': mask.sum(),
            'Percentage': f"{100*mask.sum()/len(df):.1f}%",
            'Mean': cluster_data.mean(),
            'Std': cluster_data.std(),
            'Min': cluster_data.min(),
            'Max': cluster_data.max(),
            'Median': cluster_data.median()
        })
    
    cluster_df = pd.DataFrame(cluster_stats).sort_values('Mean')
    
    # Assign categories based on mean
    categories = ['Weak', 'Moderate', 'Strong', 'Very_Strong'][:len(cluster_df)]
    cluster_df['Category'] = categories
    
    print("\nCluster Statistics:")
    print(cluster_df.to_string(index=False))
    
    # Log detailed cluster statistics
    stats_logger.add_line("Cluster Assignment Results:")
    stats_logger.add_line("="*60)
    
    for idx, row in cluster_df.iterrows():
        cluster_id = row['Cluster_ID']
        mask = df['gmm_cluster'] == cluster_id
        cluster_data = df[mask][cnn_col]
        
        stats_logger.add_line(f"CLUSTER {cluster_id} ({row['Category']}):")
        stats_logger.add_line(f"  Size:            {row['Size']:,} molecules ({100*row['Size']/len(df):.2f}%)")
        stats_logger.add_line(f"  Mean (μ):        {row['Mean']:.6f}")
        stats_logger.add_line(f"  Std Dev (σ):     {row['Std']:.6f}")
        stats_logger.add_line(f"  Variance (σ²):   {row['Std']**2:.6f}")
        stats_logger.add_line(f"  Minimum:         {row['Min']:.6f}")
        stats_logger.add_line(f"  25th percentile: {cluster_data.quantile(0.25):.6f}")
        stats_logger.add_line(f"  Median:          {row['Median']:.6f}")
        stats_logger.add_line(f"  75th percentile: {cluster_data.quantile(0.75):.6f}")
        stats_logger.add_line(f"  Maximum:         {row['Max']:.6f}")
        stats_logger.add_line(f"  Range:           {row['Max'] - row['Min']:.6f}")
        stats_logger.add_line(f"  IQR:             {cluster_data.quantile(0.75) - cluster_data.quantile(0.25):.6f}")
        stats_logger.add_line(f"  Skewness:        {cluster_data.skew():.6f}")
        stats_logger.add_line(f"  Kurtosis:        {cluster_data.kurtosis():.6f}")
        stats_logger.add_blank()
    
    # Log confidence statistics
    stats_logger.add_line("Cluster Assignment Confidence:")
    stats_logger.add_line("-" * 60)
    stats_logger.add_line(f"Mean max probability:     {df['max_cluster_prob'].mean():.6f}")
    stats_logger.add_line(f"Median max probability:   {df['max_cluster_prob'].median():.6f}")
    stats_logger.add_line(f"Min max probability:      {df['max_cluster_prob'].min():.6f}")
    stats_logger.add_blank()
    
    confidence_bins = [
        (0.5, 0.7, "Low confidence"),
        (0.7, 0.85, "Moderate confidence"),
        (0.85, 0.95, "High confidence"),
        (0.95, 1.0, "Very high confidence")
    ]
    
    for low, high, label in confidence_bins:
        count = ((df['max_cluster_prob'] >= low) & (df['max_cluster_prob'] < high)).sum()
        pct = 100 * count / len(df)
        stats_logger.add_line(f"  {label} ({low:.2f}-{high:.2f}): {count:,} ({pct:.2f}%)")
    
    # Map categories to main dataframe
    cluster_to_category = dict(zip(cluster_df['Cluster_ID'], cluster_df['Category']))
    df['cluster_category'] = df['gmm_cluster'].map(cluster_to_category)
    
    return df, cluster_df


def validate_known_actives(df, cluster_df, stats_logger, cnn_col='CNN_VS'):
    """Check where known actives fall in the clustering"""
    print("\n" + "="*80)
    print("STEP 5: KNOWN ACTIVE VALIDATION")
    print("="*80)
    
    stats_logger.add_section("STEP 5: KNOWN ACTIVE MOLECULE VALIDATION")
    
    known_actives = {
        'BP79': 3.48302,
        'Galbelgin': 3.24381,
        'Fragment_3': 3.24966
    }
    
    n_components = len(cluster_df)
    cluster_to_category = dict(zip(cluster_df['Cluster_ID'], cluster_df['Category']))
    
    print("\nKnown active molecules:")
    
    for name, score in known_actives.items():
        # Find closest molecule in dataset
        closest_idx = (df[cnn_col] - score).abs().idxmin()
        closest_row = df.loc[closest_idx]
        
        percentile = (df[cnn_col] < score).sum() / len(df) * 100
        
        print(f"\n{name} (CNN_VS = {score:.3f}):")
        print(f"  Percentile: {percentile:.1f}th")
        print(f"  Closest match in dataset:")
        print(f"    Compound: {closest_row['Compound']}")
        print(f"    CNN_VS: {closest_row[cnn_col]:.3f}")
        print(f"    Cluster: {closest_row['cluster_category']}")
        print(f"  Cluster probabilities:")
        for i in range(n_components):
            category = cluster_to_category[i]
            prob = closest_row[f'prob_cluster_{i}']
            print(f"    P({category}): {prob:.3f}")
        
        # Log to stats file
        stats_logger.add_line(f"{name}:")
        stats_logger.add_line(f"  Reference CNN_VS:     {score:.6f}")
        stats_logger.add_line(f"  Percentile rank:      {percentile:.2f}th")
        stats_logger.add_line(f"  Molecules scoring higher: {len(df[df[cnn_col] > score]):,} ({100 - percentile:.2f}%)")
        stats_logger.add_line(f"  Closest match compound: {closest_row['Compound']}")
        stats_logger.add_line(f"  Closest match CNN_VS:   {closest_row[cnn_col]:.6f}")
        stats_logger.add_line(f"  Distance from reference: {abs(closest_row[cnn_col] - score):.6f}")
        stats_logger.add_line(f"  Assigned cluster:       {closest_row['cluster_category']} (Cluster {closest_row['gmm_cluster']})")
        stats_logger.add_line(f"  Assignment confidence:  {closest_row['max_cluster_prob']:.6f}")
        
        stats_logger.add_line(f"  Cluster probabilities:")
        for i in range(n_components):
            category = cluster_to_category[i]
            prob = closest_row[f'prob_cluster_{i}']
            stats_logger.add_line(f"    P({category}): {prob:.6f}")
        
        # Calculate Z-score within assigned cluster
        cluster_id = closest_row['gmm_cluster']
        cluster_data = df[df['gmm_cluster'] == cluster_id][cnn_col]
        z_within = (closest_row[cnn_col] - cluster_data.mean()) / cluster_data.std()
        
        stats_logger.add_line(f"  Z-score within cluster: {z_within:.6f}")
        stats_logger.add_blank()


def select_molecules(df, cluster_df, stats_logger, cnn_col='CNN_VS', target_tier1=1350):
    """Select molecules using Strategy A - ~1,600 total"""
    print("\n" + "="*80)
    print("STEP 6: MOLECULE SELECTION - STRATEGY A")
    print("="*80)
    
    stats_logger.add_section("STEP 6: MOLECULE SELECTION - STRATEGY A")
    stats_logger.add_line(f"Target Tier 1 size: {target_tier1}")
    stats_logger.add_blank()
    
    # Get cluster IDs sorted by mean
    clusters_sorted = cluster_df.sort_values('Mean')['Cluster_ID'].tolist()
    cluster_to_category = dict(zip(cluster_df['Cluster_ID'], cluster_df['Category']))
    
    weak_cluster = clusters_sorted[0]
    moderate_cluster = clusters_sorted[1] if len(clusters_sorted) > 1 else None
    strong_cluster = clusters_sorted[-1]
    
    stats_logger.add_line("Cluster Assignments:")
    stats_logger.add_line(f"  Weak cluster:     Cluster {weak_cluster} ({cluster_to_category[weak_cluster]})")
    if moderate_cluster is not None:
        stats_logger.add_line(f"  Moderate cluster: Cluster {moderate_cluster} ({cluster_to_category[moderate_cluster]})")
    stats_logger.add_line(f"  Strong cluster:   Cluster {strong_cluster} ({cluster_to_category[strong_cluster]})")
    stats_logger.add_blank()
    
    selected_list = []
    
    # =========================================================================
    # TIER 1: Top N from Strong cluster with high confidence
    # =========================================================================
    print(f"\n--- TIER 1: Top {target_tier1} from Strong Cluster ---")
    stats_logger.add_line("TIER 1: Strong Cluster (High Confidence)")
    stats_logger.add_line("-" * 60)
    
    tier1_pool = df[
        (df['gmm_cluster'] == strong_cluster) &
        (df[f'prob_cluster_{strong_cluster}'] > 0.7)
    ].copy()
    
    stats_logger.add_line(f"Selection criteria:")
    stats_logger.add_line(f"  Cluster: {strong_cluster} ({cluster_to_category[strong_cluster]})")
    stats_logger.add_line(f"  P(Strong) > 0.7")
    stats_logger.add_line(f"  Take top {target_tier1} by CNN_VS")
    stats_logger.add_blank()
    
    stats_logger.add_line(f"Pool of eligible molecules: {len(tier1_pool):,}")
    
    # Select top N by CNN_VS
    n_tier1 = min(target_tier1, len(tier1_pool))
    tier1 = tier1_pool.nlargest(n_tier1, cnn_col).copy()
    
    tier1['selection_tier'] = 'Tier1_Strong_HighConf'
    tier1['selection_reason'] = f'Top {n_tier1} with high probability (P>0.7)'
    selected_list.append(tier1)
    
    print(f"Selected: {len(tier1):,} molecules")
    print(f"CNN_VS range: [{tier1[cnn_col].min():.2f}, {tier1[cnn_col].max():.2f}]")
    print(f"Mean CNN_VS: {tier1[cnn_col].mean():.2f}")
    
    stats_logger.add_line(f"Selected: {len(tier1):,} molecules")
    stats_logger.add_line(f"CNN_VS statistics:")
    stats_logger.add_line(f"  Mean:    {tier1[cnn_col].mean():.6f}")
    stats_logger.add_line(f"  Std Dev: {tier1[cnn_col].std():.6f}")
    stats_logger.add_line(f"  Median:  {tier1[cnn_col].median():.6f}")
    stats_logger.add_line(f"  Minimum: {tier1[cnn_col].min():.6f}")
    stats_logger.add_line(f"  Maximum: {tier1[cnn_col].max():.6f}")
    stats_logger.add_line(f"  Range:   {tier1[cnn_col].max() - tier1[cnn_col].min():.6f}")
    
    if len(tier1_pool) > target_tier1:
        cutoff_score = tier1[cnn_col].min()
        print(f"Cutoff score: {cutoff_score:.3f}")
        print(f"  ({100*n_tier1/len(tier1_pool):.1f}% of high-confidence molecules)")
        
        stats_logger.add_line(f"Cutoff CNN_VS score: {cutoff_score:.6f}")
        stats_logger.add_line(f"Percentage of pool selected: {100*n_tier1/len(tier1_pool):.2f}%")
    
    stats_logger.add_line(f"Mean probability: {tier1['max_cluster_prob'].mean():.6f}")
    stats_logger.add_blank()
    
    # =========================================================================
    # TIER 2B: Known active region
    # =========================================================================
    if moderate_cluster is not None:
        print("\n--- TIER 2B: Known Active Window ---")
        stats_logger.add_line("TIER 2B: Known Active Window (Validation)")
        stats_logger.add_line("-" * 60)
        
        known_actives = {
            'BP79': 3.48302,
            'Galbelgin': 3.24381,
            'Fragment_3': 3.24966
        }
        
        std_dev = df[cnn_col].std()
        window = 0.5 * std_dev
        
        print(f"Window size: ±{window:.3f} (0.5 × SD)")
        
        stats_logger.add_line(f"Selection criteria:")
        stats_logger.add_line(f"  Window size: ±{window:.6f} (0.5 × standard deviation)")
        stats_logger.add_line(f"  Standard deviation used: {std_dev:.6f}")
        stats_logger.add_blank()
        
        # Get all already selected indices
        already_selected = pd.concat(selected_list).index
        
        moderate_df = df[df['gmm_cluster'] == moderate_cluster].copy()
        
        tier2b_list = []
        for active_name, active_score in known_actives.items():
            in_window = moderate_df[
                (moderate_df[cnn_col] >= active_score - window) &
                (moderate_df[cnn_col] <= active_score + window) &
                (~moderate_df.index.isin(already_selected))
            ].copy()
            
            in_window['reference_active'] = active_name
            tier2b_list.append(in_window)
            
            print(f"  Near {active_name} ({active_score:.3f}): "
                  f"{len(in_window)} molecules in [{active_score-window:.2f}, {active_score+window:.2f}]")
            
            stats_logger.add_line(f"Window around {active_name} (CNN_VS = {active_score:.6f}):")
            stats_logger.add_line(f"  Range: [{active_score-window:.6f}, {active_score+window:.6f}]")
            stats_logger.add_line(f"  Molecules in window: {len(in_window):,}")
        
        if tier2b_list:
            tier2b_all = pd.concat(tier2b_list).drop_duplicates()
            n_sample = min(100, len(tier2b_all))
            tier2b = tier2b_all.sample(n=n_sample, random_state=42).copy()
            tier2b['selection_tier'] = 'Tier2B_KnownActive_Window'
            tier2b['selection_reason'] = 'Within ±0.5 SD of known actives'
            selected_list.append(tier2b)
            
            print(f"\nSelected: {len(tier2b):,} molecules (sampled from {len(tier2b_all)})")
            print(f"CNN_VS range: [{tier2b[cnn_col].min():.2f}, {tier2b[cnn_col].max():.2f}]")
            
            stats_logger.add_blank()
            stats_logger.add_line(f"Total unique molecules in all windows: {len(tier2b_all):,}")
            stats_logger.add_line(f"Sampled for Tier 2B: {len(tier2b):,}")
            stats_logger.add_line(f"CNN_VS statistics:")
            stats_logger.add_line(f"  Mean:    {tier2b[cnn_col].mean():.6f}")
            stats_logger.add_line(f"  Std Dev: {tier2b[cnn_col].std():.6f}")
            stats_logger.add_line(f"  Minimum: {tier2b[cnn_col].min():.6f}")
            stats_logger.add_line(f"  Maximum: {tier2b[cnn_col].max():.6f}")
        
        stats_logger.add_blank()
    
    # =========================================================================
    # TIER 3: Borderline molecules
    # =========================================================================
    if moderate_cluster is not None and len(clusters_sorted) >= 2:
        print("\n--- TIER 3: Borderline Cases ---")
        stats_logger.add_line("TIER 3: Borderline Cases (Exploration)")
        stats_logger.add_line("-" * 60)
        
        borderline = df[
            (df[f'prob_cluster_{strong_cluster}'] > 0.3) &
            (df[f'prob_cluster_{strong_cluster}'] < 0.7) &
            (df[f'prob_cluster_{moderate_cluster}'] > 0.3)
        ].copy()
        
        stats_logger.add_line(f"Selection criteria:")
        stats_logger.add_line(f"  0.3 < P(Strong) < 0.7")
        stats_logger.add_line(f"  P(Moderate) > 0.3")
        stats_logger.add_line(f"  High uncertainty (entropy)")
        stats_logger.add_blank()
        
        # Exclude already selected
        already_selected_idx = pd.concat(selected_list).index
        borderline = borderline[~borderline.index.isin(already_selected_idx)]
        
        stats_logger.add_line(f"Borderline molecules (before selection): {len(borderline):,}")
        
        n_tier3 = min(100, len(borderline))
        tier3 = borderline.nlargest(n_tier3, cnn_col).copy()
        tier3['selection_tier'] = 'Tier3_Borderline'
        tier3['selection_reason'] = 'Uncertain between strong/moderate clusters'
        selected_list.append(tier3)
        
        print(f"Selected: {len(tier3):,} molecules (from {len(borderline)} borderline)")
        print(f"CNN_VS range: [{tier3[cnn_col].min():.2f}, {tier3[cnn_col].max():.2f}]")
        
        stats_logger.add_line(f"Selected (top by CNN_VS): {len(tier3):,}")
        stats_logger.add_line(f"CNN_VS statistics:")
        stats_logger.add_line(f"  Mean:    {tier3[cnn_col].mean():.6f}")
        stats_logger.add_line(f"  Minimum: {tier3[cnn_col].min():.6f}")
        stats_logger.add_line(f"  Maximum: {tier3[cnn_col].max():.6f}")
        stats_logger.add_line(f"Mean entropy: {tier3['cluster_uncertainty'].mean():.6f}")
        stats_logger.add_line(f"Mean P(Strong): {tier3[f'prob_cluster_{strong_cluster}'].mean():.6f}")
        stats_logger.add_blank()
    
    # =========================================================================
    # TIER 4: Negative controls
    # =========================================================================
    print("\n--- TIER 4: Negative Controls ---")
    stats_logger.add_line("TIER 4: Negative Controls")
    stats_logger.add_line("-" * 60)
    
    weak_pool = df[
        (df['gmm_cluster'] == weak_cluster) &
        (df[f'prob_cluster_{weak_cluster}'] > 0.85)
    ]
    
    stats_logger.add_line(f"Selection criteria:")
    stats_logger.add_line(f"  Cluster: {weak_cluster} ({cluster_to_category[weak_cluster]})")
    stats_logger.add_line(f"  P(Weak) > 0.85")
    stats_logger.add_line(f"  Random sample of 50")
    stats_logger.add_blank()
    
    stats_logger.add_line(f"Pool of eligible molecules: {len(weak_pool):,}")
    
    n_controls = min(50, len(weak_pool))
    tier4 = weak_pool.sample(n=n_controls, random_state=42).copy()
    tier4['selection_tier'] = 'Tier4_NegativeControls'
    tier4['selection_reason'] = 'Negative controls from weak cluster'
    selected_list.append(tier4)
    
    print(f"Selected: {len(tier4):,} molecules")
    print(f"CNN_VS range: [{tier4[cnn_col].min():.2f}, {tier4[cnn_col].max():.2f}]")
    
    stats_logger.add_line(f"Selected (random sample): {len(tier4):,}")
    stats_logger.add_line(f"CNN_VS statistics:")
    stats_logger.add_line(f"  Mean:    {tier4[cnn_col].mean():.6f}")
    stats_logger.add_line(f"  Minimum: {tier4[cnn_col].min():.6f}")
    stats_logger.add_line(f"  Maximum: {tier4[cnn_col].max():.6f}")
    stats_logger.add_line(f"Mean P(Weak): {tier4[f'prob_cluster_{weak_cluster}'].mean():.6f}")
    stats_logger.add_blank()
    
    # =========================================================================
    # Combine all tiers
    # =========================================================================
    selected = pd.concat(selected_list)
    
    # VERIFICATION: Check for duplicates
    n_duplicates = selected.duplicated(subset='Compound').sum()
    if n_duplicates > 0:
        print(f"\n⚠ WARNING: Found {n_duplicates} duplicate molecules!")
        print("  Removing duplicates (keeping first occurrence)...")
        selected = selected.drop_duplicates(subset='Compound', keep='first')
        stats_logger.add_line(f"WARNING: Found and removed {n_duplicates} duplicate molecules")
    else:
        print(f"\n✓ No duplicates found - all {len(selected):,} molecules are unique")
        stats_logger.add_line(f"✓ No duplicate molecules found")
    
    stats_logger.add_blank()
    stats_logger.add_line(f"TOTAL SELECTED: {len(selected):,} unique molecules")
    
    return selected


def generate_summary(df, selected, stats_logger, cnn_col='CNN_VS'):
    """Generate summary statistics"""
    print("\n" + "="*80)
    print("STEP 7: FINAL SUMMARY")
    print("="*80)
    
    stats_logger.add_section("STEP 7: FINAL SUMMARY AND ENRICHMENT ANALYSIS")
    
    print(f"\n✓ TOTAL SELECTED: {len(selected):,} unique molecules")
    print(f"  Selection rate: {100*len(selected)/len(df):.1f}% of unique compounds")
    
    stats_logger.add_line(f"Total molecules analyzed: {len(df):,}")
    stats_logger.add_line(f"Total molecules selected: {len(selected):,}")
    stats_logger.add_line(f"Selection rate: {100*len(selected)/len(df):.2f}%")
    stats_logger.add_blank()
    
    print(f"\nDistribution by tier:")
    tier_summary = selected['selection_tier'].value_counts().sort_index()
    
    stats_logger.add_line("Distribution by tier:")
    stats_logger.add_line("-" * 60)
    for tier, count in tier_summary.items():
        pct = 100 * count / len(selected)
        print(f"  {tier:30s}: {count:5,} ({pct:5.1f}%)")
        stats_logger.add_line(f"  {tier}: {count:,} ({pct:.2f}%)")
    
    stats_logger.add_blank()
    
    print(f"\nCNN_VS score comparison:")
    print(f"  All molecules:")
    print(f"    Mean:   {df[cnn_col].mean():.3f}")
    print(f"    Median: {df[cnn_col].median():.3f}")
    print(f"    Range:  [{df[cnn_col].min():.2f}, {df[cnn_col].max():.2f}]")
    
    print(f"\n  Selected molecules:")
    print(f"    Mean:   {selected[cnn_col].mean():.3f}")
    print(f"    Median: {selected[cnn_col].median():.3f}")
    print(f"    Range:  [{selected[cnn_col].min():.2f}, {selected[cnn_col].max():.2f}]")
    
    enrichment = selected[cnn_col].mean() / df[cnn_col].mean()
    print(f"\n  Enrichment factor: {enrichment:.2f}x")
    
    # Detailed statistics in log
    stats_logger.add_line("CNN_VS Score Comparison:")
    stats_logger.add_line("="*60)
    stats_logger.add_line("ALL MOLECULES:")
    stats_logger.add_line(f"  Count:          {len(df):,}")
    stats_logger.add_line(f"  Mean:           {df[cnn_col].mean():.6f}")
    stats_logger.add_line(f"  Std Dev:        {df[cnn_col].std():.6f}")
    stats_logger.add_line(f"  Median:         {df[cnn_col].median():.6f}")
    stats_logger.add_line(f"  Minimum:        {df[cnn_col].min():.6f}")
    stats_logger.add_line(f"  Maximum:        {df[cnn_col].max():.6f}")
    stats_logger.add_line(f"  25th percentile: {df[cnn_col].quantile(0.25):.6f}")
    stats_logger.add_line(f"  75th percentile: {df[cnn_col].quantile(0.75):.6f}")
    stats_logger.add_blank()
    
    stats_logger.add_line("SELECTED MOLECULES:")
    stats_logger.add_line(f"  Count:          {len(selected):,}")
    stats_logger.add_line(f"  Mean:           {selected[cnn_col].mean():.6f}")
    stats_logger.add_line(f"  Std Dev:        {selected[cnn_col].std():.6f}")
    stats_logger.add_line(f"  Median:         {selected[cnn_col].median():.6f}")
    stats_logger.add_line(f"  Minimum:        {selected[cnn_col].min():.6f}")
    stats_logger.add_line(f"  Maximum:        {selected[cnn_col].max():.6f}")
    stats_logger.add_line(f"  25th percentile: {selected[cnn_col].quantile(0.25):.6f}")
    stats_logger.add_line(f"  75th percentile: {selected[cnn_col].quantile(0.75):.6f}")
    stats_logger.add_blank()
    
    stats_logger.add_line("ENRICHMENT ANALYSIS:")
    stats_logger.add_line(f"  Mean enrichment factor:   {enrichment:.4f}x")
    stats_logger.add_line(f"  Median enrichment factor: {selected[cnn_col].median() / df[cnn_col].median():.4f}x")
    stats_logger.add_line(f"  Mean difference:          {selected[cnn_col].mean() - df[cnn_col].mean():.6f}")
    stats_logger.add_blank()
    
    # Percentile coverage
    stats_logger.add_line("PERCENTILE COVERAGE:")
    stats_logger.add_line("(What percentile of all molecules does the selection span?)")
    min_selected_percentile = (df[cnn_col] < selected[cnn_col].min()).sum() / len(df) * 100
    max_selected_percentile = (df[cnn_col] <= selected[cnn_col].max()).sum() / len(df) * 100
    
    stats_logger.add_line(f"  Lowest selected molecule:  {min_selected_percentile:.2f}th percentile")
    stats_logger.add_line(f"  Highest selected molecule: {max_selected_percentile:.2f}th percentile")
    stats_logger.add_line(f"  Coverage: {min_selected_percentile:.2f}th to {max_selected_percentile:.2f}th percentile")


def save_results(df, selected, output_dir, output_prefix, stats_logger):
    """Save results to CSV files"""
    print("\n" + "="*80)
    print("STEP 8: SAVING RESULTS")
    print("="*80)
    
    stats_logger.add_section("STEP 8: OUTPUT FILES")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Select columns for output
    output_cols = [
        'Compound', 'Pose', 'CNNscore', 'CNNaffinity', 'CNN_VS',
        'gmm_cluster', 'cluster_category', 'max_cluster_prob',
        'cluster_uncertainty', 'selection_tier', 'selection_reason'
    ]
    
    if 'reference_active' in selected.columns:
        output_cols.append('reference_active')
    
    available_cols = [col for col in output_cols if col in selected.columns]
    
    # Save selected molecules
    selected_file = os.path.join(output_dir, f"{output_prefix}_for_GFN2_xTB.csv")
    selected_output = selected[available_cols].sort_values('CNN_VS', ascending=False)
    selected_output.to_csv(selected_file, index=False)
    print(f"✓ Selected molecules saved to: {selected_file}")
    
    # Verify no duplicates in saved file
    verify_df = pd.read_csv(selected_file)
    n_dupes = verify_df.duplicated(subset='Compound').sum()
    print(f"  Verified: {len(verify_df):,} rows, {n_dupes} duplicates")
    
    stats_logger.add_line(f"1. Selected molecules CSV: {selected_file}")
    stats_logger.add_line(f"   Rows: {len(verify_df):,}")
    stats_logger.add_line(f"   Duplicates: {n_dupes}")
    stats_logger.add_blank()
    
    # Save full analysis
    full_file = os.path.join(output_dir, f"{output_prefix}_full_analysis.csv")
    df.to_csv(full_file, index=False)
    print(f"✓ Full analysis saved to: {full_file}")
    
    stats_logger.add_line(f"2. Full analysis CSV: {full_file}")
    stats_logger.add_line(f"   Rows: {len(df):,}")
    stats_logger.add_blank()
    
    # Save summary
    summary_file = os.path.join(output_dir, f"{output_prefix}_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MOLECULE SELECTION SUMMARY - STRATEGY A\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total molecules analyzed: {len(df):,}\n")
        f.write(f"Total molecules selected: {len(selected):,}\n")
        f.write(f"Selection rate: {100*len(selected)/len(df):.1f}%\n\n")
        f.write("Selection by tier:\n")
        tier_summary = selected['selection_tier'].value_counts().sort_index()
        for tier, count in tier_summary.items():
            pct = 100 * count / len(selected)
            f.write(f"  {tier}: {count:,} ({pct:.1f}%)\n")
    
    print(f"✓ Summary saved to: {summary_file}")
    
    stats_logger.add_line(f"3. Brief summary TXT: {summary_file}")
    stats_logger.add_blank()
    
    # Save detailed statistics
    stats_file = os.path.join(output_dir, f"{output_prefix}_detailed_statistics.txt")
    stats_logger.save(stats_file)
    print(f"✓ Detailed statistics saved to: {stats_file}")
    
    stats_logger.add_line(f"4. Detailed statistics TXT: {stats_file}")
    stats_logger.add_line("   (THIS FILE - contains all GMM parameters, scores, and intermediate values)")
    
    return selected_file, full_file, summary_file, stats_file


def create_visualizations(df, selected, cluster_df, output_dir, output_prefix):
    """Create visualization plots with FIXED label overlaps"""
    print("\n" + "="*80)
    print("STEP 9: GENERATING VISUALIZATIONS")
    print("="*80)
    
    fig = plt.figure(figsize=(22, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
    
    clusters_sorted = cluster_df.sort_values('Mean')['Cluster_ID'].tolist()
    cluster_to_category = dict(zip(cluster_df['Cluster_ID'], cluster_df['Category']))
    colors_cluster = ['#e74c3c', '#f39c12', '#27ae60', '#3498db']
    
    known_actives = {'BP79': 3.48302, 'Galbelgin': 3.24381, 'Fragment_3': 3.24966}
    
    # Plot 1: Distribution with GMM clusters - FIXED LABEL POSITIONS
    ax1 = fig.add_subplot(gs[0, :2])
    for i, cluster_id in enumerate(clusters_sorted):
        cluster_data = df[df['gmm_cluster'] == cluster_id]['CNN_VS']
        category = cluster_to_category[cluster_id]
        ax1.hist(cluster_data, bins=60, alpha=0.6,
                label=f'{category} (n={len(cluster_data):,}, μ={cluster_data.mean():.2f})',
                color=colors_cluster[i % len(colors_cluster)])
    
    # FIXED: Better positioning of known active labels
    y_max = ax1.get_ylim()[1]
    label_positions = [0.95, 0.85, 0.75]
    for idx, (name, score) in enumerate(known_actives.items()):
        ax1.axvline(score, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.text(score, y_max * label_positions[idx], name, 
                rotation=0,
                va='top', ha='left', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='black'))
    
    ax1.set_xlabel('CNN_VS Score', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=13, fontweight='bold')
    ax1.set_title('GMM Clustering of CNN_VS Scores', fontsize=15, fontweight='bold', pad=15)
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax1.grid(alpha=0.3)
    
    # Plot 2: Selected vs All
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['CNN_VS'], bins=60, alpha=0.4, label=f'All (n={len(df):,})', color='gray', density=True)
    ax2.hist(selected['CNN_VS'], bins=60, alpha=0.7, label=f'Selected (n={len(selected):,})', 
            color='#2ecc71', density=True)
    ax2.axvline(selected['CNN_VS'].mean(), color='darkgreen', linestyle='--', linewidth=2,
               label=f'Selected μ={selected["CNN_VS"].mean():.2f}')
    ax2.set_xlabel('CNN_VS Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax2.set_title('Selection Overview', fontsize=13, fontweight='bold', pad=15)
    ax2.legend(fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3)
    
    # Plot 3: Box plot
    ax3 = fig.add_subplot(gs[1, 0])
    cluster_box_data = [df[df['gmm_cluster'] == c]['CNN_VS'].values for c in clusters_sorted]
    bp = ax3.boxplot(cluster_box_data, labels=[cluster_to_category[c] for c in clusters_sorted], 
                     patch_artist=True)
    for patch, color in zip(bp['boxes'], colors_cluster):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax3.set_ylabel('CNN_VS Score', fontsize=12, fontweight='bold')
    ax3.set_title('Score Distribution by Cluster', fontsize=13, fontweight='bold', pad=15)
    ax3.grid(alpha=0.3, axis='y')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=10)
    
    # Plot 4: Selection by tier - FIXED LABELS
    ax4 = fig.add_subplot(gs[1, 1])
    tier_counts = selected['selection_tier'].value_counts().sort_index()
    colors_tier = plt.cm.viridis(np.linspace(0, 0.9, len(tier_counts)))
    
    tier_labels_short = []
    for tier in tier_counts.index:
        if 'Tier1' in tier:
            tier_labels_short.append('Tier 1\nHigh Conf')
        elif 'Tier2B' in tier:
            tier_labels_short.append('Tier 2B\nValidation')
        elif 'Tier3' in tier:
            tier_labels_short.append('Tier 3\nBorderline')
        elif 'Tier4' in tier:
            tier_labels_short.append('Tier 4\nControls')
        else:
            tier_labels_short.append(tier.replace('_', '\n'))
    
    ax4.barh(range(len(tier_counts)), tier_counts.values, color=colors_tier)
    ax4.set_yticks(range(len(tier_counts)))
    ax4.set_yticklabels(tier_labels_short, fontsize=9)
    ax4.set_xlabel('Number of Molecules', fontsize=12, fontweight='bold')
    ax4.set_title('Selection Count by Tier', fontsize=13, fontweight='bold', pad=15)
    ax4.grid(alpha=0.3, axis='x')
    
    for i, v in enumerate(tier_counts.values):
        ax4.text(v + max(tier_counts.values)*0.02, i, f'{v:,}', 
                va='center', fontweight='bold', fontsize=10)
    
    # Plot 5: Cluster probability
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(df['max_cluster_prob'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax5.axvline(0.7, color='red', linestyle='--', linewidth=2, label='Threshold (0.7)')
    ax5.set_xlabel('Max Cluster Probability', fontsize=12, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax5.set_title('Cluster Assignment\nConfidence', fontsize=13, fontweight='bold', pad=15)
    ax5.legend(fontsize=9, framealpha=0.9)
    ax5.grid(alpha=0.3)
    
    # Plot 6: Cumulative distribution
    ax6 = fig.add_subplot(gs[2, :2])
    sorted_all = np.sort(df['CNN_VS'])
    sorted_sel = np.sort(selected['CNN_VS'])
    ax6.plot(sorted_all, np.linspace(0, 100, len(sorted_all)), 
            linewidth=2.5, color='gray', alpha=0.7, label='All molecules')
    ax6.plot(sorted_sel, np.linspace(0, 100, len(sorted_sel)), 
            linewidth=2.5, color='#2ecc71', label='Selected molecules')
    
    for pct in [50, 75, 90, 95]:
        val = np.percentile(sorted_all, pct)
        ax6.axhline(pct, color='red', linestyle=':', alpha=0.3, linewidth=1)
        ax6.axvline(val, color='red', linestyle=':', alpha=0.3, linewidth=1)
        ax6.text(val + 0.15, pct - 3, f'P{pct}={val:.2f}', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.8, edgecolor='gray'))
    
    ax6.set_xlabel('CNN_VS Score', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Cumulative Percentage', fontsize=13, fontweight='bold')
    ax6.set_title('Cumulative Distribution Comparison', fontsize=15, fontweight='bold', pad=15)
    ax6.legend(fontsize=11, framealpha=0.9)
    ax6.grid(alpha=0.3)
    
    # Plot 7: Pie chart - FIXED LABELS
    ax7 = fig.add_subplot(gs[2, 2])
    tier_pcts = 100 * tier_counts / len(selected)
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(tier_counts)))
    
    wedges, texts, autotexts = ax7.pie(
        tier_pcts, 
        labels=None,
        autopct='%1.1f%%',
        colors=colors_pie, 
        startangle=90,
        pctdistance=0.85
    )
    
    plt.setp(autotexts, size=10, weight='bold')
    
    legend_labels = [f'{tier.split("_")[0]}: {count:,}' for tier, count in zip(tier_counts.index, tier_counts.values)]
    ax7.legend(legend_labels, loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), 
              fontsize=9, framealpha=0.9)
    ax7.set_title('Selection Distribution', fontsize=13, fontweight='bold', pad=15)
    
    fig.suptitle(f'Hybrid GMM + Z-Score Selection Analysis - Strategy A\n'
                 f'Total: {len(selected):,}/{len(df):,} unique molecules selected ({100*len(selected)/len(df):.1f}%)',
                 fontsize=17, fontweight='bold', y=0.997)
    
    viz_file = os.path.join(output_dir, f"{output_prefix}_analysis.png")
    plt.savefig(viz_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Visualization saved to: {viz_file}")
    plt.close()


def main():
    """Main pipeline"""
    # Initialize statistics logger
    stats_logger = StatisticsLogger()
    
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  GMM + Z-Score Hybrid Selection - Strategy A (~1,600)".center(78) + "║")
    print("║" + "  FINAL VERSION - Complete Statistical Output".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")
    
    print(f"Configuration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Output prefix: {OUTPUT_PREFIX}")
    print(f"  Target Tier 1 size: {TARGET_TIER1_SIZE}")
    print()
    
    stats_logger.add_section("CONFIGURATION")
    stats_logger.add_line(f"Input file: {INPUT_FILE}")
    stats_logger.add_line(f"Output directory: {OUTPUT_DIR}")
    stats_logger.add_line(f"Output prefix: {OUTPUT_PREFIX}")
    stats_logger.add_line(f"Target Tier 1 size: {TARGET_TIER1_SIZE}")
    stats_logger.add_line(f"Strategy: A (Comprehensive)")
    
    # Load data
    df = load_data(INPUT_FILE, stats_logger)
    
    # Select best poses
    df_best = select_best_poses(df, stats_logger)
    
    # Fit GMM
    gmm, optimal_n = fit_gmm(df_best, stats_logger)
    
    # Assign clusters
    df_best, cluster_df = assign_clusters(df_best, gmm, stats_logger)
    
    # Validate with known actives
    validate_known_actives(df_best, cluster_df, stats_logger)
    
    # Select molecules - Strategy A
    selected = select_molecules(df_best, cluster_df, stats_logger, target_tier1=TARGET_TIER1_SIZE)
    
    # Generate summary
    generate_summary(df_best, selected, stats_logger)
    
    # Save results
    selected_file, full_file, summary_file, stats_file = save_results(
        df_best, selected, OUTPUT_DIR, OUTPUT_PREFIX, stats_logger)
    
    # Create visualizations
    create_visualizations(df_best, selected, cluster_df, OUTPUT_DIR, OUTPUT_PREFIX)
    
    # Final message
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput files in '{OUTPUT_DIR}' directory:")
    print(f"  1. {OUTPUT_PREFIX}_for_GFN2_xTB.csv - Selected molecules")
    print(f"  2. {OUTPUT_PREFIX}_full_analysis.csv - Full analysis")
    print(f"  3. {OUTPUT_PREFIX}_summary.txt - Brief summary")
    print(f"  4. {OUTPUT_PREFIX}_detailed_statistics.txt - ALL GMM PARAMETERS & SCORES ⭐")
    print(f"  5. {OUTPUT_PREFIX}_analysis.png - Visualizations")
    print(f"\nThe detailed statistics file contains:")
    print(f"  • All GMM parameters (means, variances, weights)")
    print(f"  • BIC/AIC scores for all tested models")
    print(f"  • Complete cluster statistics")
    print(f"  • Known active validation details")
    print(f"  • Selection criteria and thresholds")
    print(f"  • Enrichment analysis")
    print(f"  • And much more!")
    print(f"\n✓ Ready for GFN2-xTB calculations")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"\nMake sure your CSV file exists at the specified path.")
        print(f"Current working directory: {os.getcwd()}")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
