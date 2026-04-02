# Data Science project: Customer Segmentation — Online Retail II

RFM-based customer segmentation using KMeans clustering on the UCI Online Retail II dataset (2009–2011, ~1M transactions). This project segments ~5,850 customers into five actionable groups to enable targeted marketing strategies.

---

## Pipeline

```
Raw Excel → Cleaning → Feature Engineering → Outlier Removal
→ Yeo-Johnson Transform → PCA (3 components) → KMeans (k=5)
→ Outlier Re-integration → Visualization and Recommendations
```

---

## Data Cleaning

Raw data contains 1,067,371 transactions across two sheets. Cleaning steps include filtering invoices to standard 6-digit format (removing cancellations prefixed with `C` and accounting adjustments prefixed with `A`), filtering stock codes to valid 5-digit product codes only, removing rows with zero or negative prices and quantities, dropping missing Customer IDs, and removing duplicates. Final cleaned dataset retains 776,577 rows — approximately 73% of the original.

Schema validation via Pandera is enforced both before and after cleaning to guarantee column types and value constraints.

---

## Feature Engineering

Each customer is aggregated into five features:

- **Recency** — days since last purchase relative to the reference date (2011-12-09)
- **Frequency** — count of unique invoices
- **MonetaryValue** — total spend across all orders
- **AOV** — average order value (MonetaryValue / Frequency)
- **Tenure** — days between first and last purchase

Tenure was ultimately dropped before clustering due to extreme multicollinearity with Frequency (r=0.88) and MonetaryValue (r=0.75), poor data distribution even after transformation, and because it is a passive metric that does not reflect current customer intent.

---

## Outlier Handling

Outliers are identified per feature using the IQR method (1.5× fence) on MonetaryValue, Frequency, and AOV. Outlier customers are not discarded — they are excluded from KMeans training to prevent centroid distortion, then assigned to clusters using the trained model and flagged separately. This preserves all customers in the final output while keeping the model robust.

---

## Transformation and Dimensionality Reduction

All features are heavily right-skewed, confirmed by skewness/kurtosis descriptive stats and visualizations. Yeo-Johnson power transformation is applied to normalize distributions. A secondary robust Z-score check (using median and MAD) confirms that remaining post-transformation outliers are negligible in both count and impact.

PCA reduces the four transformed features to three components explaining ~99.8% of variance, decorrelating the high multicollinearity between MonetaryValue and Frequency (r=0.85) before clustering.

---

## Clustering

KMeans is evaluated for k=2 through k=8 using inertia, global silhouette score, Davies-Bouldin index, and per-cluster silhouette distributions visualized as boxplots for each k.

No single metric unanimously agrees on an optimal k, so the final choice integrates statistical evidence with business interpretability.

k=5 is selected. It wins two of three metrics against k=4 (Davies-Bouldin and Silhouette), and k=4 is further ruled out because it merges At-Risk High-Value and At-Risk Frequent into a single segment — two groups that differ critically in AOV (£426 vs £216) and require entirely different marketing strategies. Convergence is confirmed at 14 iterations; max_iter=50 is retained 
as a safety margin.

GMM was evaluated across all four covariance types (spherical, tied, diag, full) for k=2–8 on pre-PCA transformed features — PCA was intentionally skipped for GMM, as it destroys the elliptical correlation structure GMM is designed to exploit. KMeans narrowly wins silhouette (0.301 vs 0.280); GMM wins Davies-Bouldin (1.000 vs 1.057), but the margin is negligible. Spherical GMM performing best reinforces that the RFM feature space, after Yeo-Johnson, is compact enough for spherical assumptions — meaning KMeans' assumption holds and GMM's elliptical flexibility adds no practical benefit. KMeans is also preferred for producing hard, actionable labels suited to the marketing use case.

---

## Segments

**VIP** (2,060 customers) — £6,965 avg spend, 13.4 orders, 51 days recency. Accounts for 84.1% of total revenue. 722 of these customers are statistical outliers, likely wholesale or B2B buyers — generating 64.8% of total revenue alone vs 19.3% for the core VIP segment. Loyalty programs, early access, and dedicated account managers for the outlier sub-group.

**Loyal** (855 customers) — £521 avg spend, 2.7 orders, 31 days recency. Most recently active segment after VIP. Cross-sell and upsell to migrate toward VIP.

**At-Risk High-Value** (1,074 customers) — £1,252 avg spend, 1.5 orders, 338 days recency, highest AOV at £740. They spend big when they buy but are now disengaged. Personalized win-back campaigns before full churn.

**At-Risk Frequent** (839 customers) — £909 avg spend, 4.4 orders, 299 days recency. Were regular buyers now disengaged. Bundle deals and volume incentives to reactivate purchase habits.

**Churned** (1,024 customers) — £165 avg spend, 1.3 orders, 410 days recency. Lowest value segment. Low-cost automated email only; deprioritize after 2–3 attempts with no response.

---

## Model Validation

Temporal consistency is verified by splitting the dataset at December 2010 and computing segment distributions independently on each half — no segment shifts by more than 1.6 percentage points. Cluster stability is confirmed via Adjusted Rand Index across four random seeds (ARI ≥ 0.99). Bootstrap stability is confirmed across 100 resampled iterations (mean ARI: 0.857 ± 0.082).  Statistical separation between clusters is confirmed by Kruskal-Wallis tests (p ≈ 0 for all four features). An end-to-end inference demo validates that seven synthetic customers with known profiles each land in the correct segment.


---

## Outputs

- `outputs/customer_segments.xlsx` — full clustered customer dataset for the marketing team
- `outputs/cluster_summary.xlsx` — cluster mean statistics
- `artifacts/kmeans_k5.pkl` — trained KMeans model
- `artifacts/power_transformer.pkl` — fitted Yeo-Johnson transformer
- `artifacts/pca.pkl` — fitted PCA (3 components)
- `artifacts/iqr_bounds.pkl` — outlier detection bounds
- `artifacts/reference_date.pkl` — training reference date
- `artifacts/cluster_labels_names.pkl` — cluster index to segment name mapping

All artifacts enable end-to-end inference on new customers without retraining.

---

## Limitations

Silhouette scores are weak (0.28–0.36), which is expected given that customer behavior is inherently continuous and cluster overlap is unavoidable. 

Data covers 2009–2011 and retraining is required for current customer behavior. 
The reference date is fixed to the training data, meaning Recency values will inflate systematically over time in production without updating it.

MonetaryValue = AOV × Frequency is an algebraic identity introducing feature redundancy, partially mitigated by PCA decorrelation but not fully eliminated.

---

## Stack

`pandas` · `numpy` · `scikit-learn` · `seaborn` · `matplotlib` · `scipy` · `pandera` · `joblib`

---

**Notes**
- The purpose of this code is to explicitly demonstrate the reasoning behind every choice balanced between statistics and business interpretability without deleting post-hoc known facts (for example tenure is preserved initally then dropped instead of not being included in the first place)
- The code includes descriptive markdown cells throughout, and every decision is justified visually, statistically, and from a business perspective.
