# Manuscript Restructuring Plan: Two-Part Analysis

## Overview
Restructure the manuscript to include two complementary analyses:
1. **Part 1**: Test hypothesis about semantic distinctiveness (current analysis)
2. **Part 2**: Explore what semantic features drive leadership clustering patterns

---

## Part 1: Hypothesis Testing (Current Analysis)

### Formalized Hypothesis

**H1 (Primary Hypothesis)**: Leadership measurement constructs will demonstrate significantly lower semantic distinctiveness compared to established personality constructs, as evidenced by:
- Lower accuracy in distinguishing same-construct from different-construct item pairs
- Smaller effect sizes (Cohen's d) for construct separation
- Greater overlap in embedding space

**Theoretical Rationale**: Despite theoretical claims of distinctiveness, leadership constructs have emerged from overlapping theoretical traditions and share common behavioral referents, leading to semantic redundancy in their operationalization.

**Operational Definition**: Semantic distinctiveness = probability of correctly ranking same-construct pairs as more similar than different-construct pairs using contrastive learning embeddings.

---

## Part 2: Semantic Feature Analysis (New Addition)

### Research Question
**RQ1**: Which semantic features best explain the clustering patterns observed in leadership measurement items, and do these align with classical leadership taxonomies?

### Modern ML Approaches for Feature-Based Cluster Analysis

#### 1. **Theory-Driven Feature Engineering + Interpretable ML**

**Approach**: Code items on theoretical dimensions, then use interpretable ML to predict cluster membership

**Theoretical Dimensions to Code**:
- **Task vs. Relational** (classical leadership taxonomy)
- **Abstract vs. Concrete** (linguistic abstraction)
- **Temporal Focus**: Past/Present/Future oriented
- **Agency Level**: Individual/Dyadic/Group/Organizational
- **Valence**: Positive/Negative/Neutral behaviors
- **Cognitive vs. Behavioral** focus
- **Internal vs. External** attribution

**Implementation**:
```python
# Example workflow
1. Expert coding of items on theoretical dimensions (0-1 scale)
2. Use XGBoost/Random Forest to predict cluster membership
3. Extract SHAP values for feature importance
4. Variance decomposition analysis
```

#### 2. **Canonical Correlation Analysis (CCA) with Deep Learning**

**Modern Approach**: Deep CCA or Kernel CCA
- Maps theoretical codings and embeddings to shared latent space
- Identifies which theoretical dimensions align with embedding clusters
- Quantifies variance explained by each theoretical framework

#### 3. **Supervised Dimensionality Reduction**

**Options**:
- **Linear Discriminant Analysis (LDA)** with theoretical labels
- **Supervised UMAP** using theoretical codings as targets
- **Metric Learning** (e.g., Large Margin Nearest Neighbor)

**Benefit**: Creates interpretable dimensions that maximize separation according to theory

#### 4. **Mixed-Effects Modeling with Cluster Validation**

**Approach**:
```r
# Multilevel model predicting similarity from theoretical features
similarity ~ task_focus * relational_focus + 
             abstract_level + valence + 
             (1|construct) + (1|item_pair)
```

**Outputs**:
- Variance components for each theoretical dimension
- Interaction effects between dimensions
- R² decomposition by theoretical framework

#### 5. **Prototype Theory Analysis**

**Novel Approach**:
1. Identify cluster prototypes (most central items)
2. Analyze linguistic/semantic features of prototypes
3. Use transformer attention weights to identify distinguishing features
4. Compare to theoretical predictions

### Suggested Implementation Plan

#### Step 1: Theoretical Coding
- Develop coding scheme with 5-7 theoretical dimensions
- Have 2-3 expert coders rate all items
- Calculate inter-rater reliability (ICC > 0.80)

#### Step 2: Cluster Extraction
- Apply hierarchical clustering to embeddings
- Use silhouette analysis to determine optimal clusters
- Validate against original construct labels

#### Step 3: Feature Importance Analysis
- Train XGBoost/CatBoost to predict cluster membership from theoretical features
- Extract SHAP values for global feature importance
- Calculate variance explained (R², adjusted R²)

#### Step 4: Theoretical Validation
- Test if classical taxonomies (task/relational) emerge
- Compare to modern frameworks (transformational/transactional)
- Identify novel patterns not captured by existing theory

### Expected Outcomes

**Scenario A**: Classical task/relational taxonomy explains most variance
- Validates traditional leadership theory
- Suggests measurement redundancy within categories

**Scenario B**: Abstract/concrete dimension dominates
- Indicates linguistic rather than conceptual differences
- Questions construct validity

**Scenario C**: Novel patterns emerge
- Opportunity for new theoretical framework
- Data-driven taxonomy development

### Statistical Power Considerations
- With 434 leadership items across 11 constructs
- Sufficient for 5-7 theoretical predictors
- Can detect medium effect sizes (f² > 0.15)

### Visualization Strategy
1. **Feature Importance Plot**: SHAP summary plot
2. **Variance Decomposition**: Stacked bar chart by theory
3. **Decision Tree**: Interpretable clustering rules
4. **2D Projection**: Colored by dominant theoretical feature

---

## Integration with Current Manuscript

### Modified Results Section Structure

```
# Results

## Part 1: Testing Construct Distinctiveness

### Hypothesis Testing
[Current results - 87.4% vs 62.9% accuracy]

## Part 2: Understanding Clustering Patterns

### Theoretical Feature Analysis
- Feature coding reliability
- Cluster extraction and validation
- Feature importance results
- Variance decomposition by theory

### Key Finding
"While leadership constructs show poor semantic distinctiveness (Part 1), 
their clustering patterns are primarily explained by [dominant feature], 
accounting for X% of variance, rather than claimed theoretical distinctions."
```

### Discussion Implications
- Reconceptualize leadership taxonomy based on empirical patterns
- Practical guidelines for measure selection
- Theory development recommendations

---

## Next Steps
1. Finalize theoretical coding scheme
2. Recruit expert coders
3. Implement analysis pipeline
4. Update manuscript with two-part structure 