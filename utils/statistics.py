import numpy as np
from scipy import stats
import pandas as pd
from statsmodels.stats.multitest import multipletests

def perform_differential_analysis(data, group1, group2, method="ttest"):
    """Perform differential expression analysis."""
    results = []
    
    for gene in data.index:
        g1_data = data.loc[gene, group1]
        g2_data = data.loc[gene, group2]
        
        if method == "ttest":
            stat, pval = stats.ttest_ind(g1_data, g2_data)
        elif method == "mannwhitney":
            stat, pval = stats.mannwhitneyu(g1_data, g2_data)
            
        log2fc = np.log2(g2_data.mean() / g1_data.mean())
        
        results.append({
            'gene': gene,
            'log2fc': log2fc,
            'pvalue': pval,
            'statistic': stat
        })
    
    results_df = pd.DataFrame(results)
    results_df['padj'] = multipletests(results_df['pvalue'], method='fdr_bh')[1]
    
    return results_df

def calculate_sample_correlations(data):
    """Calculate correlation matrix between samples."""
    return data.corr()

def perform_enrichment_analysis(gene_list, background_list, annotation_dict):
    """Perform basic gene set enrichment analysis."""
    results = []
    
    for term, term_genes in annotation_dict.items():
        overlap = set(gene_list) & set(term_genes)
        
        # Fisher's exact test
        table = np.array([
            [len(overlap), len(term_genes) - len(overlap)],
            [len(gene_list) - len(overlap), len(background_list) - len(gene_list) - (len(term_genes) - len(overlap))]
        ])
        
        _, pval = stats.fisher_exact(table)
        
        results.append({
            'term': term,
            'overlap_size': len(overlap),
            'pvalue': pval
        })
    
    return pd.DataFrame(results)
