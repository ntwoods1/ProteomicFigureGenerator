import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

def create_interactive_volcano(df, x_col, y_col, gene_col, cutoffs):
    """
    Create interactive volcano plot with hover effects.

    Args:
        df (pd.DataFrame): Input dataframe
        x_col (str): Column name for log2 fold change values
        y_col (str): Column name for p-values
        gene_col (str): Column name for gene identifiers (optional)
        cutoffs (dict): Dictionary with 'p_value' and 'fold_change' cutoffs
    """
    plot_df = df.copy()

    # Ensure required columns exist
    if x_col not in plot_df.columns:
        raise ValueError(f"Column '{x_col}' not found in dataframe")
    if y_col not in plot_df.columns:
        raise ValueError(f"Column '{y_col}' not found in dataframe")

    # Calculate -log10(p-value) if needed
    if not y_col.startswith('-log10'):
        plot_df['-log10(p-value)'] = -np.log10(plot_df[y_col])
        y_plot_col = '-log10(p-value)'
    else:
        y_plot_col = y_col

    # Create significance categories
    plot_df['significance'] = 'Not Significant'
    plot_df.loc[
        (plot_df[x_col] >= cutoffs['fold_change']) & 
        (plot_df[y_plot_col] >= cutoffs['p_value']), 
        'significance'
    ] = 'Upregulated'
    plot_df.loc[
        (plot_df[x_col] <= -cutoffs['fold_change']) & 
        (plot_df[y_plot_col] >= cutoffs['p_value']), 
        'significance'
    ] = 'Downregulated'

    # Create hover data dictionary
    hover_data = {}
    if gene_col and gene_col in plot_df.columns:
        hover_data[gene_col] = True
    hover_data[x_col] = ':.2f'
    hover_data[y_plot_col] = ':.2f'

    # Create the scatter plot
    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_plot_col,
        color='significance',
        color_discrete_map={
            'Upregulated': 'red',
            'Downregulated': 'blue',
            'Not Significant': 'grey'
        },
        hover_data=hover_data,
        template="simple_white"
    )

    # Update layout
    fig.update_layout(
        title="Volcano Plot",
        xaxis_title="Log2 Fold Change",
        yaxis_title="-Log10(p-value)",
        showlegend=True
    )

    # Add cutoff lines
    fig.add_hline(y=cutoffs['p_value'], line_dash="dash", line_color="red", opacity=0.5)
    fig.add_vline(x=cutoffs['fold_change'], line_dash="dash", line_color="blue", opacity=0.5)
    fig.add_vline(x=-cutoffs['fold_change'], line_dash="dash", line_color="blue", opacity=0.5)

    return fig

def create_pca_plot(pca_result, variance_ratio, labels, conditions):
    """Create PCA plot with confidence ellipses."""
    fig = px.scatter(
        pca_result,
        x="PC1",
        y="PC2",
        color=conditions,
        labels=labels,
        template="simple_white"
    )
    
    fig.update_layout(
        title="PCA Plot",
        xaxis_title=f"PC1 ({variance_ratio[0]:.1f}%)",
        yaxis_title=f"PC2 ({variance_ratio[1]:.1f}%)"
    )
    
    return fig

def create_heatmap(data, row_labels, col_labels):
    """Create clustered heatmap with dendrogram."""
    # Compute linkage
    row_linkage = linkage(data, method='ward')
    
    # Create clustered heatmap
    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=col_labels,
        y=row_labels,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Clustered Heatmap",
        xaxis_title="Samples",
        yaxis_title="Genes",
        height=800
    )
    
    return fig

def create_box_plot(df, value_col, group_col):
    """Create box plot with individual points."""
    fig = px.box(
        df,
        y=value_col,
        x=group_col,
        points="all",
        template="simple_white"
    )
    
    fig.update_layout(
        title="Expression Distribution",
        xaxis_title="Condition",
        yaxis_title="Expression Level"
    )
    
    return fig