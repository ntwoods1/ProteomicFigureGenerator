
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from typing import Tuple, List, Dict, Optional

class Visualizer:
    @staticmethod
    def create_heatmap(df: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        """
        Create interactive heatmap
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))

        fig.update_layout(
            title=title,
            width=800,
            height=800
        )

        return fig

    @staticmethod
    def create_cv_histogram(df: pd.DataFrame, group_selections: dict, cutoff: float = 0.2) -> go.Figure:
        """
        Create histogram of CV values for each group with cutoff line

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            cutoff: CV cutoff value (0-1)

        Returns:
            Plotly figure with CV histograms
        """
        # Calculate number of groups for subplot layout
        num_groups = len(group_selections)
        if num_groups == 0:
            return go.Figure().update_layout(title="No groups defined")

        # Create subplot grid (1 row per group)
        fig = make_subplots(rows=num_groups, cols=1, 
                            subplot_titles=[f"Group: {group}" for group in group_selections.keys()],
                            vertical_spacing=0.1)

        # Process each group
        for i, (group_name, columns) in enumerate(group_selections.items(), 1):
            # Skip if no columns in group
            if not columns:
                continue

            # Calculate CV for this group
            group_data = df[columns]
            # Handle division by zero and NaN values
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_values = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_values.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Remove NaN values for plotting
            cv_values = cv_values.dropna()

            # Skip if all values are NaN
            if len(cv_values) == 0:
                continue

            # Create histogram trace
            hist_trace = go.Histogram(
                x=cv_values,
                name=group_name,
                nbinsx=50,
                marker_color=f'rgba({(i*60)%255}, {(i*100)%255}, {(i*160)%255}, 0.7)'
            )

            # Calculate histogram values manually with explicit range to avoid NaN issues
            hist_vals, bin_edges = np.histogram(
                cv_values, 
                bins=50, 
                range=(0, min(3, cv_values.max() * 1.2) if not cv_values.empty else 3)
            )
            max_count = max(hist_vals) * 1.1 if len(hist_vals) > 0 else 10  # Add 10% for visibility

            # Create more visible vertical line for cutoff
            cutoff_line = go.Scatter(
                x=[cutoff, cutoff],
                y=[0, max_count],
                mode='lines',
                name=f'CV Cutoff: {cutoff:.2f}',
                line=dict(color='red', width=3, dash='dash')
            )

            # Add traces to subplot
            fig.add_trace(hist_trace, row=i, col=1)
            fig.add_trace(cutoff_line, row=i, col=1)

            # Update layout for this subplot
            fig.update_xaxes(title_text="Coefficient of Variation (CV)", row=i, col=1, range=[0, min(3, cv_values.max()*1.2) if not cv_values.empty else 3])
            fig.update_yaxes(title_text="Count", row=i, col=1)

        # Update overall layout
        fig.update_layout(
            title="Distribution of Coefficient of Variation (CV) by Group",
            height=300*num_groups,
            width=800,
            showlegend=False
        )

        return fig

    @staticmethod
    def get_figure_as_svg(fig: go.Figure) -> str:
        """
        Convert a plotly figure to SVG format

        Args:
            fig: Plotly figure to convert

        Returns:
            SVG data as string
        """
        try:
            return fig.to_image(format="svg").decode("utf-8")
        except Exception as e:
            # Fallback message for debugging
            return f'''<?xml version="1.0" encoding="utf-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="800" height="600">
    <text x="50" y="50" font-family="sans-serif" font-size="16">Error creating SVG: {str(e)}</text>
</svg>'''

    @staticmethod
    def create_cv_histogram_matplotlib(df: pd.DataFrame, group_selections: dict, cutoff: float = 0.2):
        """
        Create histogram of CV values using matplotlib (for reliable SVG export)

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists
            cutoff: CV cutoff value (0-1)

        Returns:
            Matplotlib figure and a dictionary of CV values by group
        """
        import matplotlib.pyplot as plt
        import io

        # Calculate number of groups for subplot layout
        num_groups = len(group_selections)
        if num_groups == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No groups defined", ha='center', va='center')
            return fig, {}

        # Create figure with subplots (one per group)
        fig, axes = plt.subplots(num_groups, 1, figsize=(8, 3*num_groups), constrained_layout=True)

        # Handle case with only one group
        if num_groups == 1:
            axes = [axes]

        # Process each group and store CV values for table export
        cv_values_by_group = {}

        for i, (group_name, columns) in enumerate(group_selections.items()):
            ax = axes[i]

            # Skip if no columns in group
            if not columns:
                ax.text(0.5, 0.5, f"No columns in group: {group_name}", ha='center', va='center', transform=ax.transAxes)
                continue

            # Calculate CV for this group
            group_data = df[columns]
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_series = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_series.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Store for table export - include protein IDs
            protein_col = None
            if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
                protein_col = st.session_state.protein_col

            cv_df = pd.DataFrame({
                'Protein': df[protein_col] if protein_col and protein_col in df.columns else cv_series.index,
                'CV': cv_series
            })
            cv_values_by_group[group_name] = cv_df

            # Remove NaN values for plotting
            cv_values = cv_series.dropna()

            # Skip if all values are NaN
            if len(cv_values) == 0:
                ax.text(0.5, 0.5, f"No valid CV values for group: {group_name}", ha='center', va='center', transform=ax.transAxes)
                continue

            # Create histogram
            n, bins, patches = ax.hist(
                cv_values, 
                bins=50, 
                range=(0, min(3, cv_values.max() * 1.2) if not cv_values.empty else 3),
                alpha=0.7, 
                color=f'C{i}'
            )

            # Add cutoff line
            ylim = ax.get_ylim()
            ax.plot([cutoff, cutoff], [0, ylim[1]], 'r--', linewidth=2, label=f'CV Cutoff: {cutoff:.2f}')

            # Set title and labels
            ax.set_title(f"Group: {group_name}")
            ax.set_xlabel("Coefficient of Variation (CV)")
            ax.set_ylabel("Count")
            ax.set_xlim(0, min(3, cv_values.max()*1.2) if not cv_values.empty else 3)
            ax.legend()

            # Add stats text
            above_cutoff = (cv_values > cutoff).sum()
            percent_above = above_cutoff / len(cv_values) * 100 if len(cv_values) > 0 else 0
            stats_text = f"Total: {len(cv_values)}\nAbove cutoff: {above_cutoff} ({percent_above:.1f}%)"
            ax.text(0.95, 0.95, stats_text, ha='right', va='top', transform=ax.transAxes, 
                   bbox=dict(facecolor='white', alpha=0.7))

        # Add overall title
        fig.suptitle("Distribution of Coefficient of Variation (CV) by Group", fontsize=16)

        return fig, cv_values_by_group

    @staticmethod
    def get_matplotlib_svg(fig):
        """Get SVG string from matplotlib figure"""
        import io

        svg_io = io.StringIO()
        fig.savefig(svg_io, format='svg')
        svg_io.seek(0)
        svg_data = svg_io.getvalue()
        return svg_data

    @staticmethod
    def create_intensity_histograms(df: pd.DataFrame, group_selections: dict) -> go.Figure:
        """
        Create histograms of log2 expression values for each sample, grouped by sample group

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists

        Returns:
            Plotly figure with expression histograms
        """
        # Calculate number of groups and samples for subplot layout
        n_groups = len(group_selections)

        if n_groups == 0:
            # Return empty figure if no groups
            return go.Figure().update_layout(title="No groups defined")

        # Count total samples
        n_samples = sum(len(cols) for cols in group_selections.values())

        # Calculate grid dimensions - try to make it as square as possible
        n_cols = min(3, n_samples)  # Maximum 3 plots per row (was 4)
        n_rows = (n_samples + n_cols - 1) // n_cols  # Ceiling division

        # Create subplot titles for each sample
        subplot_titles = []
        for group_name, columns in group_selections.items():
            for col in columns:
                subplot_titles.append(f"{col} ({group_name})")

        # Create subplot grid with increased spacing
        fig = make_subplots(rows=n_rows, cols=n_cols, 
                           subplot_titles=subplot_titles,
                           vertical_spacing=0.2,    # Increased from 0.1
                           horizontal_spacing=0.15) # Increased from 0.05

        # Track position in the grid
        plot_idx = 0

        # Color palettes for each group
        color_palette = px.colors.qualitative.Plotly

        # Process each group
        for i, (group_name, columns) in enumerate(group_selections.items()):
            # Assign a color to this group
            group_color = color_palette[i % len(color_palette)]

            # For each sample in the group
            for col in columns:
                # Calculate row and column position
                row = (plot_idx // n_cols) + 1
                col_pos = (plot_idx % n_cols) + 1

                # Extract intensity values (ignore NaNs)
                intensity_values = df[col].dropna()

                if len(intensity_values) == 0:
                    # If no valid data, add empty plot with message
                    fig.add_annotation(
                        x=0.5, y=0.5,
                        text="No data",
                        showarrow=False,
                        row=row, col=col_pos
                    )
                else:
                    # Convert to log2 scale, handling zeros and negative values
                    log2_values = np.log2(intensity_values.clip(lower=1e-6))

                    # Calculate mean for the vertical line
                    mean_value = np.mean(log2_values)

                    # Create histogram trace
                    hist_trace = go.Histogram(
                        x=log2_values,
                        nbinsx=30,
                        marker_color=group_color,
                        name=col,
                        showlegend=False
                    )

                    # Add histogram to subplot
                    fig.add_trace(hist_trace, row=row, col=col_pos)

                    # Add mean line
                    fig.add_shape(
                        type="line",
                        x0=mean_value, x1=mean_value,
                        y0=0, y1=1,
                        yref="paper",
                        xref=f"x{plot_idx+1}",
                        line=dict(color="red", width=2, dash="dash"),
                        row=row, col=col_pos
                    )

                    # Add count annotations
                    protein_count = len(intensity_values)
                    fig.add_annotation(
                        x=0.95, y=0.95,
                        xref=f"x{plot_idx+1}",
                        yref=f"y{plot_idx+1}",
                        text=f"n={protein_count}",
                        showarrow=False,
                        font=dict(size=10),
                        align="right",
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="black",
                        borderwidth=1,
                        borderpad=3,
                        row=row, col=col_pos
                    )

                    # Add mean value annotation
                    fig.add_annotation(
                        x=mean_value, y=0.85,
                        xref=f"x{plot_idx+1}",
                        yref=f"y{plot_idx+1}",
                        text=f"Mean: {mean_value:.2f}",
                        showarrow=False,
                        font=dict(size=10, color="red"),
                        align="center",
                        bgcolor="rgba(255,255,255,0.7)",
                        bordercolor="red",
                        borderwidth=1,
                        borderpad=3,
                        row=row, col=col_pos
                    )

                    # Update axes
                    fig.update_xaxes(title_text="log2(Expression)", row=row, col=col_pos)
                    if col_pos == 1:  # Only for first column
                        fig.update_yaxes(title_text="Count", row=row, col=col_pos)

                # Increment position counter
                plot_idx += 1

        # Update overall layout
        fig.update_layout(
            title="Distribution of Protein Expression by Sample",
            height=300*n_rows + 50,  # Increased from 250 to give more vertical space
            width=300*n_cols + 50,   # Increased from 250 to give more horizontal space
            margin=dict(t=70, b=20, l=60, r=20),
            template="plotly_white"
        )

        return fig

    @staticmethod
    def generate_cv_table(df: pd.DataFrame, group_selections: dict) -> pd.DataFrame:
        """
        Generate a table of CV values for all proteins

        Args:
            df: Dataframe with proteomics data
            group_selections: Dictionary mapping group names to column lists

        Returns:
            DataFrame with protein IDs and CV values for each group
        """
        # Start with protein column
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col

        protein_col = protein_col if protein_col and protein_col in df.columns else df.index.name or 'Index'

        cv_table = pd.DataFrame({
            'Protein': df[protein_col] if protein_col in df.columns else df.index
        })

        # Calculate CV for each group
        for group_name, columns in group_selections.items():
            if not columns:
                continue

            # Calculate CV for this group
            group_data = df[columns]
            means = group_data.mean(axis=1).abs()
            stds = group_data.std(axis=1)

            # Calculate CV and handle potential division by zero
            cv_values = pd.Series(np.nan, index=means.index)
            mask = (means > 0) & (~means.isna()) & (~stds.isna())
            cv_values.loc[mask] = stds.loc[mask] / means.loc[mask]

            # Add to table
            cv_table[f'CV_{group_name}'] = cv_values

        return cv_table


    @staticmethod
    def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: str = None) -> go.Figure:
        """
        Create interactive scatter plot
        """
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=color_col,
            trendline="ols"
        )

        fig.update_layout(
            title=f"{y_col} vs {x_col}",
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_volcano_plot(df: pd.DataFrame, fc_col: str, 
                          pval_col: str, labels: list = None,
                          fc_threshold: float = 1.0,
                          p_threshold: float = 0.05,
                          correction_name: str = "None",
                          power: float = None,
                          alpha: float = 0.05,
                          labels_to_show: list = None) -> go.Figure:
        """
        Create volcano plot

        Args:
            df: DataFrame containing the data
            fc_col: Column name for fold change values
            pval_col: Column name for p-values
            labels: Optional list of labels for hover text
            fc_threshold: Threshold for log2 fold change significance (default: 1.0)
            p_threshold: Threshold for p-value significance (default: 0.05)
            correction_name: Name of the p-value correction method used (default: "None")
            labels_to_show: List of boolean values indicating which points to label on the plot

        Returns:
            Plotly figure with volcano plot
        """
        # Convert p-values to -log10(p) for y-axis
        neg_log_p = -np.log10(df[pval_col].astype(float))

        # Use provided significance thresholds
        neg_log_p_threshold = -np.log10(p_threshold)  # -log10 of p threshold

        # Create a new column for coloring points
        df = df.copy()
        df['significance'] = 'Not Significant'

        # Significant with fold change and p-value criteria
        sig_mask = (df[fc_col].abs() >= fc_threshold) & (df[pval_col] < p_threshold)
        df.loc[sig_mask, 'significance'] = 'Significant'

        # Significant with only p-value criteria
        p_mask = (~sig_mask) & (df[pval_col] < p_threshold)
        df.loc[p_mask, 'significance'] = 'p-value < 0.05'

        # Significant with only fold change criteria
        fc_mask = (~sig_mask) & (df[fc_col].abs() >= fc_threshold)
        df.loc[fc_mask, 'significance'] = '|Log2FC| ≥ 1'

        # Define color mapping
        color_map = {
            'Significant': 'red',
            'p-value < 0.05': 'orange',
            '|Log2FC| ≥ 1': 'blue',
            'Not Significant': 'gray'
        }

        # Create hover text
        if labels is not None:
            hover_text = [f"Protein: {label}<br>Log2FC: {fc:.3f}<br>p-value: {p:.4e}"
                        for label, fc, p in zip(labels, df[fc_col], df[pval_col])]
        else:
            hover_text = [f"Log2FC: {fc:.3f}<br>p-value: {p:.4e}"
                        for fc, p in zip(df[fc_col], df[pval_col])]

        # Create figure
        fig = px.scatter(
            df,
            x=fc_col,
            y=neg_log_p,
            color='significance',
            color_discrete_map=color_map,
            hover_name=labels if labels is not None else None,
            hover_data={fc_col: ':.3f', pval_col: ':.4e'},
            labels={fc_col: 'Log2 Fold Change', 'y': '-log10(p-value)'},
        )

        # Add vertical lines for fold change threshold
        fig.add_vline(x=fc_threshold, line_dash="dash", line_color="gray")
        fig.add_vline(x=-fc_threshold, line_dash="dash", line_color="gray")

        # Add horizontal line for p-value threshold
        fig.add_hline(y=neg_log_p_threshold, line_dash="dash", line_color="gray")

        # Update layout
        plot_title = "Volcano Plot"
        if correction_name != "None":
            plot_title += f" (Correction: {correction_name})"

        # Use the power parameter if provided, otherwise try session state
        if power is None and hasattr(st, 'session_state') and 'volcano_power' in st.session_state:
            power = st.session_state.get('volcano_power', None)
            
        if power is not None:
            plot_title += f" | Statistical Power: {power:.2f} (FDR = {alpha:.2f})"

        fig.update_layout(
            title=plot_title,
            xaxis_title="Log2 Fold Change",
            yaxis_title=f"-log10({pval_col.replace('_', ' ')})",
            width=800,
            height=600,
            legend_title="Significance"
        )

        # Update marker size
        fig.update_traces(marker=dict(size=10, opacity=0.7))

        # Add text labels for selected proteins if available
        if labels is not None and 'label_protein' in df.columns and df['label_protein'].any():
            # Get proteins to label
            label_indices = df['label_protein']
            label_x = df.loc[label_indices, fc_col]
            label_y = -np.log10(df.loc[label_indices, pval_col])
            label_texts = df.loc[label_indices, 'Protein'] if 'Protein' in df.columns else labels[label_indices]

            # Add text annotations
            for i, (x, y, text) in enumerate(zip(label_x, label_y, label_texts)):
                fig.add_annotation(
                    x=x,
                    y=y,
                    text=text,
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    font=dict(size=12, color="black"),
                    bgcolor="white",
                    bordercolor=None,  # Remove the border
                    borderwidth=0,     # Set border width to 0
                    borderpad=2,
                    opacity=0.8
                )

        return fig

    @staticmethod
    def create_box_plot(df: pd.DataFrame, value_col: str, 
                       group_col: str) -> go.Figure:
        """
        Create interactive box plot
        """
        fig = px.box(
            df,
            x=group_col,
            y=value_col,
            points="all"
        )

        fig.update_layout(
            title=f"Distribution of {value_col} by {group_col}",
            width=800,
            height=600
        )

        return fig

    @staticmethod
    def create_pca_plot(df: pd.DataFrame, group_selections: dict, show_ellipses: bool = False, confidence_level: float = 0.95) -> go.Figure:
        """
        Create PCA plot of samples within selected groups

        Args:
            df: DataFrame with proteomics data
            group_selections: Dictionary mapping group names to column lists
            show_ellipses: Whether to show confidence ellipses
            confidence_level: Confidence level for ellipses (0.90 to 1.00)

        Returns:
            Plotly figure with PCA plot
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # Extract sample data for PCA
        # For PCA we need to construct a matrix where:
        # - Rows are samples (columns in our original data)
        # - Columns are features (rows/proteins in our original data)

        # First, get all columns to include in PCA
        all_sample_cols = []
        for cols in group_selections.values():
            all_sample_cols.extend(cols)

        if len(all_sample_cols) < 3:
            fig = go.Figure()
            fig.update_layout(title="Not enough samples for PCA (need at least 3)")
            return fig

        # Extract and prepare the data
        # Create a samples x proteins matrix for PCA
        pca_data = df[all_sample_cols].T.copy()  # Transpose so that samples are rows

        # Handle missing values
        # Fill NaN with column mean (now each column is a protein)
        pca_data = pca_data.fillna(pca_data.mean())

        # Skip if no valid data after preprocessing
        if pca_data.empty or pca_data.isnull().all().all():
            fig = go.Figure()
            fig.update_layout(title="Not enough valid data for PCA after NaN handling")
            return fig

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(pca_data)

        # Perform PCA
        pca = PCA(n_components=2)
        pc_coords = pca.fit_transform(X_scaled)

        # Create a DataFrame with sample projections
        sample_projections = []

        # Map each sample to its group
        sample_to_group = {}
        for group_name, sample_cols in group_selections.items():
            for col in sample_cols:
                sample_to_group[col] = group_name

        # Create projection dataframe
        projection_df = pd.DataFrame({
            'PC1': pc_coords[:, 0],
            'PC2': pc_coords[:, 1],
            'sample': pca_data.index,
            'group': [sample_to_group.get(sample, 'Unknown') for sample in pca_data.index]
        })

        # Create the plot with explicit color map
        # Use a set of distinct colors from Plotly qualitative color scales
        from plotly.express.colors import qualitative

        # Get unique groups and assign colors
        unique_groups = projection_df['group'].unique()
        n_groups = len(unique_groups)
        color_palette = qualitative.Plotly if n_groups <= 10 else qualitative.Dark24

        # Create explicit color map
        color_map = {group: color_palette[i % len(color_palette)] 
                    for i, group in enumerate(unique_groups)}

        fig = px.scatter(
            projection_df,
            x='PC1', 
            y='PC2',
            color='group',
            color_discrete_map=color_map,  # Use explicit color map
            title=f'PCA Plot - Explained variance: PC1 {pca.explained_variance_ratio_[0]:.2%}, PC2 {pca.explained_variance_ratio_[1]:.2%}',
        )

        # Update marker size and hover info with explicit colors
        fig.update_traces(
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')),
            hoverinfo='text',
            hovertext=projection_df['sample']
        )

        # Draw ellipses if requested
        if show_ellipses:
            from scipy.stats import chi2

            # Draw confidence ellipses for each group
            for group in projection_df['group'].unique():
                group_df = projection_df[projection_df['group'] == group]

                if len(group_df) < 3:  # Need at least 3 points for covariance
                    continue

                # Calculate the covariance matrix
                x = group_df['PC1']
                y = group_df['PC2']
                cov = np.cov(x, y)

                # Get the eigenvalues and eigenvectors
                eigenvals, eigenvecs = np.linalg.eigh(cov)

                # Get the indices of the eigenvalues in descending order
                order = eigenvals.argsort()[::-1]
                eigenvals = eigenvals[order]
                eigenvecs = eigenvecs[:, order]

                # Get the largest eigenvalue and eigenvector
                theta = np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0])

                # Chi-square value for the specified confidence level
                chisquare_val = chi2.ppf(confidence_level, 2)

                # Calculate ellipse parameters
                width = 2 * np.sqrt(chisquare_val * eigenvals[0])
                height = 2 * np.sqrt(chisquare_val * eigenvals[1])

                # Generate ellipse points
                t = np.linspace(0, 2*np.pi, 100)
                ellipse_x = width/2 * np.cos(t)
                ellipse_y = height/2 * np.sin(t)

                # Rotate the ellipse
                x_rot = ellipse_x * np.cos(theta) - ellipse_y * np.sin(theta)
                y_rot = ellipse_x * np.sin(theta) + ellipse_y * np.cos(theta)

                # Shift to the mean position
                x_rot += np.mean(x)
                y_rot += np.mean(y)

                # Get the color for this group
                group_color = None
                for trace in fig.data:
                    if trace.name == group:
                        group_color = trace.marker.color
                        break

                # Add the ellipse as a scatter trace with fill
                fig.add_scatter(
                    x=x_rot, 
                    y=y_rot, 
                    mode='lines', 
                    line=dict(color=group_color, width=2),
                    fill='toself',
                    fillcolor=f'rgba({",".join([str(int(c)) for c in px.colors.hex_to_rgb(group_color)])},0.2)' if isinstance(group_color, str) and group_color.startswith('#') else f'rgba(0,0,0,0.1)',
                    name=f'{group} ({int(confidence_level * 100)}% confidence)',
                    showlegend=True
                )

        # Update layout
        fig.update_layout(
            width=800,
            height=600,
            xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.2%})",
            yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.2%})",
            legend_title="Group"
        )

        return fig
    
    @staticmethod
    def create_protein_bar_plot(df: pd.DataFrame, protein_names: list, group_selections: dict, selected_groups: dict = None) -> Tuple[dict, pd.DataFrame]:
        """
        Create bar plot showing protein expression across different sample groups with error bars
        
        Args:
            df: DataFrame with proteomics data
            protein_names: List of protein names to plot
            group_selections: Dictionary mapping group names to column lists
            selected_groups: Dictionary of groups to include (keys are group names, values are booleans)
            
        Returns:
            Tuple of (Dictionary of protein name to figure, DataFrame with statistics)
        """
        from scipy import stats
        import numpy as np
        import random
        import matplotlib.pyplot as plt
        import matplotlib
        import io
        from PIL import Image
        
        # Use non-interactive Agg backend for server-side plotting
        matplotlib.use('Agg')
        
        # Get protein column from session state if available
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col
            
        if not protein_col or protein_col not in df.columns:
            # Try to find a protein column
            for col in df.columns:
                if 'protein' in col.lower() or 'gene' in col.lower() or 'pg.genes' in col.lower():
                    protein_col = col
                    break
            
            if not protein_col:
                # Use index as last resort
                df = df.copy()
                df['Protein_ID'] = df.index
                protein_col = 'Protein_ID'
        
        # Filter for specified proteins
        filtered_df = df[df[protein_col].isin(protein_names)]
        
        if filtered_df.empty:
            # No matching proteins found
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No matching proteins found", ha='center', va='center', transform=ax.transAxes, fontsize=16)
            ax.set_xlabel("Group", fontsize=16)
            ax.set_ylabel("Expression", fontsize=16)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            # Convert matplotlib figure to image for Streamlit
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            
            # Also save SVG version
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_buf.seek(0)
            svg_data = svg_buf.getvalue()
            
            return {"No matching proteins": (img, svg_data)}, pd.DataFrame()
        
        # Create a DataFrame to store statistics
        stats_df_rows = []
        
        # Dictionary to store figures for each protein
        protein_figures = {}
        
        # Process each protein
        for protein_name in protein_names:
            if protein_name not in filtered_df[protein_col].values:
                continue
                
            protein_data = filtered_df[filtered_df[protein_col] == protein_name]
            
            # Create a new matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Calculate expression values for each group
            group_means = []
            group_sems = []
            group_names = []
            
            # Store all data for statistical comparison
            group_values = {}
            
            for group_name, columns in group_selections.items():
                # Skip if group not selected for display/comparison
                if selected_groups is not None and not selected_groups.get(group_name, True):
                    continue
                    
                if not columns:
                    continue
                    
                # Extract values for this protein and group
                values = protein_data[columns].values.flatten()
                values = values[~np.isnan(values)]  # Remove NaN values
                
                if len(values) == 0:
                    continue
                    
                # Store values for statistical tests
                group_values[group_name] = values
                    
                # Calculate statistics
                mean_val = np.mean(values)
                sem_val = stats.sem(values) if len(values) > 1 else 0
                
                group_means.append(mean_val)
                group_sems.append(sem_val)
                group_names.append(group_name)
            
            # Add bar chart with error bars
            x_pos = np.arange(len(group_names))
            ax.bar(x_pos, group_means, yerr=group_sems, align='center', alpha=0.7, capsize=10, color='#1f77b4', edgecolor='black')
            
            # Add scatter points for individual samples with controlled jitter
            for i, (group_name, columns) in enumerate(group_selections.items()):
                if not columns or group_name not in group_names:
                    continue
                
                # Get group index in the ordered arrays
                group_idx = group_names.index(group_name)
                
                # Extract values for each sample in this group
                sample_values = []
                sample_names = []
                
                for col in columns:
                    if col in protein_data.columns:
                        value = protein_data[col].values[0]
                        if not pd.isna(value):
                            sample_values.append(value)
                            sample_names.append(col)
                
                if sample_values:
                    # Generate controlled jitter - ensure points stay over the bar
                    # Use a narrow jitter range (0.2) centered on the bar position
                    jitter_width = 0.2
                    n_samples = len(sample_values)
                    
                    # Create fixed jitter positions that span the width evenly
                    if n_samples > 1:
                        # Equal spacing between points
                        jitter = np.linspace(-jitter_width/2, jitter_width/2, n_samples)
                        # Add small random noise to prevent perfect alignment
                        small_noise = np.random.normal(0, 0.02, n_samples)
                        jitter = jitter + small_noise
                    else:
                        # Single point, center it
                        jitter = [0]
                    
                    # Plot points with controlled jitter
                    ax.scatter(
                        x_pos[group_idx] + jitter, 
                        sample_values, 
                        color='black', 
                        alpha=0.7, 
                        s=50,  # Increased point size
                        zorder=3  # Ensure points are drawn on top
                    )
            
            # Perform statistical tests between all group pairs (for stats table only)
            group_pairs = [(i, j) for i in range(len(group_names)) for j in range(i+1, len(group_names))]
            
            # Add statistical data to the table but don't draw on plot
            for i, j in group_pairs:
                group1_name = group_names[i]
                group2_name = group_names[j]
                
                group1_values = group_values[group1_name]
                group2_values = group_values[group2_name]
                
                if len(group1_values) > 1 and len(group2_values) > 1:
                    # Perform Welch's t-test (unequal variances)
                    t_stat, p_val = stats.ttest_ind(
                        group1_values, 
                        group2_values, 
                        equal_var=False,
                        nan_policy='omit'
                    )
                    
                    # Calculate fold change
                    mean1 = np.mean(group1_values)
                    mean2 = np.mean(group2_values)
                    log2fc = np.log2(mean2 / mean1) if mean1 > 0 else float('inf')
                    
                    # Add to stats DataFrame
                    stats_df_rows.append({
                        'Protein': protein_name,
                        'Group 1': group1_name,
                        'Group 2': group2_name,
                        'Mean 1': mean1,
                        'Mean 2': mean2,
                        'Log2 Fold Change': log2fc,
                        't-statistic': t_stat,
                        'p-value': p_val,
                        'Significant': p_val < 0.05
                    })
                    
                    # Note: No significance bars or annotations are added to the plot
            
            # Set plot labels and styling with increased font sizes
            ax.set_title(f"Expression of {protein_name}", fontsize=16, pad=20)
            ax.set_ylabel("Expression Level", fontsize=16)
            ax.set_xlabel("Group", fontsize=16)
            
            # Set x-axis ticks to group names with increased font size
            ax.set_xticks(x_pos)
            ax.set_xticklabels(group_names, fontsize=14)
            
            # Increase y-axis tick font size
            ax.tick_params(axis='y', which='major', labelsize=14)
            
            # Add grid lines for easier reading
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Update y-axis limits without extending for significance bars
            # Get current limits
            y_min, y_max = ax.get_ylim()
            
            # Only add a small margin (10%) for better readability
            ax.set_ylim(y_min, y_max * 1.1)
            
            # Adjust layout
            plt.tight_layout()
            
            # Convert matplotlib figure to PIL Image for Streamlit
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            img = Image.open(buf)
            
            # Also save SVG version
            svg_buf = io.BytesIO()
            fig.savefig(svg_buf, format='svg', bbox_inches='tight')
            svg_buf.seek(0)
            svg_data = svg_buf.getvalue()
            
            plt.close(fig)  # Close figure to free memory
            
            # Store image and SVG data in dictionary
            protein_figures[protein_name] = (img, svg_data)
        
        # Create stats DataFrame
        stats_df = pd.DataFrame(stats_df_rows)
        
        return protein_figures, stats_df

    @staticmethod
    def create_protein_rank_plot(df: pd.DataFrame, intensity_columns: list, 
                                highlight_removed: bool = False, 
                                filtered_data: pd.DataFrame = None,
                                proteins_to_highlight: list = None) -> go.Figure:
        """
        Create a new implementation of the protein rank plot showing proteins ranked by their abundance
        
        Args:
            df: DataFrame with proteomics data
            intensity_columns: Columns to use for calculating protein abundance
            highlight_removed: Whether to highlight proteins that were removed in filtered dataset
            filtered_data: Filtered DataFrame (used when highlight_removed is True)
            proteins_to_highlight: List of protein names to highlight on the plot
            
        Returns:
            Plotly figure with protein rank plot
        """
        # Get protein column if available
        protein_col = None
        if hasattr(st, 'session_state') and 'protein_col' in st.session_state:
            protein_col = st.session_state.protein_col
            
        # Calculate mean intensity across selected columns for each protein
        mean_intensities = df[intensity_columns].mean(axis=1)
        
        # Create a new DataFrame for plotting
        plot_data = pd.DataFrame({
            'intensity': mean_intensities
        })
        
        # Add protein names if available
        if protein_col and protein_col in df.columns:
            plot_data['protein'] = df[protein_col].values
        
        # Sort by intensity in descending order and add rank
        plot_data = plot_data.sort_values('intensity', ascending=False).reset_index(drop=True)
        plot_data['rank'] = range(1, len(plot_data) + 1)
        
        # First create a basic figure
        fig = go.Figure()
        
        # Define nice colors
        base_color = 'rgba(31, 119, 180, 0.7)'  # Light blue
        highlight_color = 'rgba(214, 39, 40, 0.9)'  # Red
        filtered_color = 'rgba(148, 103, 189, 0.7)'  # Purple
        kept_color = 'rgba(44, 160, 44, 0.7)'  # Green
        
        # Handle different display scenarios
        if highlight_removed and filtered_data is not None:
            # Setup for filtering status
            kept_indices = filtered_data.index
            plot_data['status'] = 'Removed'
            plot_data.loc[plot_data.index.isin(kept_indices), 'status'] = 'Kept'
            
            # Create two separate traces for kept and removed proteins
            kept_data = plot_data[plot_data['status'] == 'Kept']
            removed_data = plot_data[plot_data['status'] == 'Removed']
            
            # Add scatter trace for kept proteins
            fig.add_trace(go.Scatter(
                x=kept_data['rank'],
                y=kept_data['intensity'],
                mode='markers',
                name='Kept after filtering',
                marker=dict(
                    color=kept_color,
                    size=8,
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=kept_data['protein'] if 'protein' in kept_data.columns else None
            ))
            
            # Add scatter trace for removed proteins
            fig.add_trace(go.Scatter(
                x=removed_data['rank'],
                y=removed_data['intensity'],
                mode='markers',
                name='Removed by filtering',
                marker=dict(
                    color=filtered_color,
                    size=8,
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=removed_data['protein'] if 'protein' in removed_data.columns else None
            ))
        
        elif proteins_to_highlight and 'protein' in plot_data.columns:
            # Split the data for highlighted and non-highlighted proteins
            plot_data['highlighted'] = plot_data['protein'].isin(proteins_to_highlight)
            
            highlighted_data = plot_data[plot_data['highlighted']]
            other_data = plot_data[~plot_data['highlighted']]
            
            # Add scatter trace for non-highlighted proteins
            fig.add_trace(go.Scatter(
                x=other_data['rank'],
                y=other_data['intensity'],
                mode='markers',
                name='Other proteins',
                marker=dict(
                    color=base_color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=other_data['protein'] if 'protein' in other_data.columns else None
            ))
            
            # Add scatter trace for highlighted proteins with larger markers
            fig.add_trace(go.Scatter(
                x=highlighted_data['rank'],
                y=highlighted_data['intensity'],
                mode='markers',
                name='Highlighted proteins',
                marker=dict(
                    color=highlight_color,
                    size=12,
                    line=dict(width=2, color='black')
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=highlighted_data['protein'] if 'protein' in highlighted_data.columns else None
            ))
            
            # Add text annotations for highlighted proteins
            for _, row in highlighted_data.iterrows():
                fig.add_annotation(
                    x=row['rank'],
                    y=row['intensity'] * 1.2,  # Position slightly above the point
                    text=row['protein'],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="black",
                    font=dict(size=12, color="black", family="Arial Black"),
                    bgcolor="white",
                    bordercolor="black",
                    borderwidth=1,
                    borderpad=4,
                    opacity=1.0
                )
        
        else:
            # Just add a single trace for all proteins
            fig.add_trace(go.Scatter(
                x=plot_data['rank'],
                y=plot_data['intensity'],
                mode='markers',
                name='Proteins',
                marker=dict(
                    color=base_color,
                    size=8,
                    opacity=0.7
                ),
                hovertemplate='<b>%{text}</b><br>Rank: %{x}<br>Intensity: %{y:.2e}<extra></extra>',
                text=plot_data['protein'] if 'protein' in plot_data.columns else None
            ))
        
        # Add a smooth line to show the trend
        sorted_intensities = plot_data['intensity'].values
        x_range = plot_data['rank'].values
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=sorted_intensities,
            mode='lines',
            line=dict(color='rgba(0,0,0,0.3)', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        # Update layout with log scale for y-axis
        fig.update_layout(
            title='Protein Rank Plot - Dynamic Range of Proteome',
            xaxis_title='Protein Rank (by abundance)',
            yaxis_title='Signal Intensity (log scale)',
            yaxis_type='log',
            height=600,
            width=900,
            template='plotly_white',
            hovermode='closest',
            # Explicitly define colorway for download compatibility
            colorway=px.colors.qualitative.Plotly
        )
        
        # Add annotations about dynamic range
        if len(sorted_intensities) > 0:
            max_intensity = sorted_intensities[0]
            min_intensity = sorted_intensities[-1] if len(sorted_intensities) > 1 else max_intensity
            
            dynamic_range = max_intensity / min_intensity if min_intensity > 0 else float('inf')
            
            fig.add_annotation(
                x=0.95,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"Dynamic Range: {dynamic_range:.1f}x",
                showarrow=False,
                font=dict(size=14),
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
            
            fig.add_annotation(
                x=0.95,
                y=0.89,
                xref="paper",
                yref="paper",
                text=f"Total Proteins: {len(sorted_intensities)}",
                showarrow=False,
                font=dict(size=14),
                align="right",
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="black",
                borderwidth=1,
                borderpad=4
            )
                
        return fig
