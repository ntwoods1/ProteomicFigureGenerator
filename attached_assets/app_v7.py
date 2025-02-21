import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
from upsetplot import UpSet, from_contents
import tempfile
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse
from seaborn import clustermap
import io
from scipy.stats import ttest_ind
from scipy.cluster.hierarchy import linkage, dendrogram



# Title
st.title("Proteomic Data Analysis")

# File Upload
st.sidebar.header("Upload Datasets")
uploaded_files = st.sidebar.file_uploader(
	"Upload one or more datasets (Excel format)",
	accept_multiple_files=True,
	type=["xlsx"]
)
# def extract_gene_name(description):
# 	match = re.search(r"GN=([^\s]+)", description)
# 	return match.group(1) if match else None
	
# Function to extract gene names from the Description column
def extract_gene_name(description):
	if pd.isna(description):
		return None
	if "GN=" in description:
		try:
			return description.split("GN=")[1].split()[0]
		except IndexError:
			return None
	return None

# Function to compute scores for genes based on intra- and inter-condition variance
def compute_gene_scores(condition_data):
	scores = []
	# Use the gene names as unique identifiers for shared genes
	shared_genes = set.intersection(*[set(df.index) for df in condition_data.values()])
	if not shared_genes:
		raise ValueError("No shared genes found across all conditions. Please check your selections.")

	for gene in shared_genes:
		# Extract values for each condition
		group_values = [condition_data[group].loc[gene].values for group in condition_data]

		# Compute intra-condition variance
		intra_variance = np.mean([np.var(values, ddof=1) for values in group_values])

		# Compute inter-condition variance
		group_means = [np.mean(values) for values in group_values]
		inter_variance = np.var(group_means, ddof=1)

		# Calculate score
		score = inter_variance - intra_variance
		scores.append((gene, score))

	return pd.DataFrame(scores, columns=["Gene", "Score"]).sort_values(by="Score", ascending=False)

# Function to get top N ranked genes based on the score
def get_top_n_ranked_genes(condition_data, n):
	score_df = compute_gene_scores(condition_data)
	top_genes = score_df.head(n)["Gene"]
	return pd.concat([condition_data[group].loc[top_genes] for group in condition_data], axis=1)


# Function to extract significant proteins from a dataset
def get_significant_proteins(data, log2fc_col, pval_col, pval_cutoff, log2fc_cutoff, direction="up"):
	data["-log10(p-value)"] = -np.log10(data[pval_col])
	if direction == "up":
		return set(data.loc[
			(data[log2fc_col] >= log2fc_cutoff) & (data["-log10(p-value)"] >= pval_cutoff),
			"Gene Name"
		])
	elif direction == "down":
		return set(data.loc[
			(data[log2fc_col] <= -log2fc_cutoff) & (data["-log10(p-value)"] >= pval_cutoff),
			"Gene Name"
		])
	
# Placeholder for datasets
datasets = {}

if uploaded_files:
	# Load and store datasets
	for uploaded_file in uploaded_files:
		try:
			data = pd.read_excel(uploaded_file)
			if "Description" in data.columns:
				# Extract gene names
				data["Gene Name"] = data["Description"].apply(extract_gene_name)
			datasets[uploaded_file.name] = data
		except Exception as e:
			st.error(f"Error processing file {uploaded_file.name}: {e}")

	# Tabs for different functionalities
	tab1, tab2, tab3, tab4 = st.tabs(["Data Overview", "Volcano Plot", "PCA", "Heat Map"])

	# Data Overview Tab
	with tab1:
		st.header("Data Overview")
		dataset_name = st.selectbox(
			"Select a dataset to view",
			options=list(datasets.keys())
		)

		if dataset_name:
			selected_data = datasets[dataset_name]
			st.write("**Preview of the Dataset**")
			st.dataframe(selected_data.head(10))

			st.write("**Basic Statistics**")
			st.write(selected_data.describe())

	# Volcano Plot Tab
	with tab2:
		try:
			st.header("Volcano Plot")
			dataset_name = st.selectbox(
				"Select a dataset for the Volcano Plot",
				options=list(datasets.keys())
			)
	
			if dataset_name:
				selected_data = datasets[dataset_name]
	
				# Dropdowns for column selection
				columns = selected_data.columns
				log2fc_col = st.selectbox("Select Log2 Fold Change Column", options=["Select a column"] + list(columns))
				pvalue_col = st.selectbox("Select P-value Column", options=["Select a column"] + list(columns))
				
				# Validate the selected columns
				if log2fc_col and pvalue_col:
					# Ensure the columns are numeric and handle non-numeric values
					try:
						selected_data[log2fc_col] = pd.to_numeric(selected_data[log2fc_col], errors='coerce')
						selected_data[pvalue_col] = pd.to_numeric(selected_data[pvalue_col], errors='coerce')
				
						# Drop rows with invalid or missing data in the selected columns
						selected_data = selected_data.dropna(subset=[log2fc_col, pvalue_col])
				
						if selected_data.empty:
							st.error("The selected columns contain no valid numeric data. Please select appropriate columns.")
						else:
							# Add -log10(p-value) for visualization
							selected_data["-log10(p-value)"] = -np.log10(selected_data[pvalue_col])
							
					except Exception as e:
						st.error(f"An error occurred while processing the selected columns: {e}")
						
					# Adjust cutoffs
					st.write("**Adjust Cutoffs**")
					pval_cutoff = st.slider(
						"P-value Cutoff (as -log10(p-value))",
						min_value=0.0,
						max_value=10.0,
						value=1.3,  # Equivalent to p=0.05
						step=0.1
					)
					log2fc_cutoff = st.slider(
						"Log2 Fold Change Cutoff",
						min_value=0.0,
						max_value=5.0,
						value=1.0,
						step=0.1
					)
	
					# Assign colors based on significance
					selected_data["color"] = "gray"  # Default color
					selected_data.loc[
						(selected_data[log2fc_col] >= log2fc_cutoff) &
						(selected_data["-log10(p-value)"] >= pval_cutoff),
						"color"
					] = "red"  # Significant on the positive side
					selected_data.loc[
						(selected_data[log2fc_col] <= -log2fc_cutoff) &
						(selected_data["-log10(p-value)"] >= pval_cutoff),
						"color"
					] = "blue"  # Significant on the negative side
					
					# Generate Volcano Plot
					st.write("**Volcano Plot**")
					plt.figure(figsize=(8, 6))
					plt.scatter(
						selected_data[log2fc_col],
						selected_data["-log10(p-value)"],
						c=selected_data["color"],
						alpha=0.6,
						edgecolor="w"
					)
					plt.axhline(pval_cutoff, color="red", linestyle="--")
					plt.axvline(log2fc_cutoff, color="blue", linestyle="--")
					plt.axvline(-log2fc_cutoff, color="blue", linestyle="--")
					plt.title("Volcano Plot")
					plt.xlabel("Log2 Fold Change")
					plt.ylabel("-Log10(p-value)")
					st.pyplot(plt)
					
					# Save the plot as an SVG file
					with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
						plt.savefig(temp_file.name, format="svg")
						svg_file_path = temp_file.name
				
					# Add a download button for the SVG file
					st.download_button(
						label=f"Download Volcano Plot as SVG",
						data=open(svg_file_path, "rb").read(),
						file_name=f"VolcanoPlot.svg",
						mime="image/svg+xml"
					)
					
					# Generate Volcano Plot with Plotly
					st.write("**Interactive Volcano Plot**")
					if log2fc_col and pvalue_col:
						# Ensure Gene Name column exists
						if "Gene Name" not in selected_data.columns:
							selected_data["Gene Name"] = selected_data["Description"].apply(extract_gene_name)
						
						# Add -log10(p-value)
						selected_data["-log10(p-value)"] = -np.log10(selected_data[pvalue_col])
						
						# Assign colors based on significance
						selected_data["color"] = "gray"  # Default color
						selected_data.loc[
							(selected_data[log2fc_col] >= log2fc_cutoff) &
							(selected_data["-log10(p-value)"] >= pval_cutoff),
							"color"
						] = "red"  # Significant on the positive side
						selected_data.loc[
							(selected_data[log2fc_col] <= -log2fc_cutoff) &
							(selected_data["-log10(p-value)"] >= pval_cutoff),
							"color"
						] = "blue"  # Significant on the negative side
						
						# Plotly scatter plot
						fig = px.scatter(
							selected_data,
							x=log2fc_col,
							y="-log10(p-value)",
							color="color",
							hover_name="Gene Name",  # Display Gene Name on hover
							hover_data={log2fc_col: True, "-log10(p-value)": True, "Description": True},
							title="Volcano Plot",
							labels={log2fc_col: "Log2 Fold Change", "-log10(p-value)": "-Log10(p-value)"},
							color_discrete_map={"red": "red", "blue": "blue", "gray": "gray"}
						)
						fig.update_traces(marker=dict(size=8, opacity=0.7), showlegend=False)
						fig.update_layout(title_x=0.5, title_font_size=20, xaxis_title="Log2 Fold Change", yaxis_title="-Log10(p-value)")
						st.plotly_chart(fig, use_container_width=True)
	
					# Bar Chart: Significant Proteins
					st.write("**Significant Proteins Summary**")
					upregulated = selected_data[
						(selected_data[log2fc_col] >= log2fc_cutoff) &
						(selected_data["-log10(p-value)"] >= pval_cutoff)
					]
					downregulated = selected_data[
						(selected_data[log2fc_col] <= -log2fc_cutoff) &
						(selected_data["-log10(p-value)"] >= pval_cutoff)
					]
	
					# Create a bar chart
					plt.figure(figsize=(6, 4))
					categories = ["Downregulated", "Upregulated"]
					counts = [len(downregulated), len(upregulated)]
					colors = ["blue", "red"]
					bars = plt.bar(categories, counts, color=colors)
					for bar, count in zip(bars, counts):
						plt.text(
							bar.get_x() + bar.get_width() / 2,  # X-coordinate
							bar.get_height() + 0.5,  # Y-coordinate (slightly above the bar)
							str(count),  # Text (the count)
							ha="center",  # Horizontal alignment
							va="bottom",  # Vertical alignment
							fontsize=12  # Font size
						)
					plt.title("Significant Proteins")
					plt.ylabel("Number of Proteins")
					st.pyplot(plt)
					
					# Save the plot as an SVG file
					with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
						plt.savefig(temp_file.name, format="svg")
						svg_file_path = temp_file.name
				
					# Add a download button for the SVG file
					st.download_button(
						label=f"Download Bar Chart as SVG",
						data=open(svg_file_path, "rb").read(),
						file_name=f"BarChart.svg",
						mime="image/svg+xml"
					)
	
					# Highlight significant points
					significant = pd.concat([upregulated, downregulated])
					st.write("**Significant Proteins**")
					st.dataframe(significant)
	
					# Option to download significant proteins
					st.write("**Download Significant Proteins**")
					csv = significant.to_csv(index=False)
					st.download_button(
						label="Download CSV",
						data=csv,
						file_name="significant_proteins.csv",
						mime="text/csv"
					)
			else:
				st.warning("Please select valid columns for Log2 Fold Change and P-value.")
			
			dataset_names = list(datasets.keys())
		
			# Check if at least 2 datasets are uploaded
			if len(dataset_names) >= 2:
				st.write("**UpSet Plots**: Overlap of Significant Proteins")
		
				# Extract upregulated and downregulated proteins for each dataset
				significant_up = {}
				significant_down = {}
		
				for name in dataset_names:
					data = datasets[name]
		
					# Ensure Gene Name column exists
					if "Gene Name" not in data.columns:
						data["Gene Name"] = data["Description"].apply(extract_gene_name)
		
					# Get upregulated and downregulated proteins
					significant_up[name] = get_significant_proteins(
						data, log2fc_col, pvalue_col, pval_cutoff, log2fc_cutoff, direction="up"
					)
					significant_down[name] = get_significant_proteins(
						data, log2fc_col, pvalue_col, pval_cutoff, log2fc_cutoff, direction="down"
					)
		
				# Create UpSet plots
				for direction, significant_sets in zip(["Upregulated", "Downregulated"], [significant_up, significant_down]):
					st.subheader(f"{direction} Proteins")
		
					# Convert dictionary of sets to UpSet input
					upset_data = from_contents(significant_sets)
		
					# Plot UpSet plot
					plt.figure(figsize=(10, 6))
					UpSet(upset_data).plot()
					plt.title(f"{direction} Proteins Overlap")
					st.pyplot(plt)
					
						# Save the plot as an SVG file
					with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
						plt.savefig(temp_file.name, format="svg")
						svg_file_path = temp_file.name
				
					# Add a download button for the SVG file
					st.download_button(
						label=f"Download {direction} Proteins Overlap as SVG",
						data=open(svg_file_path, "rb").read(),
						file_name=f"{direction.lower()}_proteins_overlap.svg",
						mime="image/svg+xml"
					)
			else:
				st.info("Upload at least 2 datasets to generate UpSet plots.")
				
		except Exception as e:
			st.error(f"An error occurred in Volcano Plot: {e}")
				
	# Add PCA Plots Tab
	with tab3:
		st.header("PCA")

		# Initialize session state for PCA configuration
		if "pca_configured" not in st.session_state:
			st.session_state["pca_configured"] = False
		if "dataset_config" not in st.session_state:
			st.session_state["dataset_config"] = {}
		if "selected_datasets" not in st.session_state:
			st.session_state["selected_datasets"] = []
		
		if not st.session_state["pca_configured"]:
			if st.button("Configure PCA"):
				st.session_state["pca_configured"] = True
		
		if st.session_state["pca_configured"]:
			st.header("PCA Configuration")
		
			# Dataset selection
			st.subheader("Select Datasets for PCA")
			selected_datasets = [name for name in datasets.keys() if st.checkbox(f"Include {name}", key=f"include_{name}")]
			st.session_state["selected_datasets"] = selected_datasets
		
			if len(selected_datasets) == 0:
				st.warning("Please select at least one dataset for PCA.")
			else:
				for dataset_name in selected_datasets:
					st.write(f"Configuring Dataset: {dataset_name}")
					
					num_conditions = st.selectbox(
						f"How many conditions in {dataset_name}?",
						options=range(1, 11),
						key=f"{dataset_name}_num_conditions"
					)
					st.session_state["dataset_config"].setdefault(dataset_name, {"conditions": {}})["num_conditions"] = num_conditions
		
					for condition_idx in range(num_conditions):
						condition_name = st.text_input(
							f"Name of condition {condition_idx + 1} in {dataset_name}",
							value=f"Condition {condition_idx + 1}",
							key=f"{dataset_name}_condition_{condition_idx}"
						)
		
						num_replicates = st.selectbox(
							f"How many replicates for {condition_name} in {dataset_name}?",
							options=range(1, 11),
							key=f"{dataset_name}_num_replicates_{condition_idx}"
						)
		
						replicate_columns = []
						for replicate_idx in range(num_replicates):
							col = st.selectbox(
								f"Select column for {condition_name}, replicate {replicate_idx + 1} in {dataset_name}",
								options=datasets[dataset_name].columns,
								key=f"{dataset_name}_{condition_idx}_replicate_{replicate_idx}"
							)
							replicate_columns.append(col)
		
						st.session_state["dataset_config"][dataset_name]["conditions"][condition_name] = replicate_columns
		
				# Identifier matching
				if len(selected_datasets) > 1:
					st.subheader("Identifier Matching")
					identifier_column = st.selectbox(
						"Select the identifier column to match between datasets",
						options=datasets[selected_datasets[0]].columns,
						key="identifier_column"
					)
					# identifier_column is already managed by Streamlit, no need to set it manually.
		
				# PCA button
				if st.button("Perform PCA"):
					try:
						pca_data = []
						labels = []
						dataset_labels = []
		
						for dataset_name, config in st.session_state["dataset_config"].items():
							selected_data = datasets[dataset_name]
		
							if len(selected_datasets) > 1:
								selected_identifiers = set(datasets[selected_datasets[0]][st.session_state["identifier_column"]])
								selected_data = selected_data[selected_data[st.session_state["identifier_column"]].isin(selected_identifiers)]
		
							for condition, replicate_columns in config["conditions"].items():
								condition_data = []
		
								for col in replicate_columns:
									if np.issubdtype(selected_data[col].dtype, np.number) and not selected_data[col].isnull().any():
										condition_data.append(selected_data[col].values)
										labels.append(f"{condition} ({dataset_name})")
										dataset_labels.append(dataset_name)
									else:
										st.warning(f"Skipping column '{col}' in {dataset_name} due to invalid data.")
		
								if condition_data:
									condition_data = np.column_stack(condition_data)
									pca_data.append(condition_data)
		
						if len(pca_data) > 1:
							min_rows = min([data.shape[0] for data in pca_data])
							pca_data = [data[:min_rows, :] for data in pca_data]
		
						if pca_data:
							pca_data = np.hstack(pca_data).T
		
							st.write(f"PCA Data Shape (Samples x Features): {pca_data.shape}")
							st.write(f"Number of Labels: {len(labels)}")
		
							if pca_data.shape[0] != len(labels):
								st.error(f"Mismatch: PCA data has {pca_data.shape[0]} rows but {len(labels)} labels provided.")
							else:
								pca = PCA(n_components=2)
								explained_variance = pca.fit(pca_data).explained_variance_ratio_ * 100
								pca_result = pca.fit_transform(pca_data)
		
								pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
								pca_df["Condition"] = labels
								pca_df["Dataset"] = dataset_labels
		
								plt.figure(figsize=(8, 6))
								for condition in set(labels):
									condition_data = pca_df[pca_df["Condition"] == condition]
									plt.scatter(condition_data["PC1"], condition_data["PC2"], label=condition, s=200)
		
								plt.title(f"PCA Plot", fontsize=16)
								plt.xlabel(f"PC1: {explained_variance[0]:.2f}% variance", fontsize=16)
								plt.ylabel(f"PC2: {explained_variance[1]:.2f}% variance", fontsize=16)
								plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=3)
								st.pyplot(plt)
								# Save the plot as an SVG file
								with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
									plt.savefig(temp_file.name, format="svg")
									svg_file_path = temp_file.name
							
								# Add a download button for the SVG file
								st.download_button(
									label=f"Download PCA plot as SVG",
									data=open(svg_file_path, "rb").read(),
									file_name=f"PCA_plot.svg",
									mime="image/svg+xml"
								)
								
								# Second PCA Plot with Ellipses
								plt.figure(figsize=(8, 6))
								for condition in set(labels):
									condition_data = pca_df[pca_df["Condition"] == condition]
									plt.scatter(condition_data["PC1"], condition_data["PC2"], label=condition, s=200)
									
									# Calculate ellipse parameters
									mean_x = condition_data["PC1"].mean()
									mean_y = condition_data["PC2"].mean()
									cov = np.cov(condition_data["PC1"], condition_data["PC2"])
									eigvals, eigvecs = np.linalg.eigh(cov)
									angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
									width, height = 2 * np.sqrt(eigvals) * np.sqrt(5.991)
					
									# Add ellipse
									ellipse = Ellipse(xy=(mean_x, mean_y), width=width, height=height, angle=angle, alpha=0.2)
									plt.gca().add_patch(ellipse)
					
								plt.title(f"PCA Plot with Ellipses for {dataset_name}", fontsize=16)
								plt.xlabel(f"PC1: {explained_variance[0]:.2f}% variance", fontsize=16)
								plt.ylabel(f"PC2: {explained_variance[1]:.2f}% variance", fontsize=16)
								plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.55), ncol=3)
								st.pyplot(plt)
								# Save the plot as an SVG file
								with tempfile.NamedTemporaryFile(delete=False, suffix=".svg") as temp_file:
									plt.savefig(temp_file.name, format="svg")
									svg_file_path = temp_file.name
							
								# Add a download button for the SVG file
								st.download_button(
									label=f"Download PCA plot with Ellipses as SVG",
									data=open(svg_file_path, "rb").read(),
									file_name=f"PCA_plot_ellipses.svg",
									mime="image/svg+xml"
								)
		
						else:
							st.error("No valid data for PCA. Ensure numeric data and no NaN values in selected columns.")
		
					except Exception as e:
						st.error(f"An error occurred during PCA: {e}")

	# Add Heat Map Tab
	with tab4:
		st.header("Heat Map")
		
		if uploaded_files:
			st.write("### Heat Map Configuration")
			included_files = {name: datasets[name] for name in datasets.keys()}

			# Load datasets into a dictionary
			datasets = {}
			for file in uploaded_files:
				data = pd.read_excel(file)
				if "Description" in data.columns:
					data["Gene Name"] = data["Description"].apply(extract_gene_name)
					data = data.set_index("Gene Name")
				elif "Accession" in data.columns:
					data = data.set_index("Accession")
				datasets[file.name] = data
		
			st.header("Select Datasets to Include")
			included_datasets = {name: st.checkbox(name, value=False) for name in datasets.keys()}
			included_files = {name: datasets[name] for name, include in included_datasets.items() if include}
		
			if included_files:
				selected_dataset_name = st.selectbox("Choose a dataset", list(included_files.keys()))
				selected_data = included_files[selected_dataset_name]
	
				st.subheader("Condition Configuration")
				n_conditions = st.number_input("How many conditions?", min_value=2, max_value=10, value=2)
				condition_groups = {}
	
				for i in range(1, n_conditions + 1):
					condition_groups[f"Condition {i}"] = st.multiselect(
						f"Select replicates for Condition {i}", 
						selected_data.select_dtypes(include=[np.number]).columns.tolist()
					)
		
				# Validate that all conditions have been configured
				if not all(condition_groups.values()):
					st.sidebar.warning("Please configure all conditions.")
				else:
					try:
						# Subset data for numeric columns only and align to condition groups
						condition_data = {
							key: selected_data[group] for key, group in condition_groups.items()
						}
		
						tab1, tab2 = st.tabs(["Overview", "Heatmap of Ranked Genes"])
		
						with tab1:
							st.write("Explore your data in this section.")
							st.write("### Preview of the Selected Dataset")
							st.dataframe(selected_data.head())
		
						with tab2:
							st.write("### Heatmap of Ranked Genes")
		
							# Add slider for top N genes
							n = st.slider("Select number of top ranked genes", min_value=1, max_value=100, value=10)
		
							# Get top N ranked genes
							top_n_genes = get_top_n_ranked_genes(condition_data, n)
		
							# Check if data for clustering is non-empty
							if top_n_genes.empty:
								st.warning("No data available for clustering. Please adjust your selections.")
							else:
								# Perform clustering on top N genes and plot heatmap with dendrogram
								st.write("### Heatmap with Clustering Dendrogram")
								clustered_heatmap = clustermap(
									top_n_genes,
									method='ward',
									cmap="viridis",
									yticklabels=True,
									xticklabels=True,
									figsize=(10, (10+(n/10)-1))
								)
								st.pyplot(clustered_heatmap)
		
								# Reorder table rows to match the clustered heatmap
								ordered_genes = clustered_heatmap.dendrogram_row.reordered_ind
								ordered_table = top_n_genes.iloc[ordered_genes]
		
								# Add a button to download the clustered heatmap as SVG
								st.write("### Download Options")
								svg_buffer = io.BytesIO()
								clustered_heatmap.savefig(svg_buffer, format="svg")
								svg_buffer.seek(0)
								st.download_button(
									label="Download Heatmap as SVG",
									data=svg_buffer,
									file_name="clustered_heatmap.svg",
									mime="image/svg+xml"
								)
		
								# Add a button to download the table as CSV
								csv_buffer = ordered_table.to_csv(index=True).encode('utf-8')
								st.download_button(
									label="Download Table as CSV",
									data=csv_buffer,
									file_name="clustered_table.csv",
									mime="text/csv"
								)
		
								# Show the complete table below the heatmap
								st.write("### Complete Table Used in the Heatmap (Ordered by Clustering)")
								st.dataframe(ordered_table)
		
					except ValueError as ve:
						st.warning(f"Warning: {ve}")
					except Exception as e:
						st.error(f"An error occurred during computation: {e}")
			else:
				st.write("Please select at least one dataset to include.")
		else:
			st.write("Please upload at least one dataset to proceed.")


else:
	st.info("Please upload one or more datasets to get started.")
