# Example Data

This directory contains example proteomics datasets that demonstrate the expected input format for the application.

## File Format Requirements

The application expects Excel files (.xlsx) with the following columns:

### Required Columns
- `Description`: Protein descriptions (used for gene name extraction)
- `[X]_CellLine_Treatment_ReplicateNumber.PG.Quantity`: Quantitative measurements for each sample
- `[X]_CellLine_Treatment_ReplicateNumber.PG.NrOfStrippedSequencesIdentified`: Number of peptides identified
  or
- `[X]_CellLine_Treatment_ReplicateNumber.PG.NrOfPrecursorsMeasured`: Number of precursors measured

### Column Naming Convention
Sample columns should follow this format:
```
[Number]_CellLine_Treatment_ReplicateNumber.PG.Quantity
```

Example: `[1]_HeLa_Control_1.PG.Quantity`

## Example Files

1. `example_dataset.xlsx` - A minimal example dataset containing:
   - 2 cell lines
   - 2 conditions
   - 3 replicates each
   - ~100 proteins

Note: Due to size limitations, example datasets contain a reduced number of proteins. Real datasets typically contain thousands of proteins.

## Using Example Data

1. Download the example dataset
2. Upload it to the application
3. Follow the tutorial in the main README to analyze the data

This will help you understand how the application processes and visualizes proteomics data.
