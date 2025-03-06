# Example Data Directory

Place your Excel (.xlsx) files in this directory. Files should follow this format:

## Required Columns
- `Description`: Protein descriptions
- `[X]_CellLine_Treatment_ReplicateNumber.PG.Quantity`: Quantitative measurements
- `[X]_CellLine_Treatment_ReplicateNumber.PG.NrOfStrippedSequencesIdentified`: Peptide counts
or
- `[X]_CellLine_Treatment_ReplicateNumber.PG.NrOfPrecursorsMeasured`: Number of precursors measured


## Example Format
```
Description                         [1]_HeLa_Control_1.PG.Quantity  [1]_HeLa_Control_1.PG.NrOfStrippedSequencesIdentified
Protein ABC >GN=ABC                1234.56                         5
Protein XYZ >GN=XYZ                789.01                          3
```

A sample dataset will be provided in future updates.

## Column Naming Convention
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