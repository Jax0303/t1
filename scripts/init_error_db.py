"""
Initialize Error Database Schema

Creates the Error DB CSV with predefined schema for systematic error tracking.
"""

import pandas as pd
from pathlib import Path


# Error DB Schema
ERROR_DB_SCHEMA = {
    # Identification
    'table_id': 'str',
    'split': 'str',  # 'train' | 'test' | 'comp'
    
    # Error Classification
    'primary_type': 'str',  # Main error category
    'secondary_types': 'str',  # Comma-separated additional types
    'stage': 'str',  # Processing stage where error occurred
    'severity': 'float',  # Error severity score [0, 1]
    
    # Structure Metrics
    'struct_f1': 'float',
    'struct_precision': 'float',
    'struct_recall': 'float',
    'horizontal_f1': 'float',
    'vertical_f1': 'float',
    'edge_FP': 'int',
    'edge_FN': 'int',
    
    # Content Metrics
    'cer': 'float',
    'wer': 'float',
    'cer_numeric': 'float',
    'cer_alpha': 'float',
    
    # Grid Stats
    'num_pred_cells': 'int',
    'num_gt_cells': 'int',
    'pred_rows': 'int',
    'pred_cols': 'int',
    'gt_rows': 'int',
    'gt_cols': 'int',
    
    # Attribution (from counterfactual experiments)
    'ocr_impact': 'float',  # F1 improvement with perfect OCR
    'tsr_impact': 'float',  # F1 improvement with perfect TSR
    'ocr_attribution': 'float',  # Normalized attribution to OCR
    'tsr_attribution': 'float',  # Normalized attribution to TSR
    
    # Labels
    'auto_label': 'str',  # Heuristic classification
    'manual_label': 'str',  # Human-verified label
    'verified': 'bool',  # Whether manually verified
    
    # Notes
    'notes': 'str',  # Freeform notes
    'timestamp': 'str'  # When processed
}

# Error Type Taxonomy
ERROR_TYPES = [
    'Row Merge/Split',
    'Col Merge/Split',
    'Row/Col Shift',
    'Span Error',
    'Table Boundary Error',
    'OCR Content Error',
    'Pure TSR Error',
    'Combined Error',
    'Success',
    'Unknown'
]

# Processing Stages
PROCESSING_STAGES = [
    'OCR Detection',
    'OCR Recognition',
    'OCR-to-TSR Transform',
    'TSR Grid Detection',
    'TSR Spanning/Merge',
    'Post-Processing',
    'Cell Assignment'
]


def initialize_error_db(output_path: str = 'results/error_db.csv'):
    """
    Initialize error database with schema
    
    Args:
        output_path: Path to save the CSV file
    """
    # Create output directory
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create empty DataFrame with schema
    df = pd.DataFrame(columns=list(ERROR_DB_SCHEMA.keys()))
    
    # Set data types
    for col, dtype in ERROR_DB_SCHEMA.items():
        if dtype == 'str':
            df[col] = df[col].astype('object')
        elif dtype == 'float':
            df[col] = df[col].astype('float64')
        elif dtype == 'int':
            df[col] = df[col].astype('Int64')  # Nullable integer
        elif dtype == 'bool':
            df[col] = df[col].astype('bool')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Error DB initialized at: {output_path}")
    print(f"  Columns: {len(ERROR_DB_SCHEMA)}")
    print(f"  Error Types: {len(ERROR_TYPES)}")
    print(f"  Processing Stages: {len(PROCESSING_STAGES)}")
    
    return df


def save_taxonomy_definitions(output_path: str = 'results/taxonomy.json'):
    """Save error type and stage definitions"""
    import json
    
    taxonomy = {
        'error_types': ERROR_TYPES,
        'processing_stages': PROCESSING_STAGES,
        'error_type_descriptions': {
            'Row Merge/Split': 'Multiple rows incorrectly merged or single row split',
            'Col Merge/Split': 'Multiple columns incorrectly merged or single column split',
            'Row/Col Shift': 'Content shifted to adjacent row/column',
            'Span Error': 'Merged/spanning cell incorrectly handled',
            'Table Boundary Error': 'Table boundary detection failure',
            'OCR Content Error': 'Text recognition error (structure correct)',
            'Pure TSR Error': 'Structure completely wrong (OCR correct)',
            'Combined Error': 'Both OCR and TSR failed',
            'Success': 'No significant errors',
            'Unknown': 'Cannot classify automatically'
        },
        'stage_descriptions': {
            'OCR Detection': 'Bounding box detection for text regions',
            'OCR Recognition': 'Character recognition within detected boxes',
            'OCR-to-TSR Transform': 'Conversion of OCR output to TSR input format',
            'TSR Grid Detection': 'Table row/column line detection',
            'TSR Spanning/Merge': 'Merged cell and spanning detection',
            'Post-Processing': 'Rule-based cleanup and correction',
            'Cell Assignment': 'Assigning text content to detected cells'
        }
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    print(f"✓ Taxonomy definitions saved to: {output_path}")


if __name__ == '__main__':
    # Initialize
    df = initialize_error_db()
    save_taxonomy_definitions()
    
    print("\n" + "="*50)
    print("Error Database Schema:")
    print("="*50)
    for col, dtype in ERROR_DB_SCHEMA.items():
        print(f"  {col:<30} {dtype}")
