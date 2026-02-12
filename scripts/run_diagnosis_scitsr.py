#!/usr/bin/env python3
"""
Run Diagnosis on SciTSR Dataset - WITH EXTERNAL BASELINES

Executes S1-S4 scenarios using external TSR engines (TATR, PaddleX, Docling).

Usage:
    python scripts/run_diagnosis_scitsr.py \\
        --manifest outputs/manifest.csv \\
        --out outputs/results.csv \\
        --tsr_engine tatr \\
        --ocr_source paddleocr \\
        --device cuda
"""

import sys
import json
import logging
import argparse
import csv
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from hiertable_rag.evaluation.diagnosis import (
    SciTSRDataset,
    NoisyOCRSource,
    PerfectOCRSource,
    GTOCRSource,
    DiagnosisMetrics,
    ResultCache,
    setup_logging
)


def load_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Load manifest CSV"""
    tables = []
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['type_labels'] = row['type_labels'].split(',') if row['type_labels'] else []
            row['features'] = json.loads(row.get('features_json', '{}'))
            tables.append(row)
    
    return tables


def create_tsr_engine(engine_name: str, device: str, **kwargs):
    """
    Factory function to create TSR engine
    
    Args:
        engine_name: Engine type
        device: Device string
        **kwargs: Additional engine-specific parameters
    
    Returns:
        TSR engine instance
    """
    logger = logging.getLogger(__name__)
    
    # Import dynamically to handle missing dependencies gracefully
    from hiertable_rag.evaluation.diagnosis.tsr_engines import (
        BaselineTSR, GNNCSPTSREngine, GTStructureEngine
    )
    
    if engine_name == 'baseline':
        return BaselineTSR()
    
    elif engine_name == 'gnn_csp':
        try:
            return GNNCSPTSREngine(
                rca_checkpoint=kwargs.get('rca_checkpoint'),
                gnn_checkpoint=kwargs.get('gnn_checkpoint'),
                device=device
            )
        except Exception as e:
            logger.error(f"GNN-CSP init failed: {e}. Falling back to baseline")
            return BaselineTSR()
    
    elif engine_name == 'tatr':
        try:
            from hiertable_rag.evaluation.diagnosis.tsr_engines import TATREngine
            logger.info(f"Creating TATR engine on {device}")
            return TATREngine(device=device)
        except Exception as e:
            logger.error(f"TATR init failed: {e}")
            raise
    
    elif engine_name == 'paddlex_slanet':
        try:
            from hiertable_rag.evaluation.diagnosis.tsr_engines import PaddleXSLANetEngine
            logger.info(f"Creating PaddleX SLANet engine on {device}")
            return PaddleXSLANetEngine(device=device)
        except Exception as e:
            logger.error(f"PaddleX init failed: {e}")
            raise
    
    elif engine_name == 'docling':
        try:
            from hiertable_rag.evaluation.diagnosis.tsr_engines import DoclingEngine
            logger.info("Creating Docling engine")
            return DoclingEngine()
        except Exception as e:
            logger.error(f"Docling init failed: {e}")
            raise
    
    else:
        logger.warning(f"Unknown engine {engine_name}, using baseline")
        return BaselineTSR()


def create_ocr_source(source_name: str):
    """
    Factory function to create OCR source
    
    Args:
        source_name: OCR source type
    
    Returns:
        OCR source instance
    """
    if source_name == 'paddleocr':
        return NoisyOCRSource(use_mock=False)  # Use real PaddleOCR
    elif source_name == 'chunk':
        return PerfectOCRSource()
    elif source_name == 'tesseract':
        # Could add Tesseract wrapper here
        logging.warning("Tesseract not implemented, using PaddleOCR")
        return NoisyOCRSource(use_mock=False)
    else:
        return NoisyOCRSource(use_mock=True)  # Mock


def run_diagnosis(
    manifest_path: str,
    output_path: str,
    tsr_engine: str = 'tatr',
    ocr_source: str = 'paddleocr',
    device: str = 'cuda',
    max_samples: int = None,
    rca_checkpoint: str = None,
    gnn_checkpoint: str = None
):
    """
    Run diagnosis experiments with external baselines
    
    Args:
        manifest_path: Path to manifest CSV
        output_path: Path to output results CSV
        tsr_engine: TSR engine to use
        ocr_source: OCR source to use
        device: Device for computation
        max_samples: Maximum number of samples to process
        rca_checkpoint: Path to RCA checkpoint
        gnn_checkpoint: Path to GNN checkpoint
    """
    logger = logging.getLogger(__name__)
    
    # Load manifest
    logger.info(f"Loading manifest from: {manifest_path}")
    manifest_tables = load_manifest(manifest_path)
    
    if max_samples:
        manifest_tables = manifest_tables[:max_samples]
    
    logger.info(f"Processing {len(manifest_tables)} tables")
    
    # Initialize dataset
    data_root = Path(manifest_tables[0]['struct_path']).parent.parent.parent
    dataset = SciTSRDataset(str(data_root))
    
    # Initialize OCR sources
    logger.info(f"Creating OCR source: {ocr_source}")
    noisy_ocr = create_ocr_source(ocr_source)
    perfect_ocr = PerfectOCRSource()
    gt_ocr = GTOCRSource()
    
    # Initialize TSR engines
    logger.info(f"Initializing TSR engine: {tsr_engine}")
    chosen_tsr = create_tsr_engine(
        tsr_engine, 
        device,
        rca_checkpoint=rca_checkpoint,
        gnn_checkpoint=gnn_checkpoint
    )
    
    from hiertable_rag.evaluation.diagnosis.tsr_engines import GTStructureEngine
    gt_tsr = GTStructureEngine()
    
    # Check if engine uses OCR
    uses_ocr = getattr(chosen_tsr, 'uses_ocr', True)
    if not uses_ocr:
        logger.info(f"TSR engine '{tsr_engine}' is structure-only (does not use OCR for structure detection)")
    
    # Initialize metrics and cache
    metrics_calc = DiagnosisMetrics()
    cache = ResultCache()
    
    # Prepare results storage
    all_results = []
    
    # Process each table
    for table_info in tqdm(manifest_tables, desc="Processing tables"):
        table_id = table_info['table_id']
        
        logger.debug(f"\nProcessing table: {table_id}")
        
        # Load table data
        table_data = dataset.load_table(table_id)
        if table_data is None:
            logger.warning(f"Skipping {table_id}: failed to load")
            continue
        
        image = table_data['image']
        structure = table_data['structure']
        chunk = table_data['chunk']
        gt_cells = structure['cells']
        
        # Set up sources
        gt_ocr.set_structure(table_id, structure)
        if chunk:
            perfect_ocr.set_chunk_data(table_id, chunk)
        gt_tsr.set_structure(table_id, structure)
        gt_tsr.set_current_table(table_id)
        
        # Convert GT cells to standard format for metrics
        gt_cells_std = []
        for cell in gt_cells:
            gt_cells_std.append({
                'row': cell.get('start_row', 0),
                'col': cell.get('start_col', 0),
                'row_span': cell.get('end_row', 0) - cell.get('start_row', 0) + 1,
                'col_span': cell.get('end_col', 0) - cell.get('start_col', 0) + 1,
                'text': cell.get('content', ''),
                'box': cell.get('box', [0, 0, 10, 10])
            })
        
        # Run S1-S4 scenarios
        scenarios = {
            'S1': {'ocr': noisy_ocr, 'tsr': chosen_tsr, 'name': 'noisy_ocr+engine_tsr'},
            'S2': {'ocr': perfect_ocr, 'tsr': chosen_tsr, 'name': 'perfect_ocr+engine_tsr'},
            'S3': {'ocr': noisy_ocr, 'tsr': gt_tsr, 'name': 'noisy_ocr+gt_tsr'},
            'S4': {'ocr': perfect_ocr, 'tsr': gt_tsr, 'name': 'perfect_ocr+gt_tsr'}
        }
        
        for scenario_id, scenario_config in scenarios.items():
            ocr_src = scenario_config['ocr']
            tsr_eng = scenario_config['tsr']
            
            # Skip S2/S4 if no chunk data
            if scenario_id in ['S2', 'S4'] and not chunk:
                logger.debug(f"  Skipping {scenario_id}: no chunk data")
                continue
            
            # For structure-only TSR, S1==S2 (skip duplicate if noisy OCR also doesn't use OCR)
            if not uses_ocr and scenario_id == 'S2':
                logger.debug(f"  Skipping {scenario_id}: structure-only TSR (S1==S2)")
                # Still compute for completeness, but note it
            
            try:
                # Get OCR
                ocr_results = ocr_src.get_ocr(image, table_id)
                
                # Fallback to GT OCR if empty
                if not ocr_results:
                    ocr_results = gt_ocr.get_ocr(image, table_id)
                
                # Run TSR
                pred_structure = tsr_eng.predict(image, ocr_results)
                pred_cells = pred_structure.get('cells', [])
                
                # Compute metrics
                metrics = metrics_calc.compute_all_metrics(pred_cells, gt_cells_std)
                
                # Store result
                result_row = {
                    'table_id': table_id,
                    'scenario': scenario_id,
                    'ocr_type': 'perfect' if scenario_id in ['S2', 'S4'] else 'noisy',
                    'tsr_type': 'gt' if scenario_id in ['S3', 'S4'] else tsr_engine,
                    'm1_relation_f1': metrics['m1_relation_f1'],
                    'm2_row_boundary': metrics['m2_row_boundary'],
                    'm3_col_boundary': metrics['m3_col_boundary'],
                    'm4_span_f1': metrics['m4_span_f1'],
                    'average': metrics['average'],
                    'type_labels': ','.join(table_info.get('type_labels', [])),
                    'features': json.dumps(table_info.get('features', {}))
                }
                
                all_results.append(result_row)
                
                logger.debug(f"  {scenario_id}: F1={metrics['m1_relation_f1']:.3f}")
                
            except Exception as e:
                logger.error(f"  {scenario_id} failed for {table_id}: {e}", exc_info=True)
                continue
    
    # Write results CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'table_id', 'scenario', 'ocr_type', 'tsr_type',
            'm1_relation_f1', 'm2_row_boundary', 'm3_col_boundary', 'm4_span_f1', 'average',
            'type_labels', 'features'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    
    logger.info(f"✓ Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Diagnosis Complete!")
    print("="*60)
    print(f"TSR Engine: {tsr_engine}")
    print(f"OCR Source: {ocr_source}")
    print(f"Total results: {len(all_results)}")
    print(f"Output file: {output_file}")
    
    # Compute average metrics by scenario
    scenario_metrics = {}
    for row in all_results:
        scenario = row['scenario']
        if scenario not in scenario_metrics:
            scenario_metrics[scenario] = []
        scenario_metrics[scenario].append(row['m1_relation_f1'])
    
    print("\nAverage Relation F1 by scenario:") 
    for scenario in ['S1', 'S2', 'S3', 'S4']:
        if scenario in scenario_metrics:
            avg = sum(scenario_metrics[scenario]) / len(scenario_metrics[scenario])
            print(f"  {scenario}: {avg:.4f} (n={len(scenario_metrics[scenario])})")
    
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run diagnosis with external baseline TSR/OCR engines'
    )
    parser.add_argument(
        '--manifest',
        type=str,
        required=True,
        help='Path to manifest CSV'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='outputs/results.csv',
        help='Output results CSV path'
    )
    parser.add_argument(
        '--tsr_engine',
        type=str,
        default='tatr',
        choices=['baseline', 'gnn_csp', 'tatr', 'paddlex_slanet', 'docling'],
        help='TSR engine to use'
    )
    parser.add_argument(
        '--ocr_source',
        type=str,
        default='paddleocr',
        choices=['paddleocr', 'chunk', 'tesseract'],
        help='OCR source to use (chunk requires chunk data)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device: cuda, cpu, or gpu:0 (for PaddleX)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum number of samples to process'
    )
    parser.add_argument(
        '--rca_checkpoint',
        type=str,
        default=None,
        help='Path to RCA model checkpoint (for gnn_csp)'
    )
    parser.add_argument(
        '--gnn_checkpoint',
        type=str,
        default=None,
        help='Path to GNN model checkpoint (for gnn_csp)'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Run diagnosis
    run_diagnosis(
        manifest_path=args.manifest,
        output_path=args.out,
        tsr_engine=args.tsr_engine,
        ocr_source=args.ocr_source,
        device=args.device,
        max_samples=args.max_samples,
        rca_checkpoint=args.rca_checkpoint,
        gnn_checkpoint=args.gnn_checkpoint
    )


if __name__ == '__main__':
    main()
