#!/usr/bin/env python3
"""
Data collection pipeline for HierTable-RAG.

Downloads and processes datasets from multiple sources:
- DART financial reports
- HiTab dataset
- RealHiTBench dataset

Validates and creates manifest files for all collected data.
"""

import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
import logging
import json
import time
import re
from pathlib import Path
from collections import defaultdict

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

import pdfplumber
import pandas as pd
import numpy as np

from src.parsing.extractor import TableObject, TableExtractor
from src.parsing.hierarchy import HeaderTree, HierarchyExtractor

logger = logging.getLogger(__name__)


@dataclass
class TableData:
    """
    Structured table data with metadata.
    
    Attributes:
        table_id: Unique identifier
        source: Data source (dart, hitab, realhitbench)
        file_path: Path to source file
        table_object: TableObject instance
        metadata: Additional metadata (company, year, report_type, etc.)
        complexity_metrics: Complexity metrics (hierarchy levels, merged cells, etc.)
        annotations: Manual annotations (optional)
    """
    table_id: str
    source: str
    file_path: str
    table_object: Optional[TableObject] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    complexity_metrics: Dict[str, Any] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "table_id": self.table_id,
            "source": self.source,
            "file_path": str(self.file_path),
            "metadata": self.metadata,
            "complexity_metrics": self.complexity_metrics,
            "annotations": self.annotations,
        }


@dataclass
class ValidationReport:
    """
    Dataset validation report.
    
    Attributes:
        total_tables: Total number of tables
        valid_tables: Number of valid tables
        invalid_tables: Number of invalid tables
        hierarchy_stats: Statistics on hierarchy complexity
        merged_cell_stats: Statistics on merged cells
        size_distribution: Table size distribution
        problematic_tables: List of problematic table IDs with reasons
        summary: Summary statistics dictionary
    """
    total_tables: int = 0
    valid_tables: int = 0
    invalid_tables: int = 0
    hierarchy_stats: Dict[str, Any] = field(default_factory=dict)
    merged_cell_stats: Dict[str, Any] = field(default_factory=dict)
    size_distribution: Dict[str, Any] = field(default_factory=dict)
    problematic_tables: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class DataCollector:
    """
    Main data collection pipeline.
    """
    
    def __init__(
        self,
        output_dir: str = "data/collected",
        dart_api_key: Optional[str] = None,
    ) -> None:
        """
        Initialize data collector.
        
        Args:
            output_dir: Output directory for collected data
            dart_api_key: DART API key (optional, can use env var)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.dart_api_key = dart_api_key or self._get_dart_api_key()
        
        self.table_extractor = TableExtractor()
        self.hierarchy_extractor = HierarchyExtractor()
        
        # Error log
        self.error_log: List[Dict[str, Any]] = []
        
        logger.info(f"DataCollector initialized (output_dir: {output_dir})")
    
    def _get_dart_api_key(self) -> Optional[str]:
        """Get DART API key from environment."""
        import os
        return os.getenv("DART_API_KEY")
    
    def scrape_dart_tables(
        self,
        company_codes: List[str],
        years: List[int],
        report_types: Optional[List[str]] = None,
        min_header_levels: int = 2,
    ) -> List[TableData]:
        """
        Scrape tables from DART financial reports.
        
        Args:
            company_codes: List of company codes (6-digit)
            years: List of years to download
            report_types: Report types (e.g., ['A001', 'A002'] for 사업보고서, 반기보고서)
            min_header_levels: Minimum header levels to filter (default: 2)
            
        Returns:
            List of TableData objects
        """
        logger.info(
            f"Scraping DART tables for {len(company_codes)} companies, "
            f"{len(years)} years"
        )
        
        if not self.dart_api_key:
            logger.error("DART API key not provided")
            return []
        
        # Import DART downloader if available
        try:
            from src.dart import DARTDownloader
            downloader = DARTDownloader(
                self.dart_api_key,
                download_dir=str(self.output_dir / "dart_pdfs"),
            )
        except ImportError:
            logger.error("DART downloader not available")
            return []
        
        report_types = report_types or ["A001", "A002"]  # 사업보고서, 반기보고서
        
        all_tables: List[TableData] = []
        
        # Progress bar
        total_combinations = len(company_codes) * len(years) * len(report_types)
        iterator = self._get_iterator(
            range(total_combinations), desc="Downloading DART reports"
        )
        
        # Get list of companies to find company info
        companies = downloader.get_company_list(limit=100)
        company_dict = {c.get("corp_code"): c for c in companies}
        
        for company_code in company_codes:
            company_info = company_dict.get(company_code)
            if not company_info:
                # Try to get company info from DART API
                try:
                    company_info = {
                        "corp_code": company_code,
                        "corp_name": company_code,
                    }
                except Exception:
                    logger.warning(f"Company code {company_code} not found")
                    continue
            
            corp_name = company_info.get("corp_name", company_code)
            
            for year in years:
                # Get reports for the company and year
                try:
                    start_date = f"{year}0101"
                    end_date = f"{year}1231"
                    
                    reports = downloader.get_reports(
                        corp_code=company_code,
                        start_date=start_date,
                        end_date=end_date,
                        report_type="all",
                    )
                    
                    if not reports:
                        continue
                    
                    # Filter by report type
                    filtered_reports = [
                        r for r in reports
                        if r.get("report_nm") and any(
                            rt in r.get("report_nm", "") for rt in report_types
                        )
                    ]
                    
                    for report in filtered_reports[:5]:  # Limit per company/year
                        try:
                            rcept_no = report.get("rcept_no")
                            report_nm = report.get("report_nm", "")
                            
                            # Download PDF
                            pdf_path = downloader.download_report_pdf(
                                rcept_no, corp_name, report_nm
                            )
                            
                            if not pdf_path or not pdf_path.exists():
                                continue
                            
                            # Extract tables from PDF
                            tables = self._extract_tables_from_pdf(
                                pdf_path, source="dart"
                            )
                            
                            # Filter for complex tables
                            complex_tables = self._filter_complex_tables(
                                tables, min_header_levels=min_header_levels
                            )
                            
                            # Convert to TableData
                            for table_obj in complex_tables:
                                table_data = self._create_table_data(
                                    table_obj,
                                    source="dart",
                                    file_path=pdf_path,
                                    metadata={
                                        "company_code": company_code,
                                        "company_name": corp_name,
                                        "year": year,
                                        "report_type": report_nm,
                                        "rcept_no": rcept_no,
                                    },
                                )
                                all_tables.append(table_data)
                            
                            next(iterator, None)  # Update progress
                            
                        except Exception as e:
                            error_info = {
                                "type": "dart_extraction",
                                "company_code": company_code,
                                "year": year,
                                "error": str(e),
                                "timestamp": time.time(),
                            }
                            self.error_log.append(error_info)
                            logger.error(f"Error extracting tables: {e}")
                            continue
                
                except Exception as e:
                    error_info = {
                        "type": "dart_download",
                        "company_code": company_code,
                        "year": year,
                        "error": str(e),
                        "timestamp": time.time(),
                    }
                    self.error_log.append(error_info)
                    logger.error(f"Error downloading DART report: {e}")
                    continue
        
        logger.info(f"Scraped {len(all_tables)} complex tables from DART")
        return all_tables
    
    def download_hitab_dataset(self, output_dir: str) -> List[TableData]:
        """
        Download HiTab dataset from official source.
        
        Args:
            output_dir: Output directory for HiTab data
            
        Returns:
            List of TableData objects
        """
        logger.info("Downloading HiTab dataset")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_tables: List[TableData] = []
        
        # Try multiple methods to get HiTab dataset
        hitab_repo = "https://github.com/microsoft/HiTab.git"
        repo_dir = output_path / "HiTab"
        
        try:
            # Method 1: Clone from GitHub (preferred)
            import subprocess
            if not repo_dir.exists():
                logger.info(f"Cloning HiTab repository from {hitab_repo}")
                try:
                    subprocess.run(
                        ["git", "clone", hitab_repo, str(repo_dir)],
                        check=True,
                        capture_output=True,
                        timeout=300
                    )
                except (subprocess.CalledProcessError, FileNotFoundError) as e:
                    logger.warning(f"Git clone failed: {e}, trying ZIP download...")
                    # Fallback to ZIP download
                    zip_url = "https://github.com/microsoft/HiTab/archive/refs/heads/main.zip"
                    zip_path = output_path / "HiTab-main.zip"
                    
                    if not zip_path.exists():
                        logger.info(f"Downloading HiTab repository ZIP from GitHub")
                        if REQUESTS_AVAILABLE:
                            self._download_file(zip_url, zip_path)
                        else:
                            raise ImportError("requests library required for downloads")
                    
                    # Extract ZIP if needed
                    if zip_path.exists():
                        import zipfile
                        logger.info("Extracting HiTab ZIP file...")
                        extract_dir = output_path / "HiTab-main"
                        with zipfile.ZipFile(zip_path, "r") as zip_ref:
                            zip_ref.extractall(output_path)
                        if extract_dir.exists():
                            extract_dir.rename(repo_dir)
            
            # Parse HiTab format from repository
            if repo_dir.exists():
                logger.info("Parsing HiTab dataset...")
                tables = self._parse_hitab_format(repo_dir, "all")
                all_tables.extend(tables)
            else:
                raise FileNotFoundError("HiTab repository not found after download")
            
        except Exception as e:
            error_info = {
                "type": "hitab_download",
                "method": "git_clone",
                "error": str(e),
                "timestamp": time.time(),
            }
            self.error_log.append(error_info)
            logger.error(f"Error cloning HiTab repository: {e}")
            
            # Method 2: Try HuggingFace if available
            try:
                logger.info("Trying HuggingFace dataset...")
                import datasets
                hitab_dataset = datasets.load_dataset("Yale-LILY/HiTab")
                # Convert HuggingFace dataset to TableData
                for split_name in hitab_dataset.keys():
                    split_data = hitab_dataset[split_name]
                    for idx, example in enumerate(split_data):
                        table_data = self._convert_hitab_example_to_tabledata(
                            example, split_name, idx
                        )
                        all_tables.append(table_data)
                logger.info(f"Loaded {len(all_tables)} tables from HuggingFace")
            except Exception as hf_error:
                logger.warning(f"HuggingFace download also failed: {hf_error}")
                logger.info("Please manually download HiTab dataset from: https://github.com/Yale-LILY/HiTab")
        
        logger.info(f"Downloaded {len(all_tables)} tables from HiTab")
        return all_tables
    
    def download_realhitbench(self, output_dir: str) -> List[TableData]:
        """
        Download RealHiTBench dataset.
        
        Handles multiple input formats (LaTeX, HTML, PNG).
        
        Args:
            output_dir: Output directory for RealHiTBench data
            
        Returns:
            List of TableData objects
        """
        logger.info("Downloading RealHiTBench dataset")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_tables: List[TableData] = []
        
        # Try multiple methods to get RealHiTBench dataset
        realhitbench_repo = "https://github.com/RealHiTBench/RealHiTBench.git"
        repo_dir = output_path / "RealHiTBench"
        
        try:
            # Method 1: Download ZIP from GitHub
            zip_url = "https://github.com/RealHiTBench/RealHiTBench/archive/refs/heads/main.zip"
            zip_path = output_path / "RealHiTBench-main.zip"
            
            if not repo_dir.exists() and not zip_path.exists():
                logger.info(f"Downloading RealHiTBench repository ZIP from GitHub")
                if REQUESTS_AVAILABLE:
                    self._download_file(zip_url, zip_path)
                else:
                    raise ImportError("requests library required for downloads")
            
            # Extract ZIP if needed
            if zip_path.exists() and not repo_dir.exists():
                import zipfile
                logger.info("Extracting RealHiTBench ZIP file...")
                extract_dir = output_path / "RealHiTBench-main"
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(output_path)
                if extract_dir.exists():
                    extract_dir.rename(repo_dir)
            
            # Parse different formats from repository
            if repo_dir.exists():
                logger.info("Parsing RealHiTBench dataset...")
                if (repo_dir / "latex").exists():
                    latex_tables = self._parse_latex_tables(repo_dir / "latex")
                    all_tables.extend(latex_tables)
                if (repo_dir / "html").exists():
                    html_tables = self._parse_html_tables(repo_dir / "html")
                    all_tables.extend(html_tables)
                if (repo_dir / "png").exists():
                    png_tables = self._parse_png_tables(repo_dir / "png")
                    all_tables.extend(png_tables)
                
                # Also check for data directory
                if (repo_dir / "data").exists():
                    logger.info("Found data directory, parsing...")
                    data_tables = self._parse_realhitbench_data(repo_dir / "data")
                    all_tables.extend(data_tables)
            else:
                raise FileNotFoundError("RealHiTBench repository not found after download")
            
        except Exception as e:
            error_info = {
                "type": "realhitbench_download",
                "method": "git_clone",
                "error": str(e),
                "timestamp": time.time(),
            }
            self.error_log.append(error_info)
            logger.error(f"Error cloning RealHiTBench repository: {e}")
            logger.info("Please manually download RealHiTBench dataset from: https://github.com/RealHiTBench/RealHiTBench")
        
        logger.info(f"Downloaded {len(all_tables)} tables from RealHiTBench")
        return all_tables
    
    def validate_dataset(self, tables: List[TableData]) -> ValidationReport:
        """
        Validate dataset and generate report.
        
        Args:
            tables: List of TableData objects
            
        Returns:
            ValidationReport with statistics and flags
        """
        logger.info(f"Validating {len(tables)} tables")
        
        report = ValidationReport(total_tables=len(tables))
        
        hierarchy_levels = []
        merged_cell_counts = []
        table_sizes = []
        valid_count = 0
        
        for table_data in tables:
            try:
                table_obj = table_data.table_object
                if not table_obj:
                    report.problematic_tables.append({
                        "table_id": table_data.table_id,
                        "reason": "Missing table_object",
                    })
                    continue
                
                # Count hierarchy levels
                try:
                    header_tree = self.hierarchy_extractor.extract_header_tree(
                        table_obj
                    )
                    levels = header_tree.max_level
                    hierarchy_levels.append(levels)
                    table_data.complexity_metrics["hierarchy_levels"] = levels
                except Exception as e:
                    hierarchy_levels.append(0)
                    report.problematic_tables.append({
                        "table_id": table_data.table_id,
                        "reason": f"Hierarchy extraction failed: {e}",
                    })
                
                # Count merged cells
                merged_count = self._count_merged_cells(table_obj)
                merged_cell_counts.append(merged_count)
                table_data.complexity_metrics["merged_cells"] = merged_count
                
                # Table size
                num_rows = table_obj.num_rows()
                num_cols = table_obj.num_cols()
                table_sizes.append({"rows": num_rows, "cols": num_cols})
                table_data.complexity_metrics["num_rows"] = num_rows
                table_data.complexity_metrics["num_cols"] = num_cols
                
                # Validation checks
                is_valid = (
                    num_rows > 0
                    and num_cols > 0
                    and len(hierarchy_levels) > 0
                )
                
                if is_valid:
                    valid_count += 1
                else:
                    report.problematic_tables.append({
                        "table_id": table_data.table_id,
                        "reason": "Invalid table structure",
                    })
                
            except Exception as e:
                report.problematic_tables.append({
                    "table_id": table_data.table_id,
                    "reason": f"Validation error: {e}",
                })
                continue
        
        report.valid_tables = valid_count
        report.invalid_tables = len(tables) - valid_count
        
        # Hierarchy statistics
        if hierarchy_levels:
            report.hierarchy_stats = {
                "mean": float(np.mean(hierarchy_levels)),
                "median": float(np.median(hierarchy_levels)),
                "min": int(np.min(hierarchy_levels)),
                "max": int(np.max(hierarchy_levels)),
                "std": float(np.std(hierarchy_levels)),
            }
        
        # Merged cell statistics
        if merged_cell_counts:
            report.merged_cell_stats = {
                "mean": float(np.mean(merged_cell_counts)),
                "median": float(np.median(merged_cell_counts)),
                "min": int(np.min(merged_cell_counts)),
                "max": int(np.max(merged_cell_counts)),
                "total": int(np.sum(merged_cell_counts)),
            }
        
        # Size distribution
        if table_sizes:
            rows = [s["rows"] for s in table_sizes]
            cols = [s["cols"] for s in table_sizes]
            
            report.size_distribution = {
                "rows": {
                    "mean": float(np.mean(rows)),
                    "median": float(np.median(rows)),
                    "min": int(np.min(rows)),
                    "max": int(np.max(rows)),
                },
                "cols": {
                    "mean": float(np.mean(cols)),
                    "median": float(np.median(cols)),
                    "min": int(np.min(cols)),
                    "max": int(np.max(cols)),
                },
            }
        
        # Summary
        report.summary = {
            "total_tables": report.total_tables,
            "valid_tables": report.valid_tables,
            "invalid_tables": report.invalid_tables,
            "validity_rate": (
                report.valid_tables / report.total_tables
                if report.total_tables > 0
                else 0.0
            ),
            "avg_hierarchy_levels": (
                report.hierarchy_stats.get("mean", 0.0)
            ),
            "avg_merged_cells": (
                report.merged_cell_stats.get("mean", 0.0)
            ),
        }
        
        logger.info(
            f"Validation complete: {report.valid_tables}/{report.total_tables} "
            f"valid tables"
        )
        
        return report
    
    def create_manifest(
        self,
        tables: List[TableData],
        manifest_path: Optional[str] = None,
    ) -> str:
        """
        Create manifest file with table metadata.
        
        Args:
            tables: List of TableData objects
            manifest_path: Path to save manifest (optional)
            
        Returns:
            Path to created manifest file
        """
        if manifest_path is None:
            manifest_path = self.output_dir / "manifest.json"
        else:
            manifest_path = Path(manifest_path)
        
        logger.info(f"Creating manifest file: {manifest_path}")
        
        manifest = {
            "version": "1.0",
            "created_at": time.time(),
            "total_tables": len(tables),
            "tables": [table.to_dict() for table in tables],
        }
        
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Manifest saved to {manifest_path}")
        return str(manifest_path)
    
    def save_error_log(self, error_log_path: Optional[str] = None) -> str:
        """
        Save error log to file.
        
        Args:
            error_log_path: Path to save error log (optional)
            
        Returns:
            Path to error log file
        """
        if error_log_path is None:
            error_log_path = self.output_dir / "error_log.json"
        else:
            error_log_path = Path(error_log_path)
        
        with open(error_log_path, "w", encoding="utf-8") as f:
            json.dump(self.error_log, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Error log saved to {error_log_path}")
        return str(error_log_path)
    
    # Helper methods
    
    def _extract_tables_from_pdf(
        self, pdf_path: Path, source: str
    ) -> List[TableObject]:
        """Extract tables from PDF file."""
        try:
            tables = self.table_extractor.extract_from_pdf(str(pdf_path))
            return tables
        except Exception as e:
            error_info = {
                "type": "pdf_extraction",
                "file": str(pdf_path),
                "error": str(e),
                "timestamp": time.time(),
            }
            self.error_log.append(error_info)
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
            return []
    
    def _filter_complex_tables(
        self, tables: List[TableObject], min_header_levels: int = 2
    ) -> List[TableObject]:
        """Filter tables by complexity criteria."""
        complex_tables = []
        
        for table in tables:
            try:
                header_tree = self.hierarchy_extractor.extract_header_tree(table)
                if header_tree.max_level >= min_header_levels:
                    complex_tables.append(table)
            except Exception:
                # If hierarchy extraction fails, check for merged cells
                merged_count = self._count_merged_cells(table)
                if merged_count > 0:
                    complex_tables.append(table)
        
        return complex_tables
    
    def _count_merged_cells(self, table: TableObject) -> int:
        """Count merged cells in table."""
        count = 0
        for row in table.cells:
            for cell in row:
                if cell and (cell.rowspan > 1 or cell.colspan > 1):
                    count += 1
        return count
    
    def _create_table_data(
        self,
        table_obj: TableObject,
        source: str,
        file_path: Path,
        metadata: Dict[str, Any],
    ) -> TableData:
        """Create TableData object from TableObject."""
        table_id = f"{source}_{Path(file_path).stem}_{int(time.time())}"
        
        return TableData(
            table_id=table_id,
            source=source,
            file_path=str(file_path),
            table_object=table_obj,
            metadata=metadata,
        )
    
    def _download_file(self, url: str, output_path: Path) -> None:
        """Download file from URL."""
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for downloads")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        total_size = int(response.headers.get("content-length", 0))
        
        with open(output_path, "wb") as f:
            if TQDM_AVAILABLE and total_size > 0:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc="Downloading"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
    
    def _parse_hitab_format(
        self, data_dir: Path, split: str
    ) -> List[TableData]:
        """Parse HiTab dataset format."""
        tables = []
        
        # HiTab data files are in data/ directory as JSONL files
        data_dir_path = data_dir / "data"
        if not data_dir_path.exists():
            data_dir_path = data_dir
        
        # Extract tables.zip if needed
        tables_zip = data_dir_path / "tables.zip"
        tables_dir = data_dir_path / "tables"
        if tables_zip.exists() and not tables_dir.exists():
            logger.info("Extracting tables.zip...")
            import zipfile
            with zipfile.ZipFile(tables_zip, "r") as zip_ref:
                zip_ref.extractall(data_dir_path)
        
        # Load table files into memory
        table_files = {}
        if tables_dir.exists():
            for table_file in tables_dir.rglob("*.json"):
                try:
                    with open(table_file, 'r', encoding='utf-8') as f:
                        table_data = json.load(f)
                        # Extract table_id from filename or data
                        table_id = table_file.stem
                        table_files[table_id] = table_data
                except Exception as e:
                    logger.warning(f"Error loading table file {table_file}: {e}")
        
        logger.info(f"Loaded {len(table_files)} table files")
        
        jsonl_files = list(data_dir_path.glob("*_samples.jsonl"))
        json_files = list(data_dir_path.glob("*.json"))
        
        logger.info(f"Found {len(jsonl_files)} JSONL files and {len(json_files)} JSON files")
        
        # Parse JSONL files (HiTab format)
        for jsonl_file in jsonl_files:
            try:
                with open(jsonl_file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                            table_data = self._parse_hitab_item(item, jsonl_file, line_num, table_files)
                            if table_data:
                                tables.append(table_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Error parsing line {line_num} in {jsonl_file}: {e}")
                            continue
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")
                continue
        
        # Parse JSON files (common format for HiTab)
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    # Handle different JSON structures
                    if isinstance(data, list):
                        for item in data:
                            table_data = self._parse_hitab_item(item, json_file)
                            if table_data:
                                tables.append(table_data)
                    elif isinstance(data, dict):
                        # Check for common keys
                        if 'tables' in data:
                            for table_item in data['tables']:
                                table_data = self._parse_hitab_item(table_item, json_file)
                                if table_data:
                                    tables.append(table_data)
                        elif 'data' in data:
                            for item in data['data']:
                                table_data = self._parse_hitab_item(item, json_file)
                                if table_data:
                                    tables.append(table_data)
                        else:
                            table_data = self._parse_hitab_item(data, json_file)
                            if table_data:
                                tables.append(table_data)
            except Exception as e:
                logger.warning(f"Error parsing {json_file}: {e}")
                continue
        
        logger.info(f"Parsed {len(tables)} tables from HiTab format")
        return tables
    
    def _parse_hitab_item(self, item: Dict[str, Any], source_file: Path, line_num: int = 0, table_files: Dict[str, Any] = None) -> Optional[TableData]:
        """Parse a single HiTab item into TableData."""
        try:
            # HiTab format: each line has 'table_id', 'question', 'answer', etc.
            sample_id = item.get('id') or f"hitab_{Path(source_file).stem}_{line_num}"
            table_id = item.get('table_id', '')
            
            # Load table data from table_files dict
            table_info = None
            if table_files and table_id:
                # Try to find table by table_id
                table_info = table_files.get(table_id)
                if not table_info:
                    # Try with different variations of table_id
                    for tid, tdata in table_files.items():
                        if table_id in tid or tid in table_id:
                            table_info = tdata
                            break
            
            # If table_info not found, try to get from item
            if not table_info:
                table_info = item.get('table', {})
            
            if table_info:
                # HiTab table format: has 'texts' (matrix) and 'merged_regions'
                texts = table_info.get('texts', [])
                merged_regions = table_info.get('merged_regions', [])
                
                # Convert to TableObject if possible
                table_obj = None
                try:
                    if texts:
                        # Convert texts matrix to TableObject
                        table_obj = self._texts_to_table_object(texts, merged_regions)
                except Exception as e:
                    logger.debug(f"Could not convert to TableObject: {e}")
                
                return TableData(
                    table_id=sample_id,
                    source="hitab",
                    file_path=f"{source_file}:{line_num}",
                    table_object=table_obj,
                    metadata={
                        "table_id": table_id,
                        "question": item.get('question', ''),
                        "answer": item.get('answer', []),
                        "sub_sentence": item.get('sub_sentence', ''),
                        "split": Path(source_file).stem.replace('_samples', ''),
                        "linked_cells": item.get('linked_cells', {}),
                        "aggregation": item.get('aggregation', []),
                        "table_source": item.get('table_source', ''),
                    }
                )
        except Exception as e:
            logger.debug(f"Error parsing HiTab item: {e}")
            return None
    
    def _texts_to_table_object(self, texts: List[List[str]], merged_regions: List[Dict]) -> Optional[TableObject]:
        """Convert HiTab texts matrix to TableObject."""
        try:
            from src.parsing.extractor import TableObject, Cell
            
            cells = []
            for i, row in enumerate(texts):
                row_cells = []
                for j, val in enumerate(row):
                    # Check if this cell is part of a merged region
                    rowspan = 1
                    colspan = 1
                    for merged in merged_regions:
                        if (merged['first_row'] <= i <= merged['last_row'] and
                            merged['first_column'] <= j <= merged['last_column']):
                            if i == merged['first_row'] and j == merged['first_column']:
                                rowspan = merged['last_row'] - merged['first_row'] + 1
                                colspan = merged['last_column'] - merged['first_column'] + 1
                            else:
                                # This cell is merged, skip it
                                continue
                    
                    cell = Cell(
                        row=i,
                        col=j,
                        value=str(val) if val else "",
                        rowspan=rowspan,
                        colspan=colspan
                    )
                    row_cells.append(cell)
                cells.append(row_cells)
            
            return TableObject(cells=cells)
        except Exception as e:
            logger.debug(f"Error converting texts to TableObject: {e}")
            return None
    
    def _df_to_table_object(self, df: pd.DataFrame) -> Optional[TableObject]:
        """Convert pandas DataFrame to TableObject."""
        try:
            from src.parsing.extractor import TableObject, Cell
            
            cells = []
            for i, row in df.iterrows():
                row_cells = []
                for j, val in enumerate(row):
                    cell = Cell(
                        row=i,
                        col=j,
                        value=str(val) if pd.notna(val) else "",
                        rowspan=1,
                        colspan=1
                    )
                    row_cells.append(cell)
                cells.append(row_cells)
            
            return TableObject(cells=cells)
        except Exception as e:
            logger.debug(f"Error converting DataFrame: {e}")
            return None
    
    def _convert_hitab_example_to_tabledata(
        self, example: Dict[str, Any], split: str, idx: int
    ) -> TableData:
        """Convert HuggingFace HiTab example to TableData."""
        table_id = f"hitab_{split}_{idx}"
        return TableData(
            table_id=table_id,
            source="hitab",
            file_path=f"huggingface_{split}_{idx}",
            metadata={
                "question": example.get('question', ''),
                "answer": example.get('answer', ''),
                "split": split,
            }
        )
    
    def _parse_realhitbench_data(self, data_dir: Path) -> List[TableData]:
        """Parse RealHiTBench data directory."""
        tables = []
        
        # Look for JSON files with table data
        json_files = list(data_dir.rglob("*.json"))
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    if isinstance(data, list):
                        for item in data:
                            table_data = self._parse_realhitbench_item(item, json_file)
                            if table_data:
                                tables.append(table_data)
                    elif isinstance(data, dict):
                        table_data = self._parse_realhitbench_item(data, json_file)
                        if table_data:
                            tables.append(table_data)
            except Exception as e:
                logger.warning(f"Error parsing {json_file}: {e}")
                continue
        
        return tables
    
    def _parse_realhitbench_item(
        self, item: Dict[str, Any], source_file: Path
    ) -> Optional[TableData]:
        """Parse a single RealHiTBench item."""
        try:
            table_id = item.get('id') or item.get('table_id') or f"realhitbench_{int(time.time())}"
            
            return TableData(
                table_id=table_id,
                source="realhitbench",
                file_path=str(source_file),
                metadata={
                    "format": item.get('format', 'unknown'),
                    "question": item.get('question'),
                    "answer": item.get('answer'),
                }
            )
        except Exception as e:
            logger.debug(f"Error parsing RealHiTBench item: {e}")
            return None
    
    def _parse_latex_tables(self, data_dir: Path) -> List[TableData]:
        """Parse LaTeX table files."""
        tables = []
        # Implementation would parse LaTeX tables
        logger.warning("LaTeX parsing not fully implemented")
        return tables
    
    def _parse_html_tables(self, data_dir: Path) -> List[TableData]:
        """Parse HTML table files."""
        tables = []
        # Implementation would parse HTML tables
        logger.warning("HTML parsing not fully implemented")
        return tables
    
    def _parse_png_tables(self, data_dir: Path) -> List[TableData]:
        """Parse PNG table images (would require OCR)."""
        tables = []
        # Implementation would use OCR to extract tables from images
        logger.warning("PNG/OCR parsing not fully implemented")
        return tables
    
    def _get_iterator(self, iterable, desc: str = ""):
        """Get iterator with optional progress bar."""
        if TQDM_AVAILABLE:
            return tqdm(iterable, desc=desc)
        return iterable


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Data collection pipeline for HierTable-RAG"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/collected", help="Output directory"
    )
    parser.add_argument(
        "--dart-api-key", type=str, help="DART API key (or use DART_API_KEY env var)"
    )
    parser.add_argument(
        "--company-codes",
        nargs="+",
        help="Company codes for DART scraping",
    )
    parser.add_argument(
        "--years", type=int, nargs="+", help="Years for DART scraping"
    )
    parser.add_argument(
        "--download-hitab", action="store_true", help="Download HiTab dataset"
    )
    parser.add_argument(
        "--download-realhitbench",
        action="store_true",
        help="Download RealHiTBench dataset",
    )
    parser.add_argument(
        "--validate-only",
        type=str,
        help="Validate existing manifest file",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    collector = DataCollector(
        output_dir=args.output_dir, dart_api_key=args.dart_api_key
    )
    
    all_tables: List[TableData] = []
    
    # DART scraping
    if args.company_codes and args.years:
        dart_tables = collector.scrape_dart_tables(
            company_codes=args.company_codes, years=args.years
        )
        all_tables.extend(dart_tables)
    
    # HiTab download
    if args.download_hitab:
        hitab_tables = collector.download_hitab_dataset(
            str(Path(args.output_dir) / "hitab")
        )
        all_tables.extend(hitab_tables)
    
    # RealHiTBench download
    if args.download_realhitbench:
        realhitbench_tables = collector.download_realhitbench(
            str(Path(args.output_dir) / "realhitbench")
        )
        all_tables.extend(realhitbench_tables)
    
    # Validate dataset
    if all_tables:
        report = collector.validate_dataset(all_tables)
        
        # Save validation report
        report_path = Path(args.output_dir) / "validation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\nValidation Report:")
        print(f"  Total tables: {report.total_tables}")
        print(f"  Valid tables: {report.valid_tables}")
        print(f"  Invalid tables: {report.invalid_tables}")
        print(f"  Average hierarchy levels: {report.hierarchy_stats.get('mean', 0):.2f}")
        print(f"  Average merged cells: {report.merged_cell_stats.get('mean', 0):.2f}")
    
    # Create manifest
    if all_tables:
        manifest_path = collector.create_manifest(all_tables)
        print(f"\nManifest created: {manifest_path}")
    
    # Save error log
    if collector.error_log:
        error_log_path = collector.save_error_log()
        print(f"Error log saved: {error_log_path}")
        print(f"  Total errors: {len(collector.error_log)}")


if __name__ == "__main__":
    main()

