"""
Indexing module for hierarchical table structures.

Implements multi-level FAISS indexing with different granularities:
table, subtable, row, and cell levels.
"""

from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field, asdict
import logging
import json
import pickle
import time
import uuid

try:
    import faiss
    import numpy as np
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

from src.parsing.cell_merger import NormalizedTable, NormalizedCell
from src.parsing.hierarchy import HeaderTree, HeaderNode
from src.encoding.tree_encoder import TreeEncoder

logger = logging.getLogger(__name__)


@dataclass
class IndexMetadata:
    """
    Metadata for an indexed item.
    
    Attributes:
        item_id: Unique identifier for this indexed item
        source_table_id: ID of source table
        granularity: Level of granularity (table, subtable, row, cell)
        hierarchy_path: Path from root (e.g., "Table > Category A > Subcategory A1")
        coordinates: Row and column ranges ((row_start, row_end), (col_start, col_end))
        parent_ids: List of parent item IDs (coarser granularity)
        child_ids: List of child item IDs (finer granularity)
        text: Original text representation
    """
    item_id: str
    source_table_id: str
    granularity: str
    hierarchy_path: str
    coordinates: Tuple[Tuple[int, int], Tuple[int, int]]  # ((row_start, row_end), (col_start, col_end))
    parent_ids: List[str] = field(default_factory=list)
    child_ids: List[str] = field(default_factory=list)
    text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "item_id": self.item_id,
            "source_table_id": self.source_table_id,
            "granularity": self.granularity,
            "hierarchy_path": self.hierarchy_path,
            "coordinates": self.coordinates,
            "parent_ids": self.parent_ids,
            "child_ids": self.child_ids,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IndexMetadata":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            source_table_id=data["source_table_id"],
            granularity=data["granularity"],
            hierarchy_path=data["hierarchy_path"],
            coordinates=tuple(
                tuple(coord) for coord in data["coordinates"]
            ),
            parent_ids=data.get("parent_ids", []),
            child_ids=data.get("child_ids", []),
            text=data.get("text", ""),
        )


@dataclass
class HierarchicalIndex:
    """
    Multi-level hierarchical index structure.
    
    Attributes:
        table_index: FAISS index for table-level embeddings
        subtable_index: FAISS index for subtable-level embeddings
        row_index: FAISS index for row-level embeddings
        cell_index: FAISS index for cell-level embeddings
        metadata: List of IndexMetadata objects (indexed by position)
        embedding_dim: Dimension of embeddings
        index_type: Type of FAISS index used
    """
    table_index: Optional[Any] = None
    subtable_index: Optional[Any] = None
    row_index: Optional[Any] = None
    cell_index: Optional[Any] = None
    metadata: List[IndexMetadata] = field(default_factory=list)
    embedding_dim: int = 384  # Default for all-MiniLM-L6-v2
    index_type: str = "Flat"

    def get_index(self, granularity: str) -> Optional[Any]:
        """
        Get FAISS index for given granularity.
        
        Args:
            granularity: One of "table", "subtable", "row", "cell"
            
        Returns:
            FAISS index or None
        """
        index_map = {
            "table": self.table_index,
            "subtable": self.subtable_index,
            "row": self.row_index,
            "cell": self.cell_index,
        }
        return index_map.get(granularity)


class HierarchicalIndexer:
    """
    Build multi-level FAISS indices for hierarchical table structures.
    
    Creates separate indexes for different granularities:
    - table_level: Entire table representations
    - subtable_level: Logical sections (by top-level headers)
    - row_level: Individual rows
    - cell_level: Individual cells with context
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_type: str = "Flat",
        embedding_dim: Optional[int] = None,
        chunking_strategy: str = "hierarchical",
        max_chunk_size: int = 1000,
    ) -> None:
        """
        Initialize hierarchical indexer.
        
        Args:
            embedding_model: Name of sentence-transformers model
            index_type: FAISS index type ("Flat", "IVF", "HNSW")
            embedding_dim: Embedding dimension (auto-detect if None)
            chunking_strategy: Strategy for chunking large tables
                              ("hierarchical", "fixed", "adaptive")
            max_chunk_size: Maximum chunk size for chunking strategy
            
        Raises:
            ImportError: If required libraries are not available
        """
        if not FAISS_AVAILABLE:
            raise ImportError(
                "faiss-cpu is required. Install with: pip install faiss-cpu"
            )
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            )
        
        self.embedding_model_name = embedding_model
        self.index_type = index_type
        self.chunking_strategy = chunking_strategy
        self.max_chunk_size = max_chunk_size
        
        # Load embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            self.embedding_dim = (
                embedding_dim or self.embedding_model.get_sentence_embedding_dimension()
            )
            logger.info(
                f"Loaded embedding model: {embedding_model} "
                f"(dim={self.embedding_dim})"
            )
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
        
        # Initialize tree encoder
        self.tree_encoder = TreeEncoder()
        
        # Initialize index structure
        self.index: Optional[HierarchicalIndex] = None
        
        logger.info(
            f"HierarchicalIndexer initialized "
            f"(index_type={index_type}, chunking={chunking_strategy})"
        )

    def build_hierarchical_index(
        self, tables: List[Tuple[NormalizedTable, HeaderTree]]
    ) -> HierarchicalIndex:
        """
        Create separate FAISS indexes for each granularity level.
        
        Args:
            tables: List of tuples (NormalizedTable, HeaderTree)
            
        Returns:
            HierarchicalIndex with all granularity levels
        """
        logger.info(f"Building hierarchical index for {len(tables)} tables")
        start_time = time.time()
        
        # Initialize index structure
        self.index = HierarchicalIndex(
            embedding_dim=self.embedding_dim,
            index_type=self.index_type,
        )
        
        # Create FAISS indexes for each granularity
        self.index.table_index = self._create_faiss_index()
        self.index.subtable_index = self._create_faiss_index()
        self.index.row_index = self._create_faiss_index()
        self.index.cell_index = self._create_faiss_index()
        
        # Index each table
        for table, header_tree in tables:
            table_id = str(uuid.uuid4())
            self._index_table(table, header_tree, table_id)
        
        elapsed_time = time.time() - start_time
        logger.info(
            f"Built hierarchical index in {elapsed_time:.2f}s: "
            f"{len(self.index.metadata)} items indexed"
        )
        
        return self.index

    def _create_faiss_index(self) -> Any:
        """
        Create FAISS index based on configured type.
        
        Returns:
            FAISS index object
        """
        if self.index_type == "Flat":
            return faiss.IndexFlatL2(self.embedding_dim)
        elif self.index_type == "IVF":
            # IVF index requires training
            quantizer = faiss.IndexFlatL2(self.embedding_dim)
            nlist = 100  # Number of clusters
            index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, nlist)
            return index
        elif self.index_type == "HNSW":
            # HNSW parameters
            M = 32  # Number of connections
            index = faiss.IndexHNSWFlat(self.embedding_dim, M)
            return index
        else:
            logger.warning(f"Unknown index type {self.index_type}, using Flat")
            return faiss.IndexFlatL2(self.embedding_dim)

    def _index_table(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
        table_id: str,
    ) -> None:
        """
        Index a single table at all granularity levels.
        
        Args:
            table: NormalizedTable to index
            header_tree: HeaderTree with hierarchy
            table_id: Unique identifier for this table
        """
        # 1. Table-level indexing
        table_metadata = self._index_table_level(table, header_tree, table_id)
        
        # 2. Subtable-level indexing
        subtable_metadatas = self._index_subtable_level(
            table, header_tree, table_id, table_metadata.item_id
        )
        
        # 3. Row-level indexing
        row_metadatas = self._index_row_level(
            table, header_tree, table_id, table_metadata.item_id
        )
        
        # 4. Cell-level indexing
        cell_metadatas = self._index_cell_level(
            table, header_tree, table_id, subtable_metadatas, row_metadatas
        )
        
        # Update parent-child relationships
        table_metadata.child_ids = [m.item_id for m in subtable_metadatas]
        for subtable_meta in subtable_metadatas:
            subtable_meta.parent_ids = [table_metadata.item_id]

    def _index_table_level(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
        table_id: str,
    ) -> IndexMetadata:
        """Index entire table."""
        # Generate text representation
        text = self.tree_encoder.encode_for_embedding(header_tree)
        
        # Add table content summary
        text += f" Table with {table.num_rows} rows and {table.num_cols} columns."
        
        # Create embedding
        embedding = self.embedding_model.encode([text])[0]
        
        # Create metadata
        metadata = IndexMetadata(
            item_id=str(uuid.uuid4()),
            source_table_id=table_id,
            granularity="table",
            hierarchy_path="Table",
            coordinates=((0, table.num_rows), (0, table.num_cols)),
            text=text,
        )
        
        # Add to index
        self.index.table_index.add(np.array([embedding]).astype("float32"))
        self.index.metadata.append(metadata)
        
        return metadata

    def _index_subtable_level(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
        table_id: str,
        parent_id: str,
    ) -> List[IndexMetadata]:
        """Index logical sections (by top-level headers)."""
        metadatas: List[IndexMetadata] = []
        embeddings_list: List[np.ndarray] = []
        
        # Get top-level children (first level after root)
        top_level_nodes = header_tree.get_nodes_at_level(1)
        
        if not top_level_nodes:
            # Fallback: treat entire table as one subtable
            text = self.tree_encoder.encode_for_embedding(header_tree)
            embedding = self.embedding_model.encode([text])[0]
            
            metadata = IndexMetadata(
                item_id=str(uuid.uuid4()),
                source_table_id=table_id,
                granularity="subtable",
                hierarchy_path="Table > Section",
                coordinates=((0, table.num_rows), (0, table.num_cols)),
                parent_ids=[parent_id],
                text=text,
            )
            
            embeddings_list.append(embedding)
            metadatas.append(metadata)
        else:
            for node in top_level_nodes:
                # Generate text for this subtable section
                path = " > ".join(node.get_path())
                text = f"Section: {path}. "
                
                # Add section content
                row_start = node.row_end
                row_end = table.num_rows
                col_start = node.col_start
                col_end = node.col_end
                
                # Extract cells in this section
                section_texts = []
                for row_idx in range(row_start, row_end):
                    for col_idx in range(col_start, col_end):
                        cell = table.get_cell(row_idx, col_idx)
                        if cell and cell.content.strip():
                            section_texts.append(cell.content)
                
                text += " ".join(section_texts[:50])  # Limit length
                
                # Create embedding
                embedding = self.embedding_model.encode([text])[0]
                
                metadata = IndexMetadata(
                    item_id=str(uuid.uuid4()),
                    source_table_id=table_id,
                    granularity="subtable",
                    hierarchy_path=path,
                    coordinates=((row_start, row_end), (col_start, col_end)),
                    parent_ids=[parent_id],
                    text=text,
                )
                
                embeddings_list.append(embedding)
                metadatas.append(metadata)
        
        # Add all to index
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype("float32")
            self.index.subtable_index.add(embeddings_array)
            self.index.metadata.extend(metadatas)
        
        return metadatas

    def _index_row_level(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
        table_id: str,
        parent_id: str,
    ) -> List[IndexMetadata]:
        """Index individual rows."""
        metadatas: List[IndexMetadata] = []
        embeddings_list: List[np.ndarray] = []
        
        # Find data start row (after headers)
        data_start_row = header_tree.root.row_end
        
        for row_idx in range(data_start_row, table.num_rows):
            # Extract row content
            row_cells = []
            for col_idx in range(table.num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell and cell.content.strip():
                    row_cells.append(cell.content)
            
            if not row_cells:
                continue
            
            # Generate text representation
            row_text = " ".join(row_cells)
            
            # Add header context
            header_paths = []
            for col_idx in range(table.num_cols):
                # Find header for this column
                leaf_headers = header_tree.get_all_leaves()
                for header in leaf_headers:
                    if header.col_start <= col_idx < header.col_end:
                        header_paths.append(" > ".join(header.get_path()))
                        break
            
            if header_paths:
                context = " | ".join(set(header_paths))
                text = f"Row context: {context}. Data: {row_text}"
            else:
                text = f"Row: {row_text}"
            
            # Create embedding
            embedding = self.embedding_model.encode([text])[0]
            
            metadata = IndexMetadata(
                item_id=str(uuid.uuid4()),
                source_table_id=table_id,
                granularity="row",
                hierarchy_path=f"Table > Row {row_idx}",
                coordinates=((row_idx, row_idx + 1), (0, table.num_cols)),
                parent_ids=[parent_id],
                text=text,
            )
            
            embeddings_list.append(embedding)
            metadatas.append(metadata)
        
        # Add all to index
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype("float32")
            self.index.row_index.add(embeddings_array)
            self.index.metadata.extend(metadatas)
        
        return metadatas

    def _index_cell_level(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
        table_id: str,
        subtable_metadatas: List[IndexMetadata],
        row_metadatas: List[IndexMetadata],
    ) -> List[IndexMetadata]:
        """Index individual cells with context."""
        metadatas: List[IndexMetadata] = []
        embeddings_list: List[np.ndarray] = []
        
        # Find data start row
        data_start_row = header_tree.root.row_end
        
        # Build mapping from coordinates to subtable/row IDs
        coord_to_subtable = {}
        for meta in subtable_metadatas:
            (row_start, row_end), (col_start, col_end) = meta.coordinates
            for row in range(row_start, row_end):
                for col in range(col_start, col_end):
                    coord_to_subtable[(row, col)] = meta.item_id
        
        coord_to_row = {}
        for meta in row_metadatas:
            (row_start, row_end), _ = meta.coordinates
            for row in range(row_start, row_end):
                for col in range(table.num_cols):
                    coord_to_row[(row, col)] = meta.item_id
        
        for row_idx in range(data_start_row, table.num_rows):
            for col_idx in range(table.num_cols):
                cell = table.get_cell(row_idx, col_idx)
                if cell is None or not cell.content.strip():
                    continue
                
                # Skip merged cell references
                if cell.is_merged and cell.merge_root != (row_idx, col_idx):
                    continue
                
                # Generate text with context
                cell_text = cell.content
                
                # Add header context
                leaf_headers = header_tree.get_all_leaves()
                header_path = None
                for header in leaf_headers:
                    if header.col_start <= col_idx < header.col_end:
                        header_path = " > ".join(header.get_path())
                        break
                
                if header_path:
                    text = f"Header: {header_path}. Cell: {cell_text}"
                else:
                    text = f"Cell: {cell_text}"
                
                # Create embedding
                embedding = self.embedding_model.encode([text])[0]
                
                # Find parent IDs
                parent_ids = []
                if (row_idx, col_idx) in coord_to_subtable:
                    parent_ids.append(coord_to_subtable[(row_idx, col_idx)])
                if (row_idx, col_idx) in coord_to_row:
                    parent_ids.append(coord_to_row[(row_idx, col_idx)])
                
                metadata = IndexMetadata(
                    item_id=str(uuid.uuid4()),
                    source_table_id=table_id,
                    granularity="cell",
                    hierarchy_path=header_path or f"Table > Cell ({row_idx}, {col_idx})",
                    coordinates=((row_idx, row_idx + 1), (col_idx, col_idx + 1)),
                    parent_ids=parent_ids,
                    text=text,
                )
                
                embeddings_list.append(embedding)
                metadatas.append(metadata)
        
        # Add all to index
        if embeddings_list:
            embeddings_array = np.array(embeddings_list).astype("float32")
            self.index.cell_index.add(embeddings_array)
            self.index.metadata.extend(metadatas)
        
        return metadatas

    def add_table(
        self,
        table: NormalizedTable,
        header_tree: HeaderTree,
    ) -> None:
        """
        Incrementally add new table to existing index.
        
        Args:
            table: NormalizedTable to add
            header_tree: HeaderTree with hierarchy
        """
        if self.index is None:
            # Initialize index if not exists
            self.index = HierarchicalIndex(
                embedding_dim=self.embedding_dim,
                index_type=self.index_type,
            )
            self.index.table_index = self._create_faiss_index()
            self.index.subtable_index = self._create_faiss_index()
            self.index.row_index = self._create_faiss_index()
            self.index.cell_index = self._create_faiss_index()
        
        table_id = str(uuid.uuid4())
        self._index_table(table, header_tree, table_id)
        
        logger.info(f"Added table {table_id} to index")

    def save_index(self, path: str) -> None:
        """
        Serialize FAISS indexes and metadata to disk.
        
        Args:
            path: Path to save index directory
        """
        if self.index is None:
            raise ValueError("No index to save. Build index first.")
        
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving index to {path}")
        
        # Save FAISS indexes
        faiss.write_index(
            self.index.table_index,
            str(path_obj / "table_index.faiss"),
        )
        faiss.write_index(
            self.index.subtable_index,
            str(path_obj / "subtable_index.faiss"),
        )
        faiss.write_index(
            self.index.row_index,
            str(path_obj / "row_index.faiss"),
        )
        faiss.write_index(
            self.index.cell_index,
            str(path_obj / "cell_index.faiss"),
        )
        
        # Save metadata
        metadata_dicts = [meta.to_dict() for meta in self.index.metadata]
        with open(path_obj / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata_dicts, f, indent=2, ensure_ascii=False)
        
        # Save index configuration
        config = {
            "embedding_dim": self.index.embedding_dim,
            "index_type": self.index.index_type,
            "embedding_model": self.embedding_model_name,
            "chunking_strategy": self.chunking_strategy,
            "max_chunk_size": self.max_chunk_size,
        }
        with open(path_obj / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved index to {path}")

    def load_index(self, path: str) -> HierarchicalIndex:
        """
        Load FAISS indexes and metadata from disk.
        
        Args:
            path: Path to index directory
            
        Returns:
            Loaded HierarchicalIndex
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Index directory not found: {path}")
        
        logger.info(f"Loading index from {path}")
        
        # Load configuration
        with open(path_obj / "config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        
        self.embedding_dim = config["embedding_dim"]
        self.index_type = config["index_type"]
        
        # Load FAISS indexes
        self.index = HierarchicalIndex(
            embedding_dim=self.embedding_dim,
            index_type=self.index_type,
        )
        
        self.index.table_index = faiss.read_index(
            str(path_obj / "table_index.faiss")
        )
        self.index.subtable_index = faiss.read_index(
            str(path_obj / "subtable_index.faiss")
        )
        self.index.row_index = faiss.read_index(
            str(path_obj / "row_index.faiss")
        )
        self.index.cell_index = faiss.read_index(
            str(path_obj / "cell_index.faiss")
        )
        
        # Load metadata
        with open(path_obj / "metadata.json", "r", encoding="utf-8") as f:
            metadata_dicts = json.load(f)
        
        self.index.metadata = [
            IndexMetadata.from_dict(meta_dict) for meta_dict in metadata_dicts
        ]
        
        logger.info(
            f"Loaded index: {len(self.index.metadata)} items, "
            f"{self.index.table_index.ntotal} table-level items"
        )
        
        return self.index
