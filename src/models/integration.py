"""
Integration module for E2E Hierarchical Table Model with HierTable-RAG.

Provides seamless integration between:
- E2E model output (cells, hierarchy)
- HierTable-RAG pipeline (indexing, retrieval, generation)

This enables end-to-end table understanding from images to QA.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from src.models.e2e_hierarchical import (
    E2EHierarchicalTableModel,
    E2EHierarchicalPipeline,
    E2EConfig,
    E2EOutput,
)
from src.parsing.hierarchy import HeaderTree, HeaderNode, HierarchyExtractor
from src.parsing.extractor import TableObject, Cell
from src.parsing.cell_merger import NormalizedTable, NormalizedCell
from src.encoding.graph_builder import GraphBuilder
from src.retrieval.indexer import HierarchicalIndexer
from src.retrieval.retriever import StructureAwareRetriever, ContextBundle
from src.generation.prompting import StructuredPromptBuilder, LLMReasoner, Answer

logger = logging.getLogger(__name__)


class E2ETableRAGPipeline:
    """
    End-to-End Table RAG Pipeline.
    
    Combines E2E hierarchical table understanding with RAG:
    
    Image → E2E Model → HeaderTree → Index → Retrieve → Generate
    
    This is the main integration point for the research contribution:
    - Vision-based table structure recognition
    - Automatic hierarchy extraction
    - Structure-aware retrieval
    - Hierarchy-informed generation
    """
    
    def __init__(
        self,
        e2e_model: Optional[E2EHierarchicalTableModel] = None,
        e2e_config: Optional[E2EConfig] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-4",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        """
        Initialize E2E Table RAG Pipeline.
        
        Args:
            e2e_model: Pre-trained E2E model
            e2e_config: E2E model config
            llm_provider: LLM provider for generation
            llm_model: LLM model name
            embedding_model: Embedding model for retrieval
            device: Device to run on
        """
        self.device = device
        
        # Initialize E2E pipeline
        self.e2e_pipeline = E2EHierarchicalPipeline(
            model=e2e_model,
            config=e2e_config,
            device=device,
        )
        
        # Initialize RAG components
        self.graph_builder = GraphBuilder(embedding_model=embedding_model)
        self.indexer = HierarchicalIndexer(embedding_model=embedding_model)
        self.prompt_builder = StructuredPromptBuilder()
        
        # LLM reasoner (optional, requires API key)
        try:
            self.llm_reasoner = LLMReasoner(
                provider=llm_provider,
                default_model=llm_model,
            )
        except Exception as e:
            logger.warning(f"LLM reasoner not initialized: {e}")
            self.llm_reasoner = None
        
        # Cache for processed tables
        self._table_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("E2ETableRAGPipeline initialized")
    
    def process_image(
        self,
        image_path: str,
        table_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process table image through E2E model.
        
        Args:
            image_path: Path to table image
            table_id: Optional ID for the table
            
        Returns:
            Dictionary with:
            - table_object: TableObject
            - header_tree: HeaderTree
            - normalized_table: NormalizedTable
            - graph: NetworkX graph
        """
        logger.info(f"Processing image: {image_path}")
        
        # Generate table ID if not provided
        if table_id is None:
            table_id = Path(image_path).stem
        
        # Check cache
        if table_id in self._table_cache:
            logger.info(f"Using cached result for {table_id}")
            return self._table_cache[table_id]
        
        # Run E2E model
        e2e_output = self.e2e_pipeline(image_path)
        
        # Convert to TableObject
        table_object = self.e2e_pipeline.to_table_object(e2e_output)
        
        # Get HeaderTree
        header_tree = self.e2e_pipeline.to_header_tree(e2e_output)
        
        # If E2E didn't produce header tree, extract from TableObject
        if header_tree is None and table_object is not None:
            hierarchy_extractor = HierarchyExtractor()
            try:
                header_tree = hierarchy_extractor.extract_header_tree(table_object)
            except Exception as e:
                logger.warning(f"Failed to extract hierarchy: {e}")
                # Create minimal header tree
                root = HeaderNode(
                    text="TABLE_ROOT",
                    level=0,
                    span=table_object.num_cols() if table_object else 1,
                )
                header_tree = HeaderTree(root=root)
        
        # Create NormalizedTable
        normalized_table = self._create_normalized_table(table_object)
        
        # Build graph
        graph = None
        if normalized_table is not None and header_tree is not None:
            try:
                graph = self.graph_builder.build_table_graph(
                    normalized_table, header_tree
                )
            except Exception as e:
                logger.warning(f"Failed to build graph: {e}")
        
        result = {
            "table_id": table_id,
            "table_object": table_object,
            "header_tree": header_tree,
            "normalized_table": normalized_table,
            "graph": graph,
            "e2e_output": e2e_output,
        }
        
        # Cache result
        self._table_cache[table_id] = result
        
        logger.info(f"Processed table {table_id}")
        
        return result
    
    def _create_normalized_table(
        self,
        table_object: Optional[TableObject],
    ) -> Optional[NormalizedTable]:
        """
        Create NormalizedTable from TableObject.
        
        Args:
            table_object: TableObject to convert
            
        Returns:
            NormalizedTable instance
        """
        if table_object is None or not table_object.cells:
            return None
        
        # Get dimensions
        num_rows = table_object.num_rows()
        num_cols = table_object.num_cols()
        
        # Create normalized cells
        cells: List[List[Optional[NormalizedCell]]] = [
            [None] * num_cols for _ in range(num_rows)
        ]
        
        for row_idx, row in enumerate(table_object.cells):
            for cell in row:
                if cell.row < num_rows and cell.col < num_cols:
                    normalized_cell = NormalizedCell(
                        content=cell.content,
                        row=cell.row,
                        col=cell.col,
                        is_merged=(cell.rowspan > 1 or cell.colspan > 1),
                        merge_root=(cell.row, cell.col),
                        merge_span=(cell.rowspan, cell.colspan),
                    )
                    cells[cell.row][cell.col] = normalized_cell
        
        return NormalizedTable(
            cells=cells,
            num_rows=num_rows,
            num_cols=num_cols,
        )
    
    def index_tables(
        self,
        image_paths: List[str],
        table_ids: Optional[List[str]] = None,
    ) -> None:
        """
        Process and index multiple table images.
        
        Args:
            image_paths: List of image paths
            table_ids: Optional list of table IDs
        """
        logger.info(f"Indexing {len(image_paths)} tables")
        
        if table_ids is None:
            table_ids = [Path(p).stem for p in image_paths]
        
        tables = []
        header_trees = []
        
        for image_path, table_id in zip(image_paths, table_ids):
            result = self.process_image(image_path, table_id)
            
            if result["normalized_table"] is not None:
                tables.append(result["normalized_table"])
                header_trees.append(result["header_tree"])
        
        # Build index
        if tables:
            self.indexer.build_index(tables, header_trees, table_ids)
            logger.info(f"Indexed {len(tables)} tables")
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        strategy: str = "adaptive",
    ) -> ContextBundle:
        """
        Query indexed tables.
        
        Args:
            question: Natural language question
            top_k: Number of results to retrieve
            strategy: Retrieval strategy
            
        Returns:
            ContextBundle with retrieved cells and hierarchy
        """
        logger.info(f"Querying: {question}")
        
        # Create retriever
        retriever = StructureAwareRetriever(
            indexer=self.indexer,
            graph_builder=self.graph_builder,
        )
        
        # Retrieve with hierarchy
        context = retriever.retrieve_with_hierarchy(
            query=question,
            top_k=top_k,
        )
        
        return context
    
    def answer(
        self,
        question: str,
        top_k: int = 5,
        strategy: str = "adaptive",
        prompt_style: str = "explicit",
    ) -> Answer:
        """
        Answer question about indexed tables.
        
        Args:
            question: Natural language question
            top_k: Number of results to retrieve
            strategy: Retrieval strategy
            prompt_style: Prompt formatting style
            
        Returns:
            Answer with response and citations
        """
        logger.info(f"Answering: {question}")
        
        # Retrieve context
        context = self.query(question, top_k=top_k, strategy=strategy)
        
        # Build prompt
        prompt = self.prompt_builder.build_rag_prompt(
            query=question,
            context=context,
            style=prompt_style,
        )
        
        # Generate answer
        if self.llm_reasoner is not None:
            answer = self.llm_reasoner.generate_answer(prompt)
        else:
            # Return prompt as answer if no LLM
            answer = Answer(
                answer_text=f"[LLM not configured]\n\nPrompt:\n{prompt[:500]}...",
                model="none",
            )
        
        return answer
    
    def process_and_answer(
        self,
        image_path: str,
        question: str,
        prompt_style: str = "explicit",
    ) -> Dict[str, Any]:
        """
        Process single image and answer question (no indexing).
        
        Useful for one-off queries without building an index.
        
        Args:
            image_path: Path to table image
            question: Question about the table
            prompt_style: Prompt formatting style
            
        Returns:
            Dictionary with table info and answer
        """
        # Process image
        result = self.process_image(image_path)
        
        # Build context directly from result
        from src.retrieval.retriever import RetrievalResult
        from src.retrieval.indexer import IndexMetadata
        
        # Create simple context from all cells
        retrieved_cells = []
        
        if result["table_object"] is not None:
            for row_idx, row in enumerate(result["table_object"].cells):
                for cell in row:
                    # Build hierarchy path
                    hierarchy_path = self._get_hierarchy_path(
                        result["header_tree"],
                        cell.row,
                        cell.col,
                    )
                    
                    metadata = IndexMetadata(
                        item_id=f"cell_{cell.row}_{cell.col}",
                        granularity="cell",
                        source_table_id=result["table_id"],
                        hierarchy_path=hierarchy_path,
                        text=cell.content,
                        coordinates=((cell.row, cell.row + cell.rowspan),
                                   (cell.col, cell.col + cell.colspan)),
                    )
                    
                    retrieved_cells.append(RetrievalResult(
                        item_id=metadata.item_id,
                        metadata=metadata,
                        score=1.0,
                        granularity="cell",
                        content=cell.content,
                        hierarchy_path=hierarchy_path,
                    ))
        
        # Build context bundle
        context = ContextBundle(
            retrieved_cells=retrieved_cells[:10],  # Limit cells
            hierarchy_context=result["header_tree"].root.to_dict() if result["header_tree"] else {},
            table_metadata={"table_id": result["table_id"]},
            query=question,
        )
        
        # Build prompt and generate answer
        prompt = self.prompt_builder.build_rag_prompt(
            query=question,
            context=context,
            style=prompt_style,
        )
        
        if self.llm_reasoner is not None:
            answer = self.llm_reasoner.generate_answer(prompt)
        else:
            answer = Answer(
                answer_text=f"[LLM not configured]\n\nContext retrieved with {len(retrieved_cells)} cells.",
                model="none",
            )
        
        return {
            "table_info": result,
            "context": context,
            "answer": answer,
            "prompt": prompt,
        }
    
    def _get_hierarchy_path(
        self,
        header_tree: Optional[HeaderTree],
        row: int,
        col: int,
    ) -> str:
        """
        Get hierarchy path for a cell position.
        
        Args:
            header_tree: HeaderTree
            row: Row index
            col: Column index
            
        Returns:
            Hierarchy path string (e.g., "2023 > Q1 > Revenue")
        """
        if header_tree is None:
            return f"Row {row}, Col {col}"
        
        # Find headers that cover this column
        path_parts = []
        
        def find_path(node: HeaderNode, target_col: int, path: List[str]) -> List[str]:
            """Recursively find path to column."""
            if node.col_start <= target_col < node.col_end:
                if node.text != "TABLE_ROOT":
                    path.append(node.text)
                
                for child in node.children:
                    result = find_path(child, target_col, path.copy())
                    if result:
                        return result
                
                return path
            return []
        
        path_parts = find_path(header_tree.root, col, [])
        
        if path_parts:
            return " > ".join(path_parts)
        else:
            return f"Column {col}"
    
    def export_results(
        self,
        output_path: str,
        include_graphs: bool = True,
    ) -> None:
        """
        Export processed tables and results.
        
        Args:
            output_path: Output directory
            include_graphs: Whether to export graph files
        """
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for table_id, result in self._table_cache.items():
            table_dir = output_dir / table_id
            table_dir.mkdir(exist_ok=True)
            
            # Export header tree
            if result["header_tree"] is not None:
                tree_path = table_dir / "header_tree.json"
                with open(tree_path, "w", encoding="utf-8") as f:
                    import json
                    json.dump(
                        result["header_tree"].root.to_dict(),
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            
            # Export graph
            if include_graphs and result["graph"] is not None:
                graph_path = table_dir / "graph.json"
                self.graph_builder.export_for_visualization(
                    result["graph"],
                    str(graph_path),
                    format="json",
                )
        
        logger.info(f"Exported results to {output_path}")


def create_e2e_rag_pipeline(
    model_path: Optional[str] = None,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4",
    device: str = "cpu",
) -> E2ETableRAGPipeline:
    """
    Factory function to create E2E Table RAG Pipeline.
    
    Args:
        model_path: Path to pre-trained E2E model
        llm_provider: LLM provider
        llm_model: LLM model name
        device: Device to run on
        
    Returns:
        Configured E2ETableRAGPipeline instance
    """
    # Load model if path provided
    e2e_model = None
    if model_path is not None:
        try:
            e2e_model = E2EHierarchicalTableModel.from_pretrained(model_path)
            logger.info(f"Loaded E2E model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model: {e}, using default")
    
    return E2ETableRAGPipeline(
        e2e_model=e2e_model,
        llm_provider=llm_provider,
        llm_model=llm_model,
        device=device,
    )

