sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ocr_normalizer import RobustNormalizer
from src.ocr_tsr.chunk_loader import load_scitsr_chunk
from src.ocr_tsr.simple_pipeline import SimpleSpatialTSR
from src.ocr_tsr.table_transformer import TableTransformerTSR
from src.ocr_tsr.olmocr_wrapper import OlmOCROCR
from src.models.baselines.wrapper import TableCenterNetWrapper

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
