"""
Dataset Loader for Table Parsing Benchmarks
PubTabNet, FinTabNet, SciTSR 등 공개 데이터셋 로더
"""
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
from PIL import Image
import random
from tqdm import tqdm

from models.base_model import BoundingBox, TableStructure, Cell


class DatasetSample:
    """데이터셋 샘플 데이터 클래스"""
    
    def __init__(
        self,
        sample_id: str,
        image_path: str,
        ground_truth: Dict[str, Any],
        table_type: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        self.sample_id = sample_id
        self.image_path = image_path
        self.ground_truth = ground_truth
        self.table_type = table_type
        self.metadata = metadata or {}
    
    def load_image(self) -> Image.Image:
        """이미지 로드"""
        return Image.open(self.image_path).convert('RGB')
    
    def to_dict(self) -> Dict:
        return {
            'sample_id': self.sample_id,
            'image_path': self.image_path,
            'ground_truth': self.ground_truth,
            'table_type': self.table_type,
            'metadata': self.metadata
        }


class PubTabNetLoader:
    """
    PubTabNet 데이터셋 로더
    
    PubTabNet은 약 50만 개의 표 이미지를 포함하는 대규모 데이터셋
    - 이미지: PMC 논문에서 추출한 표
    - 주석: HTML 형식의 표 구조
    """
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: PubTabNet 데이터 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.val_dir = self.data_dir / "val"
        self.test_dir = self.data_dir / "test"
        
        # 주석 파일
        self.train_json = self.data_dir / "PubTabNet_2.0.0.jsonl"
        self.samples = []
        
    def load_samples(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None,
        filter_by_type: Optional[List[str]] = None
    ) -> List[DatasetSample]:
        """
        샘플 로드
        
        Args:
            split: 'train', 'val', or 'test'
            max_samples: 최대 샘플 수 (None이면 전체)
            filter_by_type: 표 유형 필터 (None이면 전체)
            
        Returns:
            List of DatasetSample
        """
        print(f"Loading PubTabNet {split} split...")
        
        # 분할에 따른 디렉토리
        if split == 'train':
            img_dir = self.train_dir
        elif split == 'val':
            img_dir = self.val_dir
        else:
            img_dir = self.test_dir
        
        samples = []
        
        # JSONL 파일이 존재하면 읽기
        if self.train_json.exists():
            with open(self.train_json, 'r', encoding='utf-8') as f:
                for idx, line in enumerate(f):
                    if max_samples and idx >= max_samples:
                        break
                    
                    data = json.loads(line)
                    
                    # 이미지 경로
                    filename = data.get('filename', '')
                    img_path = img_dir / filename
                    
                    # 이미지 파일이 없으면 스킵
                    if not img_path.exists():
                        continue
                    
                    # Ground truth 추출
                    html = data.get('html', {}).get('structure', {}).get('tokens', [])
                    html_str = ' '.join(html) if isinstance(html, list) else str(html)
                    
                    ground_truth = {
                        'html': html_str,
                        'split': split,
                        'tables': [{
                            'bbox': {
                                'x1': 0,
                                'y1': 0,
                                'x2': data.get('width', 1000),
                                'y2': data.get('height', 1000),
                                'confidence': 1.0
                            }
                        }]
                    }
                    
                    sample = DatasetSample(
                        sample_id=f"pubtabnet_{split}_{idx}",
                        image_path=str(img_path),
                        ground_truth=ground_truth,
                        metadata={'source': 'PubTabNet', 'split': split}
                    )
                    
                    samples.append(sample)
        else:
            # JSONL이 없으면 이미지 파일만 스캔
            print(f"Warning: {self.train_json} not found. Loading images without annotations...")
            if img_dir.exists():
                image_files = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.png'))
                
                for idx, img_path in enumerate(image_files[:max_samples] if max_samples else image_files):
                    sample = DatasetSample(
                        sample_id=f"pubtabnet_{split}_{idx}",
                        image_path=str(img_path),
                        ground_truth={'tables': []},
                        metadata={'source': 'PubTabNet', 'split': split, 'no_annotations': True}
                    )
                    samples.append(sample)
        
        print(f"Loaded {len(samples)} samples from PubTabNet {split}")
        return samples
    
    def create_sample_dataset(self, output_dir: str, num_samples: int = 100):
        """
        테스트용 샘플 데이터셋 생성
        
        Args:
            output_dir: 출력 디렉토리
            num_samples: 샘플 수
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        samples = self.load_samples(split='train', max_samples=num_samples)
        
        # 샘플 저장
        samples_json = [s.to_dict() for s in samples]
        with open(output_path / 'pubtabnet_sample.json', 'w', encoding='utf-8') as f:
            json.dump(samples_json, f, indent=2, ensure_ascii=False)
        
        print(f"Sample dataset saved to {output_path}")


class SyntheticDatasetLoader:
    """
    합성 표 데이터셋 로더
    테스트 및 개발용으로 간단한 표 이미지 생성
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_samples(
        self,
        split: str = 'train',
        max_samples: Optional[int] = None,
        filter_by_type: Optional[List[str]] = None
    ) -> List[DatasetSample]:
        """load_samples interface for consistency with other loaders"""
        return self.generate_samples(num_samples=max_samples or 10, table_types=filter_by_type)

    def generate_samples(
        self,
        num_samples: int = 100,
        table_types: Optional[List[str]] = None
    ) -> List[DatasetSample]:
        """
        합성 표 샘플 생성
        
        Args:
            num_samples: 생성할 샘플 수
            table_types: 생성할 표 유형 리스트
            
        Returns:
            List of DatasetSample
        """
        from PIL import ImageDraw, ImageFont
        
        if table_types is None:
            table_types = ['fully_bordered', 'unbordered', 'partially_bordered']
        
        samples = []
        
        for i in tqdm(range(num_samples), desc="Generating synthetic tables"):
            # 표 유형 선택
            table_type = random.choice(table_types)
            
            # 표 생성
            img, ground_truth = self._generate_table(i, table_type)
            
            # 이미지 저장
            img_path = self.output_dir / f"synthetic_{table_type}_{i}.png"
            img.save(img_path)
            
            sample = DatasetSample(
                sample_id=f"synthetic_{i}",
                image_path=str(img_path),
                ground_truth=ground_truth,
                table_type=table_type,
                metadata={'source': 'synthetic', 'table_type': table_type}
            )
            
            samples.append(sample)
        
        print(f"Generated {len(samples)} synthetic samples")
        return samples
    
    def _generate_table(
        self,
        idx: int,
        table_type: str,
        width: int = 800,
        height: int = 400
    ) -> Tuple[Image.Image, Dict]:
        """단일 표 생성"""
        from PIL import ImageDraw, ImageFont
        
        img = Image.new('RGB', (width, height), 'white')
        draw = ImageDraw.Draw(img)
        
        # 랜덤 행/열 수
        rows = random.randint(3, 6)
        cols = random.randint(3, 5)
        
        cell_width = width // cols
        cell_height = height // rows
        
        # 표 유형에 따라 테두리 그리기
        if table_type == 'fully_bordered':
            # 모든 셀 경계 그리기
            for i in range(rows + 1):
                y = i * cell_height
                draw.line([(0, y), (width, y)], fill='black', width=2)
            for j in range(cols + 1):
                x = j * cell_width
                draw.line([(x, 0), (x, height)], fill='black', width=2)
        
        elif table_type == 'unbordered':
            # 테두리 없음 (텍스트만)
            pass
        
        elif table_type == 'partially_bordered':
            # 외곽 테두리만
            draw.rectangle([(0, 0), (width-1, height-1)], outline='black', width=2)
            # 헤더 구분선
            draw.line([(0, cell_height), (width, cell_height)], fill='black', width=2)
        
        # 텍스트 추가
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        cells = []
        for i in range(rows):
            for j in range(cols):
                x = j * cell_width + 10
                y = i * cell_height + 10
                text = f"R{i}C{j}"
                draw.text((x, y), text, fill='black', font=font)
                
                cells.append({
                    'row': i,
                    'col': j,
                    'row_span': 1,
                    'col_span': 1,
                    'text': text,
                    'is_header': (i == 0),
                    'bbox': {
                        'x1': j * cell_width,
                        'y1': i * cell_height,
                        'x2': (j + 1) * cell_width,
                        'y2': (i + 1) * cell_height,
                        'confidence': 1.0
                    }
                })
        
        ground_truth = {
            'cells': cells,
            'html': _cells_to_html(cells),
            'tables': [{
                'bbox': {'x1': 0, 'y1': 0, 'x2': width, 'y2': height, 'confidence': 1.0},
                'num_rows': rows,
                'num_cols': cols,
                'cells': cells
            }]
        }
        
        return img, ground_truth



class SciTSRLoader:
    """
    SciTSR 데이터셋 로더
    """
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        self.test_dir = self.data_dir / "test" # or test/img
        
    def load_samples(
        self,
        split: str = 'test',
        max_samples: Optional[int] = None,
        filter_by_type: Optional[List[str]] = None
    ) -> List[DatasetSample]:
        
        print(f"Loading SciTSR {split} split...")
        if split == 'train':
            base_dir = self.train_dir
        else:
            base_dir = self.test_dir
            
        img_dir = base_dir / 'img'
        struct_dir = base_dir / 'structure'
        
        if not img_dir.exists():
            print(f"Error: {img_dir} does not exist.")
            return []
            
        samples = []
        # 이미지 파일 목록
        image_files = sorted(list(img_dir.glob('*.png')))
        
        if max_samples:
            image_files = image_files[:max_samples]
            
        for img_path in tqdm(image_files, desc=f"Loading SciTSR {split}"):
            file_id = img_path.stem
            # Load chunks for Bbox reconstruction
            chunks = []
            chunk_path = base_dir / 'chunk' / f"{file_id}.chunk"
            if chunk_path.exists():
                try:
                    with open(chunk_path, 'r', encoding='utf-8') as f:
                        chunk_data = json.load(f)
                        chunks = chunk_data.get('chunks', [])
                except Exception as e:
                    print(f"Error parsing {chunk_path}: {e}")

            ground_truth = {'cells': []}
            json_path = base_dir / 'structure' / f"{file_id}.json"
            
            if json_path.exists():
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        
                    # SciTSR format to internal format
                    cells = []
                    
                    # Helper to normalize text for matching
                    def normalize(t):
                        return "".join(t.split()).lower()

                    for cell_data in data.get('cells', []):
                        start_row = cell_data.get('start_row', 0)
                        end_row = cell_data.get('end_row', 0)
                        start_col = cell_data.get('start_col', 0)
                        end_col = cell_data.get('end_col', 0)
                        content_list = cell_data.get('content', [])
                        content = " ".join(content_list)
                        
                        # Find matching chunks to get Bbox
                        cell_chunks = []
                        target_text = normalize(content)
                        
                        # Simple matching: find chunks whose text matches or is part of the cell content
                        # This is a heuristic since rel files are missing in test split
                        for ch in chunks:
                            ch_text = normalize(ch.get('text', ''))
                            if ch_text and ch_text in target_text:
                                cell_chunks.append(ch)
                        
                        bbox = None
                        if cell_chunks:
                            # Aggregate Bboxes
                            x1 = min(c['pos'][0] for c in cell_chunks)
                            y1 = min(c['pos'][1] for c in cell_chunks)
                            x2 = max(c['pos'][2] for c in cell_chunks)
                            y2 = max(c['pos'][3] for c in cell_chunks)
                            bbox = BoundingBox(x1, y1, x2, y2)
                        
                        cells.append({
                            'row': start_row,
                            'col': start_col,
                            'row_span': end_row - start_row + 1,
                            'col_span': end_col - start_col + 1,
                            'num_cells': len(cells), # Unique ID
                            'text': content,
                            'bbox': bbox
                        })
                    ground_truth['cells'] = cells
                    ground_truth['html'] = _cells_to_html(cells)
                    
                except Exception as e:
                    print(f"Error parsing {json_path}: {e}")
            
            sample = DatasetSample(
                sample_id=f"scitsr_{split}_{file_id}",
                image_path=str(img_path),
                ground_truth=ground_truth,
                metadata={'source': 'SciTSR', 'split': split}
            )
            samples.append(sample)
            
        print(f"Loaded {len(samples)} samples from SciTSR {split}")
        return samples

def _cells_to_html(cells: List[Dict]) -> str:
    """셀 리스트를 제대로 된 HTML 테이블로 변환"""
    if not cells:
        return "<table></table>"
    
    # 최대 행/열 크기 결정
    max_row = max(c['row'] + c['row_span'] - 1 for c in cells)
    max_col = max(c['col'] + c['col_span'] - 1 for c in cells)
    
    # 그리드 생성 (어떤 셀이 어느 위치를 차지하는지 매핑)
    grid = {}
    for cell in cells:
        for r in range(cell['row'], cell['row'] + cell['row_span']):
            for c in range(cell['col'], cell['col'] + cell['col_span']):
                if r not in grid:
                    grid[r] = {}
                # 시작 셀만 표시 (merged cell 처리)
                if r == cell['row'] and c == cell['col']:
                    grid[r][c] = cell
                else:
                    grid[r][c] = None  # merged part
    
    # HTML 생성
    html = "<table>"
    for r in range(max_row + 1):
        html += "<tr>"
        for c in range(max_col + 1):
            if r in grid and c in grid[r]:
                cell = grid[r][c]
                if cell is not None:  # 실제 셀 시작 위치
                    attrs = ""
                    if cell['row_span'] > 1:
                        attrs += f' rowspan="{cell["row_span"]}"'
                    if cell['col_span'] > 1:
                        attrs += f' colspan="{cell["col_span"]}"'
                    text = cell.get('text', '')
                    html += f"<td{attrs}>{text}</td>"
                # else: merged part, skip
        html += "</tr>"
    html += "</table>"
    return html



def get_dataset_loader(dataset_name: str, data_dir: str):
    """
    데이터셋 로더 팩토리
    
    Args:
        dataset_name: 'pubtabnet', 'fintabnet', 'scitsr', or 'synthetic'
        data_dir: 데이터 디렉토리
        
    Returns:
        Dataset loader instance
    """
    if dataset_name.lower() == 'pubtabnet':
        return PubTabNetLoader(data_dir)
    elif dataset_name.lower() == 'synthetic':
        return SyntheticDatasetLoader(data_dir)
    elif dataset_name.lower() == 'scitsr':
        return SciTSRLoader(data_dir)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

