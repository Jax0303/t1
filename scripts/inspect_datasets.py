import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inspect_doclaynet():
    logger.info("Inspecting DocLayNet (docling-project)...")
    try:
        # Load streaming to avoid downloading everything
        dataset = load_dataset("docling-project/DocLayNet", split="validation", streaming=True)
        sample = next(iter(dataset))
        logger.info(f"DocLayNet Keys: {sample.keys()}")
        # Check for 'objects' or similar structure
        if 'objects' in sample:
            logger.info(f"DocLayNet Objects: {sample['objects']}")
    except Exception as e:
        logger.error(f"Error inspecting DocLayNet: {e}")

def inspect_tablebank():
    logger.info("Inspecting TableBank (liminghao1630)...")
    try:
        # TableBank might have multiple configs.
        dataset = load_dataset("liminghao1630/TableBank", split="validation", streaming=True)
        sample = next(iter(dataset))
        logger.info(f"TableBank Keys: {sample.keys()}")
    except Exception as e:
        logger.error(f"Error inspecting TableBank: {e}")
        # Try specific config if generic fails
        try:
            logger.info("Retrying TableBank with 'Word' config...")
            dataset = load_dataset("liminghao1630/TableBank", "Word", split="validation", streaming=True)
            sample = next(iter(dataset))
            logger.info(f"TableBank (Word) Keys: {sample.keys()}")
        except Exception as e2:
            logger.error(f"Error inspecting TableBank (Word): {e2}")

if __name__ == "__main__":
    inspect_doclaynet()
    inspect_tablebank()
