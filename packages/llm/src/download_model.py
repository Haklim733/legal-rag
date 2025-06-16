import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union

from huggingface_hub import snapshot_download, hf_hub_download
from sentence_transformers import SentenceTransformer
import torch  # Moved to top level since it's used in type hints


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    # Disable verbose logging from libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.INFO)


def download_model(
    model_name: str,
    output_dir: Optional[Union[str, Path]] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Path:
    """Download a sentence transformer model using huggingface_hub.
    
    Args:
        model_name: Name of the model to download (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        output_dir: Directory to save the model (default: MODEL_DIR/model_name)
        device: Device to load the model on ('cuda' or 'cpu')
        
    Returns:
        Path to the downloaded model
    """
    logger = logging.getLogger(__name__)
    
    # Set up output directory
    if output_dir is None:
        base_dir = Path(os.getenv("MODEL_DIR", "models"))
        output_dir = base_dir / model_name.replace("/", "--")
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Downloading model '{model_name}' to {output_dir}")
    
    try:
        # Use huggingface_hub to download the model files
        model_path = snapshot_download(
            repo_id=model_name,
            local_dir=output_dir,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.bin", "*.h5", "*.ot", "*.msgpack"],
        )
        
        # Verify the model can be loaded
        _ = SentenceTransformer(model_path, device=device)
        
        logger.info(f"Successfully downloaded and verified model at {output_dir}")
        return output_dir
        
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        raise


def main():
    """Main function to handle command line arguments."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    parser = argparse.ArgumentParser(description="Download Hugging Face models for LightRAG")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model name to download (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for the model (default: MODEL_DIR/model_name)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on ('cuda' or 'cpu')"
    )
    
    args = parser.parse_args()
    
    try:
        download_model(
            model_name=args.model,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to download model: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
