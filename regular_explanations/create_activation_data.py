import json
import logging
from logging.handlers import RotatingFileHandler
import os
import tempfile
import time
from typing import Dict
from dotenv import load_dotenv
import einops
import torch
from tqdm import tqdm
from transformer_lens import HookedTransformer
from sae_lens import SAE
import pandas as pd
from datasets import Dataset, Features, Sequence, Value, load_dataset, concatenate_datasets
from transformer_lens.utils import tokenize_and_concatenate

import os
from enum import Enum
from typing import Iterable
import json

load_dotenv()


class Datasets(Enum):
    SAE_2_6_10_V3 = "dreuter/apps-activations-introductory-layers-2-6-10"
    SAE_2_6_10_V1 = "dreuter/apps-activations-introductory-layers-2-6-10"

REPO_NAME = Datasets.SAE_2_6_10_V3

torch.cuda.empty_cache()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Create logs directory if it doesn't exist
os.makedirs('logs', exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create a file handler
file_handler = RotatingFileHandler('logs/activation_data.log', maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Add the file handler to the logger
logger.addHandler(file_handler)

def generate_activation_data(
  model: HookedTransformer,
  dataset,
  sae_refs: Dict[int, str] = {
    2: "layer_2/width_16k/average_l0_53",
    6: "layer_6/width_16k/average_l0_70",
    10: "layer_10/width_16k/average_l0_77"
  },
  max_seq_len: int = 512, 
  chunk_size: int = 50,
  max_rows: int = 1500
):
    sae_dict: Dict[str, SAE] = {}
    for layer, sae_id in sae_refs.items():
        logger.info(f"Loading SAE for layer {layer}...")
        sae, _, _ = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id=sae_id,
            device="cuda",
        )
        sae_dict[layer] = sae
        
    counter = 0    
    chunk = []
    
    for item in dataset:
        
        counter += 1 
        if counter > max_rows:
            break

        problem_id = int(item['problem_id'])
        solutions = json.loads(item['solutions'])
        
        logger.info(f"Processing problem {problem_id} with {len(solutions)} solutions...")
        
        for solution_id, solution in enumerate(solutions):
            
            tokens = model.tokenizer(solution, return_tensors="pt", padding=True, truncation=True, max_length=max_seq_len)["input_ids"].flatten()
            tokens = tokens.unsqueeze(0).to('cuda')  # Move tokens to GPU
            
            with torch.no_grad():
                names_filter = [sae.cfg.hook_name for sae in sae_dict.values()]
                stop_at_layer = sae_dict[10].cfg.hook_layer + 1
                _, cache = model.run_with_cache(tokens, stop_at_layer=stop_at_layer, names_filter=names_filter)
            
            token_strs = [model.tokenizer.decode(token.item()) for token in tokens[0]]
            activations = {layer: [] for layer in sae_dict.keys()}
            
            # Process activations for each SAE
            for layer, sae in sae_dict.items():
                logger.info(f"\tProcessing SAE layer {layer}...")
                sae_in = cache[sae.cfg.hook_name]
                feature_acts = sae.encode(sae_in).squeeze()
                layer_activations = (feature_acts > 0).cpu().bool().tolist()  # Move to CPU before converting to list
                activations[layer] = layer_activations
                
                # Free memory
                del sae_in, feature_acts
            
            del cache
            
            result = {
                "problem_id": problem_id,
                "solution_id": solution_id,
                "token_str": token_strs,
                "activations_2": activations[2],
                "activations_6": activations[6],
                "activations_10": activations[10],
            }
            
            chunk.append(result)

            if len(chunk) >= chunk_size:
                logger.info(f"\t\tYielding chunk...")
                yield chunk
                chunk = []
                torch.cuda.empty_cache()
                
    if chunk:  
        yield chunk

def main():
    logger.info("Loading model...")
    model = HookedTransformer.from_pretrained("gemma-2-2b")
    logger.info("Loading dataset...")
    dataset = load_dataset("codeparrot/apps", split="train")
    dataset = dataset.filter(lambda x: x['difficulty'] == 'interview')
    logger.info(f"Loaded {len(dataset)} samples")
    assert len(dataset) > 0
    
    features = Features({
        "problem_id": Value("int32"),
        "solution_id": Value("int32"),
        "token_str": Sequence(Value("string")),
        "activations_2": Sequence(Sequence(Value("bool"))),
        "activations_6": Sequence(Sequence(Value("bool"))),
        "activations_10": Sequence(Sequence(Value("bool"))),
    })

    # Create a temporary directory to store chunks
    with tempfile.TemporaryDirectory() as temp_dir:
        chunk_paths = []
        for i, chunk in enumerate(generate_activation_data(model, dataset)):
            chunk_dataset = Dataset.from_list(chunk, features=features)
            chunk_path = os.path.join(temp_dir, f"chunk_{i}.arrow")
            chunk_dataset.save_to_disk(chunk_path)
            chunk_paths.append(chunk_path)
            del chunk_dataset  # Free up memory
            torch.cuda.empty_cache()
            
        logger.info("Finished processing all chunks")

        # Free up memory
        del model, dataset
        torch.cuda.empty_cache()
        
        logger.info("Combining chunks...")

        # Combine all chunks
        combined_dataset = Dataset.load_from_disk(chunk_paths[0])
        for path in chunk_paths[1:]:
            logger.info(f"Loading chunk from {path}")
            combined_dataset = concatenate_datasets([combined_dataset, Dataset.load_from_disk(path)])


        # Push to hub
        max_retries = 20
        retry_delay = 10
        for attempt in range(max_retries):
            logger.info(f"Pushing dataset to hub (attempt {attempt + 1}/{max_retries})...")
            try:
                combined_dataset.push_to_hub(REPO_NAME, token=os.environ["HF_TOKEN"])
                logging.info(f"Successfully pushed dataset")
                return
            except Exception as e:
                logging.warning(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    logging.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    logging.error(f"Failed to push dataset after {max_retries} attempts")
                    raise


def download_dataset_and_save_to_disk(dataset_name: Datasets, base_path: str = "./data", chunk_size: int = 10, total_examples: int = 50) -> None:
    dataset_path = os.path.join(base_path, dataset_name.name.lower())
    
    if not os.path.exists(dataset_path):
        print(f"Downloading dataset {dataset_name.value}...")
        token = os.getenv("HF_TOKEN")
        assert token is not None, "Please set the HF_TOKEN environment variable with your Hugging Face API token."
        
        dataset = load_dataset(dataset_name.value, split="train", token=token)
        
        os.makedirs(dataset_path, exist_ok=True)
        
        chunk_count = (total_examples + chunk_size - 1) // chunk_size
        for i in range(chunk_count):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, total_examples)
            chunk = dataset.select(range(start, end))
            chunk_path = os.path.join(dataset_path, f"chunk_{i}")
            chunk.save_to_disk(chunk_path)
            print(f"Saved chunk {i + 1}/{chunk_count} to {chunk_path}")
        
        # Save metadata
        metadata = {
            "total_examples": total_examples,
            "chunk_size": chunk_size,
            "chunk_count": chunk_count
        }
        with open(os.path.join(dataset_path, "metadata.json"), "w") as f:
            json.dump(metadata, f)
        
        print(f"Dataset downloaded and saved to {dataset_path}")
    else:
        print(f"Dataset already exists at {dataset_path}")

def load_dataset_from_disk(dataset_name: Datasets, base_path: str = "./data") -> Iterable:
    dataset_path = os.path.join(base_path, dataset_name.name.lower())
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please download it first.")
    
    # Load metadata
    with open(os.path.join(dataset_path, "metadata.json"), "r") as f:
        metadata = json.load(f)
    
    # Load all chunks
    datasets = []
    for i in range(metadata["chunk_count"]):
        chunk_path = os.path.join(dataset_path, f"chunk_{i}")
        chunk = load_dataset("arrow", data_files=f"{chunk_path}/dataset.arrow", split="train")
        datasets.append(chunk)
    
    # Combine all chunks into a single dataset
    combined_dataset = concatenate_datasets(datasets)
    return combined_dataset.to_iterable_dataset()


if __name__ == "__main__":
    try: 
        main()
    except Exception as e:
        logging.error(e)
        raise e