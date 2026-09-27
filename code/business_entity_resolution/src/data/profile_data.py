import os
import polars as pl
import pandas as pd

def create_validation_split(data_path: str, output_dir: str, val_ratio: float = 0.2, seed: int = 42):
    """
    Creates stratified train and validation splits without data leakage.
    Ensures that all entities of the same group remain together in either train or val.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pl.read_csv(data_path, separator="\t") if data_path.endswith(".tsv") else pl.read_csv(data_path)
    
    # Stratified entity split logic
    entities = df["legal_entity_id"].unique().shuffle(seed=seed)
    val_size = int(len(entities) * val_ratio)
    
    val_entities = set(entities[:val_size].to_list())
    
    train_df = df.filter(~pl.col("legal_entity_id").is_in(val_entities))
    val_df = df.filter(pl.col("legal_entity_id").is_in(val_entities))
    
    train_path = os.path.join(output_dir, "train_split_gt.tsv")
    val_path = os.path.join(output_dir, "val_split_gt.tsv")
    
    train_df.write_csv(train_path, separator="\t")
    val_df.write_csv(val_path, separator="\t")
    
    print(f"Validation split created: Train={len(train_df)}, Val={len(val_df)}")
    return train_path, val_path
