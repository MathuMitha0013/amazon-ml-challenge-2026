"""
Multi-route blocking strategies and candidate generation coordinator.
"""

from .exact_blocking import generate_exact_blocks
from .token_blocking import generate_token_blocks
from .address_blocking import generate_address_blocks
from .ngram_blocking import generate_ngram_blocks
from .candidate_generation import generate_candidate_pairs

__all__ = [
    "generate_exact_blocks",
    "generate_token_blocks",
    "generate_address_blocks",
    "generate_ngram_blocks",
    "generate_candidate_pairs",
]
