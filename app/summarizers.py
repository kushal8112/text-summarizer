# Add these imports at the top of the file
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import logging
from typing import Optional

import nltk
import math
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

def sentence_vectors_tfidf(sentences):
    """Return TF-IDF vectors for all sentences."""
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(sentences).toarray()

def estimate_summary_length(n, min_sentences=1, max_sentences=15, x=0.20):
    """Estimate the number of sentences for the summary."""
    return max(min_sentences, min(max_sentences, int(x* n + math.log2(n))))

def extractive_summarizer(text: str, length: str = "medium") -> str:
    """
    Generate an extractive summary using TextRank algorithm with TF-IDF vectors.
    
    Args:
        text: Input text to summarize
        min_length: Minimum length of the summary in sentences (optional)
        max_length: Maximum length of the summary in sentences (optional)
        
    Returns:
        str: Generated summary
    """
    try:
        # Tokenize text into sentences
        sentences = sent_tokenize(text)
        n = len(sentences)
        
        if n <= 1:
            return text
            
        # Calculate number of sentences for the summary
        if length == "short":
            num_sentences = estimate_summary_length(n, x=0.10)
        elif length == "medium":
            num_sentences = estimate_summary_length(n, x=0.20)
        elif length == "long":
            num_sentences = estimate_summary_length(n, x=0.25)
            
        # Calculate TF-IDF vectors
        vectors = sentence_vectors_tfidf(sentences)
        
        # Build similarity matrix
        sim_matrix = cosine_similarity(vectors)
        np.fill_diagonal(sim_matrix, 0)  # Remove self-similarity
        
        # Build the graph and calculate PageRank scores
        graph = nx.from_numpy_array(sim_matrix)
        scores = nx.pagerank(graph)
        
        # Get top-ranked sentences
        top_indices = sorted(
            sorted(scores, key=scores.get, reverse=True)[:num_sentences]
        )
        summary = " ".join([sentences[i].strip() for i in top_indices])
        
        return summary
        
    except Exception as e:
        logger.error(f"Error in extractive_summarizer: {str(e)}")
        raise RuntimeError("Failed to generate extractive summary") from e

# Add the T5Summarizer class
class T5Summarizer:
    def __init__(self, 
                 model_name: str = "t5-base",
                 max_input_length: int = 4096,
                 min_summary_length: int = 30,
                 max_summary_length: int = 500,
                 num_beams: int = 4,
                 device: Optional[str] = None):
        """Initialize the T5 model for summarization.
        
        Args:
            model_name: Name of the T5 model to use
            max_input_length: Maximum length of input text
            min_summary_length: Minimum length of generated summary
            max_summary_length: Maximum length of generated summary
            num_beams: Number of beams for beam search
            device: Device to use ('cuda' or 'cpu')
        """
        try:
            # Validate parameters
            if not isinstance(model_name, str):
                raise ValueError("model_name must be a string")
            if not 100 <= max_input_length <= 4096:
                raise ValueError("max_input_length must be between 100 and 1024")
            if not 10 <= min_summary_length <= 500:
                raise ValueError("min_summary_length must be between 10 and 500")
            if not 50 <= max_summary_length <= 500:
                raise ValueError("max_summary_length must be between 50 and 500")
            if min_summary_length > max_summary_length:
                raise ValueError("min_summary_length cannot be greater than max_summary_length")
            if not 1 <= num_beams <= 8:
                raise ValueError("num_beams must be between 1 and 8")
            
            self.model_name = model_name
            self.max_input_length = max_input_length
            self.min_summary_length = min_summary_length
            self.max_summary_length = max_summary_length
            self.num_beams = num_beams
            
            # Initialize tokenizer and model
            logger.info(f"Loading T5 model: {model_name}")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            
            # Set device
            if device is None:
                self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            else:
                self.device = torch.device(device)
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Error initializing T5Summarizer: {str(e)}")
            raise

    def validate_input(self, text: str) -> str:
        """Validate and preprocess input text.
        
        Args:
            text: Input text to validate
            
        Returns:
            Cleaned and validated text
            
        Raises:
            ValueError: If input is invalid
        """
        if not isinstance(text, str):
            raise ValueError("Input must be a string")
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Check length
        if len(text) > self.max_input_length:
            logger.warning(f"Input text is too long ({len(text)} chars), truncating to {self.max_input_length} chars")
            text = text[:self.max_input_length]
            
        if len(text) < 10:
            raise ValueError("Input text is too short")
            
        return text

    def summarize(self, 
                  text: str, 
                  min_length: Optional[int] = None,
                  max_length: Optional[int] = None,
                  num_beams: Optional[int] = None) -> str:
        """Generate summary using T5 model.
        
        Args:
            text: Input text to summarize
            min_length: Minimum length of generated summary
            max_length: Maximum length of generated summary
            num_beams: Number of beams for beam search
            
        Returns:
            Generated summary as string
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If summarization fails
        """
        try:
            # Validate input
            text = self.validate_input(text)
            
            # Use default values if not provided
            max_length = max_length or self.max_summary_length
            min_length = min_length or self.min_summary_length
            num_beams = num_beams or self.num_beams
            
            # Validate parameters
            if not 10 <= min_length <= 500:
                raise ValueError("min_length must be between 10 and 500")
            if not 50 <= max_length <= 500:
                raise ValueError("max_length must be between 50 and 500")
            if min_length > max_length:
                raise ValueError("min_length cannot be greater than max_length")
            if not 1 <= num_beams <= 8:
                raise ValueError("num_beams must be between 1 and 8")
            
            # Prepare input text
            input_text = "summarize: " + text
            
            logger.info(f"Generating summary for text of length {len(text)}")
            
            # Tokenize input
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.max_input_length,
                truncation=True
            ).to(self.device)
            
            # Generate summary
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=2.0
            )
            
            # Decode and return summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            logger.info(f"Generated summary of length {len(summary)}")
            
            return summary
            
        except ValueError as ve:
            logger.error(f"Validation error: {str(ve)}")
            raise
        except RuntimeError as re:
            logger.error(f"Runtime error during summarization: {str(re)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during summarization: {str(e)}")
            raise RuntimeError("Failed to generate summary")

# Initialize the T5 summarizer with default parameters
t5_summarizer = T5Summarizer()

def capitalize_sentences(text: str) -> str:
    """Capitalize the first letter of each sentence in the text."""
    import re
    # Split into sentences using regex (looks for .!? followed by space or end of string)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Capitalize first letter of each sentence and join them back
    return ' '.join(sentence[0].upper() + sentence[1:] if sentence else '' 
                   for sentence in sentences)

# Update the abstractive_summarizer function to use the new implementation
def abstractive_summarizer(text: str, length: str = "medium", 
                         min_length: Optional[int] = None, 
                         max_length: Optional[int] = None) -> str:
    """Generate summary using T5 model with automatic length calculation.
    
    Args:
        text: Input text to summarize
        length: Summary length type - "short", "medium", or "long"
        min_length: Optional explicit minimum length in tokens
        max_length: Optional explicit maximum length in tokens
        
    Returns:
        Generated summary text
    """
    # Initialize T5Summarizer if not already done
    global t5_summarizer
    if t5_summarizer is None:
        t5_summarizer = T5Summarizer()
    
    try:
        # Tokenize input to get token count
        input_ids = t5_summarizer.tokenizer.encode(text, return_tensors=None)
        input_token_count = len(input_ids)
        
        # Calculate min and max lengths if not provided
        if min_length is None or max_length is None:
            if length == "short":
                min_len = int(0.1 * input_token_count)
                max_len = int(0.2 * input_token_count)
            elif length == "medium":
                min_len = int(0.2 * input_token_count)
                max_len = int(0.3 * input_token_count)
            elif length == "long":
                min_len = int(0.3 * input_token_count)
                max_len = int(0.45 * input_token_count)
            else:
                raise ValueError("Invalid length type. Must be 'short', 'medium', or 'long'")
            
            # Apply constraints
            min_len = max(10, min_len)  # Ensure minimum length is at least 10
            max_len = max(50, max_len)  # Ensure maximum length is at least 50
            max_len = min(500, max_len)  # Cap maximum length at 500
            
            # Use calculated values if explicit values not provided
            if min_length is None:
                min_length = min_len
            if max_length is None:
                max_length = max_len
        
        # Ensure max_length is not less than min_length
        max_length = max(min_length, max_length)
        
        # Generate summary with the calculated lengths
        return capitalize_sentences(t5_summarizer.summarize(
            text, 
            min_length=min_length,
            max_length=max_length,
            num_beams=4
        ))
        
    except Exception as e:
        logger.error(f"Error in abstractive summarization: {str(e)}")
        raise