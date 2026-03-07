import logging
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.llms import LLM

logger = logging.getLogger(__name__)

def initialize_reranker(
    llm: LLM,
    top_n: int = 5,
    choice_batch_size: int = 5
) -> LLMRerank:
    """
    Initializes a neural reranker using an LLM.

    This reranker refines the retrieved documents by using a language model
    to score their relevance to the query, ensuring only the most
    contextually appropriate documents are passed to the final synthesis step.

    Args:
        llm (LLM): The language model to use for reranking.
        top_n (int): The number of top documents to return after reranking.
        choice_batch_size (int): The batch size for reranking choices.

    Returns:
        LLMRerank: An initialized LLMRerank postprocessor instance.
    """
    try:
        reranker = LLMRerank(
            choice_batch_size=choice_batch_size, top_n=top_n, llm=llm,
        )
        logger.info(f"LLMRerank initialized with top_n={top_n}")
        return reranker
    except Exception as e:
        logger.error(f"Failed to initialize LLMRerank: {e}")
        raise