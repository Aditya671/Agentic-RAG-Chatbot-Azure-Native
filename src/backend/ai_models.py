from enum import Enum

class AIResponseMode(str, Enum):
    """Mode determining the detail level of engine responses."""
    DETAILED = 'detailed'
    CONCISE = 'concise'

class AIModelTypes(str, Enum):
    """Enum for supported AI models."""
    # O3_MINI = "o3-mini"
    # O3 = "o3"
    O4_MINI = "o4-mini"
    O4_MINI_HIGH = "o4-mini-high"
    GPT4O = "gpt-4o"
    GPT41 = "gpt-4.1"
    # GPT41NANO = "gpt-4.1-nano"
    GPT41_MINI = "gpt-4.1-mini"
    # GPT5 = "gpt-5"
    GPT51 = "gpt-5.1"
    
    # GPT5_MINI = "gpt-5-mini"

MODEL_TOKEN_LIMITS = {
    AIModelTypes.O4_MINI: 100000,
    AIModelTypes.O4_MINI_HIGH: 100000,
    AIModelTypes.GPT4O: 100000,
    AIModelTypes.GPT41: 100000,
    AIModelTypes.GPT41_MINI: 100000,
    AIModelTypes.GPT51: 180000,
}

DEFAULT_REASONING_EFFORT = {
    AIModelTypes.O4_MINI: "high",
    AIModelTypes.O4_MINI_HIGH: "high",
    AIModelTypes.GPT4O: "medium",
    AIModelTypes.GPT41: "medium",
    AIModelTypes.GPT41_MINI: "medium",
    # GPT51: no special reasoning_effort by default (treat as normal)
}