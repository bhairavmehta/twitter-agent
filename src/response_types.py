"""
Shared types and constants for response strategies used by both
content_generator.py and prompt_analyzer_agent.py
"""
from enum import Enum

# Define possible request types
class RequestType(str, Enum):
    CRYPTO_IMAGE = "crypto_image"
    CRYPTO_VIDEO = "crypto_video"
    NON_CRYPTO_MEDIA = "non_crypto_media"
    CRYPTO_INFO = "crypto_info"
    OTHER = "other"

# Define response types
class ResponseStrategy(str, Enum):
    GENERATE_CRYPTO_IMAGE = "generate_crypto_image"
    GENERATE_CRYPTO_VIDEO = "generate_crypto_video"
    SARCASTIC_RESPONSE = "sarcastic_response"
    SUGGEST_MEMECOINS = "suggest_memecoins"
    PROVIDE_CRYPTO_INFO = "provide_crypto_info"