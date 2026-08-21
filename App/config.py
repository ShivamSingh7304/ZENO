import os 
from dotenv import load_dotenv

load_dotenv()

class Settings:
    #groq config
    GROQ_API_KEY=os.getenv("GROQ_API_KEY")
    GROQ_FALLBACK_API_KEY=os.getenv("GROQ_FALLBACK_API_KEY")

    #model
    LLM_MODEL="openai/gpt-oss-20b"
    TEMPERATURE = 0.3
    
    #qdrant config
    QDRANT_URL=os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION='Mental-health-companion'

    PORTKEY_API_KEY: str = os.getenv("PORTKEY_API_KEY")
    GROQ_PRIMARY_SLUG: str = "primary"
    GROQ_FALLBACK_SLUG: str = "fallback"
    PORTKEY_CONFIG: str = os.getenv("PORTKEY_PRIMARY_CONFIG_ID")

settings =Settings()
