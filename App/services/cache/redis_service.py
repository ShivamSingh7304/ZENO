import json
import hashlib
import logfire

from upstash_redis import Redis

from App.config import settings


redis_client = Redis(
    url=settings.UPSTASH_REDIS_REST_URL,
    token=settings.UPSTASH_REDIS_REST_TOKEN
)


def generate_cache_key(prefix: str, value: str) -> str:
    """
    Generates a consistent hashed Redis cache key.
    """

    hashed_value = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return f"{prefix}:{hashed_value}"


def get_cache(key: str):
    """
    Retrieve data from Redis cache.
    """

    try:

        cached_data = redis_client.get(key)

        if cached_data is not None:

            logfire.info(
                f"Redis Cache HIT | key={key}"
            )

            return cached_data

        logfire.info(
            f"Redis Cache MISS | key={key}"
        )

        return None

    except Exception as e:

        logfire.warning(
            f"Redis cache read failed: {e}"
        )

        return None


def set_cache(
    key: str,
    value,
    ttl: int = 3600
):
    """
    Store data in Redis cache.

    ttl is in seconds.
    Default = 1 hour.
    """

    try:

        redis_client.set(
            key,
            value,
            ex=ttl
        )

        logfire.info(
            f"Redis Cache SET | key={key} | ttl={ttl}"
        )

    except Exception as e:

        logfire.warning(
            f"Redis cache write failed: {e}"
        )