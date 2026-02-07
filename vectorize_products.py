import logging
import os
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk, scan

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration from environment variables with defaults
ES_HOST = os.environ.get('ES_HOST', '62.106.95.202')
ES_PORT = int(os.environ.get('ES_PORT', '9200'))
ES_USER = os.environ.get('ES_USER', 'elastic')
ES_PASSWORD = os.environ.get('ES_PASSWORD', '13680129a')

ES_CONFIG = {
    'hosts': [{'host': ES_HOST, 'port': ES_PORT, 'scheme': 'http'}],
    'http_auth': (ES_USER, ES_PASSWORD),
    'timeout': 60,
    'retry_on_timeout': True,
    'max_retries': 3
}

EMBEDDING_API_URL = os.environ.get('EMBEDDING_API_URL', 'http://62.106.95.202:8585')
SOURCE_INDEX = os.environ.get('SOURCE_INDEX', 'blackeveryday_products_v5')
TARGET_INDEX = os.environ.get('TARGET_INDEX', 'blackeveryday_products_vectors_v1')

# Processing settings
SCROLL_SIZE = int(os.environ.get('SCROLL_SIZE', '500'))
EMBEDDING_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class EmbeddingClient:
    """Client for the embedding API."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()

    def get_batch_embeddings(
        self,
        texts: list[str],
        model: str,
        retries: int = MAX_RETRIES
    ) -> Optional[list[list[float]]]:
        """
        Get embeddings for a batch of texts.

        Args:
            texts: List of texts to embed
            model: Model to use ('heydari' or 'intfloat-small')
            retries: Number of retries on failure

        Returns:
            List of embedding vectors or None on failure
        """
        url = f"{self.base_url}/embed/batch"
        headers = {
            "model": model,
            "Content-Type": "application/json"
        }
        payload = {"texts": texts}

        for attempt in range(retries):
            try:
                response = self.session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=120
                )
                response.raise_for_status()
                return response.json()["embeddings"]
            except requests.exceptions.RequestException as e:
                logger.warning(
                    f"Embedding API error (attempt {attempt + 1}/{retries}): {e}"
                )
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        return None


class ProductVectorizer:
    """Main class for vectorizing products."""

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE):
        self.es = Elasticsearch(**ES_CONFIG)
        self.embedding_client = EmbeddingClient(EMBEDDING_API_URL)
        self.batch_size = batch_size
        self.processed_count = 0
        self.error_count = 0

    def verify_connections(self) -> bool:
        """Verify Elasticsearch and embedding API connections."""
        # Check Elasticsearch
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

        # Check embedding API
        try:
            response = requests.post(
                f"{EMBEDDING_API_URL}/embed",
                headers={"model": "heydari", "Content-Type": "application/json"},
                json={"text": "test"},
                timeout=30
            )
            response.raise_for_status()
            logger.info("Embedding API is accessible")
        except Exception as e:
            logger.error(f"Failed to connect to embedding API: {e}")
            return False

        return True

    def ensure_target_index_exists(self):
        """Create the target index if it doesn't exist."""
        if self.es.indices.exists(index=TARGET_INDEX):
            logger.info(f"Target index '{TARGET_INDEX}' already exists")
            return

        mapping = {
            "settings": {
                "number_of_shards": 2,
                "number_of_replicas": 0
            },
            "mappings": {
                "dynamic": False,
                "properties": {
                    "_source_product_id": {
                        "type": "keyword",
                        "index": False
                    },
                    "title_fa": {
                        "type": "text",
                        "index": False
                    },
                    "title_fa_vector_heydari": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "title_fa_vector_intfloat_small": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "title_en": {
                        "type": "text",
                        "index": False
                    },
                    "title_en_vector_heydari": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "title_en_vector_intfloat_small": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "brand": {
                        "type": "text",
                        "index": False
                    },
                    "brand_vector_heydari": {
                        "type": "dense_vector",
                        "dims": 1024,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "brand_vector_intfloat_small": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }

        self.es.indices.create(index=TARGET_INDEX, body=mapping)
        logger.info(f"Created target index '{TARGET_INDEX}'")

    def get_pending_products(self):
        """
        Generator that yields products that haven't been vectorized yet.
        Uses scroll API for efficient iteration over large result sets.
        """
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"is_vectorized": False}}
                    ]
                }
            },
            "_source": ["title_fa", "title_en", "brand"]
        }

        return scan(
            self.es,
            index=SOURCE_INDEX,
            query=query,
            scroll='10m',
            size=SCROLL_SIZE,
            request_timeout=120
        )

    def get_embeddings_parallel(
        self,
        texts: list[str]
    ) -> tuple[Optional[list], Optional[list]]:
        """
        Get embeddings from both models in parallel.

        Returns:
            Tuple of (heydari_embeddings, intfloat_embeddings)
        """
        results = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(
                    self.embedding_client.get_batch_embeddings,
                    texts,
                    "heydari"
                ): "heydari",
                executor.submit(
                    self.embedding_client.get_batch_embeddings,
                    texts,
                    "intfloat-small"
                ): "intfloat-small"
            }

            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"Error getting {model} embeddings: {e}")
                    results[model] = None

        return results.get("heydari"), results.get("intfloat-small")

    def process_batch(self, products: list[dict]) -> tuple[int, int]:
        """
        Process a batch of products.

        Args:
            products: List of product documents with _id and _source

        Returns:
            Tuple of (success_count, error_count)
        """
        if not products:
            return 0, 0

        # Extract title_fa texts (handle missing/empty values)
        texts = []
        valid_indices = []

        for i, product in enumerate(products):
            title_fa = product['_source'].get('title_fa', '')
            if title_fa and isinstance(title_fa, str) and title_fa.strip():
                texts.append(title_fa.strip())
                valid_indices.append(i)
            else:
                texts.append('')  # Placeholder
                valid_indices.append(i)

        # Filter out empty texts for embedding
        non_empty_texts = [t for t in texts if t]
        non_empty_indices = [i for i, t in enumerate(texts) if t]

        if not non_empty_texts:
            logger.warning("No valid texts in batch, skipping embedding")
            return 0, len(products)

        # Get embeddings for non-empty texts
        heydari_embeddings, intfloat_embeddings = self.get_embeddings_parallel(
            non_empty_texts
        )

        if heydari_embeddings is None or intfloat_embeddings is None:
            logger.error("Failed to get embeddings for batch")
            return 0, len(products)

        # Create mapping from original index to embeddings
        embedding_map_heydari = {}
        embedding_map_intfloat = {}

        for idx, orig_idx in enumerate(non_empty_indices):
            embedding_map_heydari[orig_idx] = heydari_embeddings[idx]
            embedding_map_intfloat[orig_idx] = intfloat_embeddings[idx]

        # Prepare bulk operations for target index
        vector_actions = []
        update_actions = []

        for i, product in enumerate(products):
            doc_id = product['_id']
            source = product['_source']

            # Skip if no valid embedding
            if i not in embedding_map_heydari:
                continue

            # Prepare vector document
            vector_doc = {
                "_source_product_id": doc_id,
                "title_fa": source.get('title_fa', ''),
                "title_fa_vector_heydari": embedding_map_heydari[i],
                "title_fa_vector_intfloat_small": embedding_map_intfloat[i]
            }

            # Add optional fields if present
            if source.get('title_en'):
                vector_doc['title_en'] = source['title_en']

            if source.get('brand'):
                brand = source['brand']
                if isinstance(brand, dict):
                    vector_doc['brand'] = brand.get('title_fa', '') or brand.get('title_en', '')
                else:
                    vector_doc['brand'] = str(brand)

            vector_actions.append({
                "_index": TARGET_INDEX,
                "_id": doc_id,
                "_source": vector_doc
            })

            # Prepare update action for source index
            update_actions.append({
                "_op_type": "update",
                "_index": SOURCE_INDEX,
                "_id": doc_id,
                "doc": {"is_vectorized": True}
            })

        # Execute bulk operations
        success_count = 0
        error_count = 0

        try:
            # Insert vectors
            if vector_actions:
                success, errors = bulk(
                    self.es,
                    vector_actions,
                    raise_on_error=False,
                    request_timeout=120
                )
                success_count = success

                if errors:
                    logger.error(f"Vector insert errors: {errors[:3]}")
                    error_count = len(errors)

            # Update is_vectorized flag
            if update_actions and success_count > 0:
                bulk(
                    self.es,
                    update_actions,
                    raise_on_error=False,
                    request_timeout=120
                )

        except Exception as e:
            logger.error(f"Bulk operation error: {e}")
            error_count = len(products)

        return success_count, error_count

    def run(self, limit: Optional[int] = None):
        """
        Main execution method.

        Args:
            limit: Optional limit on number of products to process (for testing)
        """
        logger.info("Starting product vectorization...")

        # Verify connections
        if not self.verify_connections():
            logger.error("Connection verification failed. Exiting.")
            return

        # Ensure target index exists
        self.ensure_target_index_exists()

        # Get count of pending products
        count_response = self.es.count(
            index=SOURCE_INDEX,
            body={"query": {"term": {"is_vectorized": False}}}
        )
        total_pending = count_response['count']
        logger.info(f"Found {total_pending:,} products to vectorize")

        if total_pending == 0:
            logger.info("No products to process")
            return

        # Process products in batches
        batch = []
        start_time = time.time()

        for product in self.get_pending_products():
            batch.append(product)

            if len(batch) >= self.batch_size:
                success, errors = self.process_batch(batch)
                self.processed_count += success
                self.error_count += errors

                # Log progress
                elapsed = time.time() - start_time
                rate = self.processed_count / elapsed if elapsed > 0 else 0
                eta_seconds = (total_pending - self.processed_count) / rate if rate > 0 else 0
                eta_hours = eta_seconds / 3600

                logger.info(
                    f"Progress: {self.processed_count:,}/{total_pending:,} "
                    f"({100 * self.processed_count / total_pending:.2f}%) | "
                    f"Rate: {rate:.1f}/s | "
                    f"ETA: {eta_hours:.1f}h | "
                    f"Errors: {self.error_count:,}"
                )

                batch = []

                # Check limit
                if limit and self.processed_count >= limit:
                    logger.info(f"Reached limit of {limit} products")
                    break

        # Process remaining products
        if batch and (not limit or self.processed_count < limit):
            success, errors = self.process_batch(batch)
            self.processed_count += success
            self.error_count += errors

        # Final summary
        elapsed = time.time() - start_time
        logger.info(
            f"\n{'=' * 50}\n"
            f"Vectorization Complete!\n"
            f"Total processed: {self.processed_count:,}\n"
            f"Total errors: {self.error_count:,}\n"
            f"Time elapsed: {elapsed / 3600:.2f} hours\n"
            f"Average rate: {self.processed_count / elapsed:.1f} products/second\n"
            f"{'=' * 50}"
        )


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Vectorize products from Elasticsearch")
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of products to process (for testing)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for embedding API'
    )

    args = parser.parse_args()

    # Use environment variable LIMIT if CLI arg not provided
    limit = args.limit
    if limit is None:
        env_limit = os.environ.get('LIMIT', '')
        if env_limit.strip():
            limit = int(env_limit)

    # Override batch size if provided via CLI
    batch_size = args.batch_size if args.batch_size else EMBEDDING_BATCH_SIZE

    vectorizer = ProductVectorizer(batch_size=batch_size)
    vectorizer.run(limit=limit)


if __name__ == "__main__":
    main()
