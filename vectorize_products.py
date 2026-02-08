import logging
import os
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
from threading import Thread, Event

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
    'hosts': [f'http://{ES_HOST}:{ES_PORT}'],
    'basic_auth': (ES_USER, ES_PASSWORD),
    'request_timeout': 60,
    'retry_on_timeout': True,
    'max_retries': 3
}

EMBEDDING_API_URL = os.environ.get('EMBEDDING_API_URL', 'http://62.106.95.202:8585')
SOURCE_INDEX = os.environ.get('SOURCE_INDEX', 'blackeveryday_products_v5')
TARGET_INDEX = os.environ.get('TARGET_INDEX', 'blackeveryday_products_vectors_v1')

# Processing settings
SCROLL_SIZE = int(os.environ.get('SCROLL_SIZE', '500'))
EMBEDDING_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
NUM_EMBEDDING_WORKERS = int(os.environ.get('NUM_EMBEDDING_WORKERS', '3'))
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds


class EmbeddingClient:
    """Client for the embedding API with connection pooling."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        # Enable connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_batch_embeddings(
        self,
        texts: list[str],
        model: str,
        retries: int = MAX_RETRIES
    ) -> Optional[list[list[float]]]:
        """Get embeddings for a batch of texts."""
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
                    timeout=180
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

    def get_embeddings_both_models(
        self,
        texts: list[str]
    ) -> tuple[Optional[list], Optional[list]]:
        """Get embeddings from both models in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {
                executor.submit(self.get_batch_embeddings, texts, "heydari"): "heydari",
                executor.submit(self.get_batch_embeddings, texts, "intfloat-small"): "intfloat-small"
            }

            for future in as_completed(futures):
                model = futures[future]
                try:
                    results[model] = future.result()
                except Exception as e:
                    logger.error(f"Error getting {model} embeddings: {e}")
                    results[model] = None

        return results.get("heydari"), results.get("intfloat-small")


class ProductVectorizer:
    """Main class for vectorizing products with pipeline processing."""

    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE, num_workers: int = NUM_EMBEDDING_WORKERS):
        self.es = Elasticsearch(**ES_CONFIG)
        self.embedding_client = EmbeddingClient(EMBEDDING_API_URL)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.processed_count = 0
        self.error_count = 0

        # Queues for pipeline processing
        self.embed_queue: Queue = Queue(maxsize=num_workers * 2)
        self.write_queue: Queue = Queue(maxsize=num_workers * 2)
        self.stop_event = Event()

    def verify_connections(self) -> bool:
        """Verify Elasticsearch and embedding API connections."""
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

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
                "number_of_replicas": 0,
                "refresh_interval": "30s"
            },
            "mappings": {
                "dynamic": False,
                "properties": {
                    "_source_product_id": {"type": "keyword", "index": False},
                    "title_fa": {"type": "text", "index": False},
                    "title_fa_vector_heydari": {
                        "type": "dense_vector", "dims": 1024,
                        "index": True, "similarity": "cosine"
                    },
                    "title_fa_vector_intfloat_small": {
                        "type": "dense_vector", "dims": 384,
                        "index": True, "similarity": "cosine"
                    },
                    "title_en": {"type": "text", "index": False},
                    "title_en_vector_heydari": {
                        "type": "dense_vector", "dims": 1024,
                        "index": True, "similarity": "cosine"
                    },
                    "title_en_vector_intfloat_small": {
                        "type": "dense_vector", "dims": 384,
                        "index": True, "similarity": "cosine"
                    },
                    "brand": {"type": "text", "index": False},
                    "brand_vector_heydari": {
                        "type": "dense_vector", "dims": 1024,
                        "index": True, "similarity": "cosine"
                    },
                    "brand_vector_intfloat_small": {
                        "type": "dense_vector", "dims": 384,
                        "index": True, "similarity": "cosine"
                    }
                }
            }
        }

        self.es.indices.create(index=TARGET_INDEX, body=mapping)
        logger.info(f"Created target index '{TARGET_INDEX}'")

    def get_pending_products(self):
        """Generator that yields products that haven't been vectorized yet."""
        query = {
            "query": {
                "bool": {
                    "must": [{"term": {"is_vectorized": False}}]
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

    def embedding_worker(self, worker_id: int):
        """Worker thread that processes embedding requests."""
        logger.info(f"Embedding worker {worker_id} started")

        while not self.stop_event.is_set():
            try:
                batch = self.embed_queue.get(timeout=1)
                if batch is None:  # Poison pill
                    break

                products, texts, non_empty_indices = batch

                # Get embeddings from both models in parallel
                heydari_emb, intfloat_emb = self.embedding_client.get_embeddings_both_models(texts)

                if heydari_emb is None or intfloat_emb is None:
                    logger.error(f"Worker {worker_id}: Failed to get embeddings")
                    self.write_queue.put((products, None, None, non_empty_indices))
                else:
                    self.write_queue.put((products, heydari_emb, intfloat_emb, non_empty_indices))

                self.embed_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Embedding worker {worker_id} stopped")

    def write_worker(self):
        """Worker thread that writes results to Elasticsearch."""
        logger.info("Write worker started")
        es = Elasticsearch(**ES_CONFIG)  # Separate connection for write worker

        while not self.stop_event.is_set():
            try:
                result = self.write_queue.get(timeout=1)
                if result is None:  # Poison pill
                    break

                products, heydari_emb, intfloat_emb, non_empty_indices = result

                if heydari_emb is None or intfloat_emb is None:
                    self.error_count += len(products)
                    self.write_queue.task_done()
                    continue

                # Create embedding maps
                embedding_map_heydari = {}
                embedding_map_intfloat = {}
                for idx, orig_idx in enumerate(non_empty_indices):
                    embedding_map_heydari[orig_idx] = heydari_emb[idx]
                    embedding_map_intfloat[orig_idx] = intfloat_emb[idx]

                # Prepare bulk operations
                vector_actions = []
                update_actions = []

                for i, product in enumerate(products):
                    if i not in embedding_map_heydari:
                        continue

                    doc_id = product['_id']
                    source = product['_source']

                    vector_doc = {
                        "_source_product_id": doc_id,
                        "title_fa": source.get('title_fa', ''),
                        "title_fa_vector_heydari": embedding_map_heydari[i],
                        "title_fa_vector_intfloat_small": embedding_map_intfloat[i]
                    }

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

                    update_actions.append({
                        "_op_type": "update",
                        "_index": SOURCE_INDEX,
                        "_id": doc_id,
                        "doc": {"is_vectorized": True}
                    })

                # Execute bulk operations in parallel
                success_count = 0
                try:
                    if vector_actions:
                        # Run both bulk operations in parallel
                        with ThreadPoolExecutor(max_workers=2) as executor:
                            insert_future = executor.submit(
                                bulk, es, vector_actions,
                                raise_on_error=False, request_timeout=120
                            )
                            update_future = executor.submit(
                                bulk, es, update_actions,
                                raise_on_error=False, request_timeout=120
                            )

                            success, errors = insert_future.result()
                            success_count = success
                            if errors:
                                logger.error(f"Vector insert errors: {errors[:3]}")
                                self.error_count += len(errors)

                            update_future.result()  # Wait for updates

                        self.processed_count += success_count
                except Exception as e:
                    logger.error(f"Bulk operation error: {e}")
                    self.error_count += len(products)

                self.write_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Write worker error: {e}")

        logger.info("Write worker stopped")

    def run(self, limit: Optional[int] = None):
        """Main execution method with pipeline processing."""
        logger.info("Starting product vectorization with pipeline processing...")
        logger.info(f"Configuration: batch_size={self.batch_size}, num_workers={self.num_workers}")

        if not self.verify_connections():
            logger.error("Connection verification failed. Exiting.")
            return

        self.ensure_target_index_exists()

        count_response = self.es.count(
            index=SOURCE_INDEX,
            body={"query": {"term": {"is_vectorized": False}}}
        )
        total_pending = count_response['count']
        logger.info(f"Found {total_pending:,} products to vectorize")

        if total_pending == 0:
            logger.info("No products to process")
            return

        # Start worker threads
        embedding_threads = []
        for i in range(self.num_workers):
            t = Thread(target=self.embedding_worker, args=(i,), daemon=True)
            t.start()
            embedding_threads.append(t)

        write_thread = Thread(target=self.write_worker, daemon=True)
        write_thread.start()

        # Process products
        batch = []
        start_time = time.time()
        last_log_time = start_time
        batches_submitted = 0

        try:
            for product in self.get_pending_products():
                batch.append(product)

                if len(batch) >= self.batch_size:
                    # Prepare batch for embedding
                    texts = []
                    non_empty_indices = []

                    for i, prod in enumerate(batch):
                        title_fa = prod['_source'].get('title_fa', '')
                        if title_fa and isinstance(title_fa, str) and title_fa.strip():
                            texts.append(title_fa.strip())
                            non_empty_indices.append(i)

                    if texts:
                        self.embed_queue.put((batch, texts, non_empty_indices))
                        batches_submitted += 1

                    batch = []

                    # Log progress periodically
                    current_time = time.time()
                    if current_time - last_log_time >= 10:  # Log every 10 seconds
                        elapsed = current_time - start_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0
                        eta_seconds = (total_pending - self.processed_count) / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600

                        logger.info(
                            f"Progress: {self.processed_count:,}/{total_pending:,} "
                            f"({100 * self.processed_count / total_pending:.2f}%) | "
                            f"Rate: {rate:.1f}/s | "
                            f"ETA: {eta_hours:.1f}h | "
                            f"Errors: {self.error_count:,} | "
                            f"Queue: {self.embed_queue.qsize()}/{self.write_queue.qsize()}"
                        )
                        last_log_time = current_time

                    # Check limit
                    if limit and self.processed_count >= limit:
                        logger.info(f"Reached limit of {limit} products")
                        break

            # Process remaining batch
            if batch:
                texts = []
                non_empty_indices = []
                for i, prod in enumerate(batch):
                    title_fa = prod['_source'].get('title_fa', '')
                    if title_fa and isinstance(title_fa, str) and title_fa.strip():
                        texts.append(title_fa.strip())
                        non_empty_indices.append(i)
                if texts:
                    self.embed_queue.put((batch, texts, non_empty_indices))

        finally:
            # Wait for queues to be processed
            logger.info("Waiting for queues to finish...")
            self.embed_queue.join()
            self.write_queue.join()

            # Stop workers
            self.stop_event.set()
            for _ in range(self.num_workers):
                self.embed_queue.put(None)
            self.write_queue.put(None)

            for t in embedding_threads:
                t.join(timeout=5)
            write_thread.join(timeout=5)

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
    parser.add_argument('--limit', type=int, default=None, help='Limit number of products')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for embedding API')
    parser.add_argument('--workers', type=int, default=None, help='Number of embedding workers')

    args = parser.parse_args()

    limit = args.limit
    if limit is None:
        env_limit = os.environ.get('LIMIT', '')
        if env_limit.strip():
            limit = int(env_limit)

    batch_size = args.batch_size if args.batch_size else EMBEDDING_BATCH_SIZE
    num_workers = args.workers if args.workers else NUM_EMBEDDING_WORKERS

    vectorizer = ProductVectorizer(batch_size=batch_size, num_workers=num_workers)
    vectorizer.run(limit=limit)


if __name__ == "__main__":
    main()
