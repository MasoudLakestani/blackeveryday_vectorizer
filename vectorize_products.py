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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
INDEX = os.environ.get('INDEX', 'blackeveryday_products_v5')
SOURCE_FIELD = os.environ.get('SOURCE_FIELD', 'title_fa')
DESTINATION_FIELD = os.environ.get('DESTINATION_FIELD', 'title_fa_vector_intfloat_base')
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'intfloat-base')

SCROLL_SIZE = int(os.environ.get('SCROLL_SIZE', '500'))
EMBEDDING_BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '100'))
NUM_EMBEDDING_WORKERS = int(os.environ.get('NUM_EMBEDDING_WORKERS', '3'))
POLL_INTERVAL_MINUTES = int(os.environ.get('POLL_INTERVAL_MINUTES', '30'))
MAX_RETRIES = 3
RETRY_DELAY = 5

IS_ACTIVE_FILTER = os.environ.get('IS_ACTIVE', 'true').strip().lower()
IS_ACTIVE = IS_ACTIVE_FILTER in ('1', 'true', 'yes', 'y')


class EmbeddingClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3
        )
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def get_batch_embeddings(self, texts: list[str], model: str, retries: int = MAX_RETRIES) -> Optional[list[list[float]]]:
        url = f"{self.base_url}/embed/batch"
        headers = {"model": model, "Content-Type": "application/json"}
        payload = {"texts": texts}

        for attempt in range(retries):
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=180)
                response.raise_for_status()
                return response.json()["embeddings"]
            except requests.exceptions.RequestException as e:
                logger.warning(f"Embedding API error (attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))

        return None


class ProductVectorizer:
    def __init__(self, batch_size: int = EMBEDDING_BATCH_SIZE, num_workers: int = NUM_EMBEDDING_WORKERS,
                 model: str = EMBEDDING_MODEL, source_field: str = SOURCE_FIELD,
                 destination_field: str = DESTINATION_FIELD):
        self.es = Elasticsearch(**ES_CONFIG)
        self.embedding_client = EmbeddingClient(EMBEDDING_API_URL)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = model
        self.source_field = source_field
        self.destination_field = destination_field
        self.processed_count = 0
        self.error_count = 0

        self.embed_queue: Queue = Queue(maxsize=num_workers * 2)
        self.write_queue: Queue = Queue(maxsize=num_workers * 2)
        self.stop_event = Event()

    def verify_connections(self) -> bool:
        try:
            info = self.es.info()
            logger.info(f"Connected to Elasticsearch: {info['version']['number']}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            return False

        try:
            response = requests.post(
                f"{EMBEDDING_API_URL}/embed",
                headers={"model": self.model, "Content-Type": "application/json"},
                json={"text": "test"},
                timeout=30
            )
            response.raise_for_status()
            logger.info(f"Embedding API accessible (model: {self.model})")
        except Exception as e:
            logger.error(f"Failed to connect to embedding API: {e}")
            return False

        return True

    def get_pending_products(self):
        """Yields products where is_active matches and destination field is not yet set."""
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"term": {"is_active": IS_ACTIVE}}
                    ],
                    "must_not": [
                        {"exists": {"field": self.destination_field}}
                    ]
                }
            },
            "_source": [self.source_field]
        }

        return scan(
            self.es,
            index=INDEX,
            query=query,
            scroll='10m',
            size=SCROLL_SIZE,
            request_timeout=120
        )

    def embedding_worker(self, worker_id: int):
        logger.info(f"Embedding worker {worker_id} started")

        while not self.stop_event.is_set():
            try:
                batch = self.embed_queue.get(timeout=1)
                if batch is None:
                    break

                products, texts, non_empty_indices = batch
                embeddings = self.embedding_client.get_batch_embeddings(texts, self.model)

                if embeddings is None:
                    logger.error(f"Worker {worker_id}: Failed to get embeddings")
                    self.write_queue.put((products, None, non_empty_indices))
                else:
                    self.write_queue.put((products, embeddings, non_empty_indices))

                self.embed_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

        logger.info(f"Embedding worker {worker_id} stopped")

    def write_worker(self):
        logger.info("Write worker started")
        es = Elasticsearch(**ES_CONFIG)

        while not self.stop_event.is_set():
            try:
                result = self.write_queue.get(timeout=1)
                if result is None:
                    break

                products, embeddings, non_empty_indices = result

                if embeddings is None:
                    self.error_count += len(products)
                    self.write_queue.task_done()
                    continue

                embedding_map = {orig_idx: embeddings[idx] for idx, orig_idx in enumerate(non_empty_indices)}

                update_actions = []
                for i, product in enumerate(products):
                    if i not in embedding_map:
                        continue

                    update_actions.append({
                        "_op_type": "update",
                        "_index": INDEX,
                        "_id": product['_id'],
                        "doc": {self.destination_field: embedding_map[i]}
                    })

                if update_actions:
                    try:
                        success, errors = bulk(
                            es, update_actions,
                            raise_on_error=False, request_timeout=120
                        )
                        self.processed_count += success
                        if errors:
                            logger.error(f"Bulk update errors: {errors[:3]}")
                            self.error_count += len(errors)
                    except Exception as e:
                        logger.error(f"Bulk operation error: {e}")
                        self.error_count += len(update_actions)

                self.write_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Write worker error: {e}")

        logger.info("Write worker stopped")

    def run(self, limit: Optional[int] = None):
        logger.info("Starting product vectorization...")
        logger.info(
            f"Config: index={INDEX}, source_field={self.source_field}, "
            f"destination_field={self.destination_field}, model={self.model}, "
            f"batch_size={self.batch_size}, workers={self.num_workers}"
        )

        if not self.verify_connections():
            logger.error("Connection verification failed. Exiting.")
            return

        count_response = self.es.count(
            index=INDEX,
            body={
                "query": {
                    "bool": {
                        "must": [{"term": {"is_active": IS_ACTIVE}}],
                        "must_not": [{"exists": {"field": self.destination_field}}]
                    }
                }
            }
        )
        total_pending = count_response['count']
        logger.info(f"Found {total_pending:,} products to vectorize")

        if total_pending == 0:
            logger.info("No products to process")
            return

        embedding_threads = []
        for i in range(self.num_workers):
            t = Thread(target=self.embedding_worker, args=(i,), daemon=True)
            t.start()
            embedding_threads.append(t)

        write_thread = Thread(target=self.write_worker, daemon=True)
        write_thread.start()

        batch = []
        start_time = time.time()
        last_log_time = start_time

        try:
            for product in self.get_pending_products():
                batch.append(product)

                if len(batch) >= self.batch_size:
                    texts = []
                    non_empty_indices = []
                    for i, prod in enumerate(batch):
                        value = prod['_source'].get(self.source_field, '')
                        if value and isinstance(value, str) and value.strip():
                            texts.append(f"passage: {value.strip()}")
                            non_empty_indices.append(i)

                    if texts:
                        self.embed_queue.put((batch, texts, non_empty_indices))

                    batch = []

                    current_time = time.time()
                    if current_time - last_log_time >= 10:
                        elapsed = current_time - start_time
                        rate = self.processed_count / elapsed if elapsed > 0 else 0
                        eta_seconds = (total_pending - self.processed_count) / rate if rate > 0 else 0
                        logger.info(
                            f"Progress: {self.processed_count:,}/{total_pending:,} "
                            f"({100 * self.processed_count / total_pending:.2f}%) | "
                            f"Rate: {rate:.1f}/s | "
                            f"ETA: {eta_seconds / 3600:.1f}h | "
                            f"Errors: {self.error_count:,}"
                        )
                        last_log_time = current_time

                    if limit and self.processed_count >= limit:
                        logger.info(f"Reached limit of {limit} products")
                        break

            if batch:
                texts = []
                non_empty_indices = []
                for i, prod in enumerate(batch):
                    value = prod['_source'].get(self.source_field, '')
                    if value and isinstance(value, str) and value.strip():
                        texts.append(f"passage: {value.strip()}")
                        non_empty_indices.append(i)
                if texts:
                    self.embed_queue.put((batch, texts, non_empty_indices))

        finally:
            logger.info("Waiting for queues to finish...")
            self.embed_queue.join()
            self.write_queue.join()

            self.stop_event.set()
            for _ in range(self.num_workers):
                self.embed_queue.put(None)
            self.write_queue.put(None)

            for t in embedding_threads:
                t.join(timeout=5)
            write_thread.join(timeout=5)

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
    import argparse

    parser = argparse.ArgumentParser(description="Vectorize products in Elasticsearch")
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--source-field', type=str, default=None)
    parser.add_argument('--destination-field', type=str, default=None)
    parser.add_argument('--poll-interval', type=int, default=None,
                        help='Polling interval in minutes (default: POLL_INTERVAL_MINUTES env var or 30)')

    args = parser.parse_args()

    limit = args.limit
    if limit is None:
        env_limit = os.environ.get('LIMIT', '')
        if env_limit.strip():
            limit = int(env_limit)

    poll_interval = (args.poll_interval or POLL_INTERVAL_MINUTES) * 60

    while True:
        vectorizer = ProductVectorizer(
            batch_size=args.batch_size or EMBEDDING_BATCH_SIZE,
            num_workers=args.workers or NUM_EMBEDDING_WORKERS,
            model=args.model or EMBEDDING_MODEL,
            source_field=args.source_field or SOURCE_FIELD,
            destination_field=args.destination_field or DESTINATION_FIELD,
        )
        vectorizer.run(limit=limit)

        logger.info(f"Sleeping for {poll_interval // 60} minutes before next check...")
        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
