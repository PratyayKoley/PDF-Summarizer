from rq import Queue
from redis import Redis
from models.summarizer import process_pdf_and_summarize

redis_conn = Redis()
q = Queue(connection=redis_conn)

def enqueue_job(file_path):
    q.enqueue(process_pdf_and_summarize, file_path)