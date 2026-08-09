import logging as l
import atlassian as a
import markdownify as m
import langchain_chroma as lc
import langchain_ollama as lo
import langchain_text_splitters as lts
from langchain_core.documents import Document

logger = l.basicConfig(filename='create_vector_db.log', filemode='w', format='%(levelname)s - %(message)s', level=l.INFO)

CONFLUENCE_URL = "https://asurc.atlassian.net/wiki"
USERNAME = "<YOUR EMAIL HERE>" # OR None if using public scraping tools
API_KEY = "<YOUR ATLASSIAN API KEY HERE>"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
DB_DIRECTORY = './asu_rc'
COLLECTION_NAME = 'ASU-Research-Computing'

loader = a.Confluence(url=CONFLUENCE_URL, username=USERNAME, password=API_KEY, cloud=True)
pages = loader.get_all_pages_from_space(space='RC', start=0, limit=500, expand='body.view', status='current', content_type='page')
l.info(f'RETRIEVED {len(pages)} PAGES FROM THE DOCUMENTATION.')

split_on_headers = [('#', "Header 1"), ('##', "Header 2"), ('###', "Header 3")]
splitter = lts.MarkdownHeaderTextSplitter(headers_to_split_on=split_on_headers)
contents = []
for page in pages:
    clean_markdown = m.markdownify(page['body']['view']['value'], heading_style="ATX", code_language="bash")
    document = Document(page_content=clean_markdown, metadata={'title': page['title'], 'id': page['title'], 'source': CONFLUENCE_URL + page['_links']['webui']})
    splits = splitter.split_text(document.page_content)
    for split in splits:
        split.metadata.update(document.metadata)
        contents.append(split)
l.info(f'CREATED {len(contents)} SPLITS FROM RETIEVED PAGES.')

splitter = lts.RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
chunks = splitter.split_documents(contents)
l.info(f'CREATED {len(chunks)} CHUNKS FROM MARKDOWN SPLITS.')

embedding_model = lo.OllamaEmbeddings(model='nomic-embed-text')
vector_store = lc.Chroma.from_documents(documents=chunks, embedding=embedding_model, collection_name=COLLECTION_NAME, persist_directory=DB_DIRECTORY)
l.info('EMBEDDING CHUNKS COMPLETED.')
l.info(f'VECTOR DATABASE CREATED at {DB_DIRECTORY}')
