from src.helper import load_pdf_files, text_split, download_hugging_face_embedding
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os
import logging

load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ['PINECONE_API_KEY']=PINECONE_API_KEY

logging.info("Extracted Data Starting.....")
extracted_data=load_pdf_files(data='data/')
text_chunks=text_split(extracted_data)
logging.info("Total Chunks:{text_chunks}")
embeddings=download_hugging_face_embedding()
logging.info("{embeddings}")
pc=Pinecone(api_key=PINECONE_API_KEY)

index_name="medical-bot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

#Embed each chunk and upsert the embeddings into your Pinecode index.
docsearch=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)