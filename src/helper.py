from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


# Extract Data from The PDF Files

def load_pdf_files(data):
    loader= DirectoryLoader(data,
                            glob="*.pdf",
                            loader_cls=PyPDFLoader)
    
    documents=loader.load()

    return documents



#Split the Data into Text Chunks

def text_split(extracted_data):
    text_splittler=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunk=text_splittler.split_documents(extracted_data)
    return text_chunk

#Download the Embedding from HuggingFace

def download_hugging_face_embedding():
    embeddings=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings