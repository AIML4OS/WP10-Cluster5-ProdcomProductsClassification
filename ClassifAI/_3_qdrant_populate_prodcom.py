import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient, AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from qdrant_client.http import models
from qdrant_client.http.models import CollectionStatus, UpdateStatus
import pandas as pd
from openai import OpenAI
from openai import AzureOpenAI
from typing import List
import uuid
import logging
import sys
from datetime import datetime

# ========================================================================================================
# Crea e popola una collection Qdrant avente come nome quello presente nella variabile "collection_name"
# La collection conterrà gli embedding delle voci presenti nel file "input_filepath"
# ========================================================================================================

collection_name = "prodcom_2023_flattened"
input_filepath = "./input/prodcom_2023_classification.csv"
output_parquet_file = "prodcom_2023_classification_embeddings.parquet"

PROGRAM_NAME = "populate_qdrant"
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
now = datetime.now()
dateTime = now.strftime("%Y-%m-%d_%H_%M_%S")
LOG_FILE_NAME = "./logs/log_" + PROGRAM_NAME + "_" + dateTime + ".log"
fileHandler = logging.FileHandler(LOG_FILE_NAME)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)


load_dotenv('.env')
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Chiave API non trovata. Assicurati che OPENAI_API_KEY sia nel file .env.")
openai_client = OpenAI()

# configuro i parametri per l'istanza privata Istat
ISTATGPT_API_KEY = os.getenv("ISTAT_OPENAI_API_KEY")
ISTATGPT_ENDPOINT_URL = os.getenv("ISTAT_ENDPOINT_URL")
ISTATGPT_DEPLOYMENT_NAME = "gpt-4.1" 
ISTATGPT_API_VERSION = os.getenv("ISTAT_API_VERSION")
istatgpt_embedding_client = AzureOpenAI(
    api_key=ISTATGPT_API_KEY,
    azure_endpoint=ISTATGPT_ENDPOINT_URL,    # URL base dell'istanza
    api_version=ISTATGPT_API_VERSION         # Versione API richiesta da Azure
    )

def get_text_embedding(text: str, openai_client: OpenAI= openai_client, model: str = "text-embedding-3-large") -> list:
    """
    Get the vector representation of the input text using the specified OpenAI embedding model.

    Args:
        openai_client (OpenAI): An instance of the OpenAI client.
        text (str): The input text to be embedded.
        model (str, optional): The name of the OpenAI embedding model to use. Defaults to "text-embedding-3-large".

    Returns:
        list: The vector representation of the input text as a list of floats.

    Raises:
        OpenAIError: If an error occurs during the API call.
    """
    try:
        embedding = openai_client.embeddings.create(
            input=text,
            model=model
        ).data[0].embedding
        return embedding
    except openai_client.OpenAIError as e:
        raise e


def get_text_embedding_azure(text: str, istatgpt_embedding_client: AzureOpenAI= istatgpt_embedding_client, model: str = "text-embedding-3-large") -> list:
    """
    Get the vector representation of the input text using the specified OpenAI embedding model.

    Args:
        istatgpt_embedding_client (AzureOpenAI): An instance of the AzureOpenAI client.
        text (str): The input text to be embedded.
        model (str, optional): The name of the OpenAI embedding model to use. Defaults to "text-embedding-3-large".

    Returns:
        list: The vector representation of the input text as a list of floats.

    Raises:
        OpenAIError: If an error occurs during the API call.
    """
    try:
        embedding = istatgpt_embedding_client.embeddings.create(
            input=text,
            model=model
        ).data[0].embedding
        return embedding
    except openai_client.OpenAIError as e:
        raise e


def add_data_to_collection(data: List[dict], qdrant_client: QdrantClient, collection_name: str):
    """
    Inserts data into the Qdrant vector database.

    Args:
        data (List[dict]): A list of dictionaries containing the data to be inserted.
            Each dictionary should have the following keys:
            - 'code'
            - 'description'
            - 'type'
            - 'parent'
        qdrant_client (QdrantClient): An instance of the QdrantClient. Defaults to qdrant_client.
        collection_name (str): The name of the collection in which to insert the data. Defaults to "arxiv_chunks".

    Returns:
        None
    """
    tot_num = len(data)
    # get the relevent data from the input dictionary
    for num, item in enumerate(data, start=1):
        points = []
        entry_id = str(uuid.uuid4())
        code = item.get("prodcom")
        description = item.get("cpa6_prodcom_descr")
        type = "prodcom"
        parent = "0"
        logger.info("processing record " + str(num) + " / " + str(tot_num) + " having code " + str(code))

        # get the vector embeddings for the description
        #description_vector = get_text_embedding(description)
        description_vector = get_text_embedding_azure(description)

        # create a dictionary with the vector embeddings
        vector_dict = {"description": description_vector}

        # create a dictionary with the payload data
        payload = {
            "entry_id": entry_id,
            "description": description,
            "code": code,
            "type": type,
            "parent": parent,
        }

        # create a PointStruct object and append it to the list of points
        point = PointStruct(id=entry_id, vector=vector_dict, payload=payload)
        points.append(point)

        operation_info = qdrant_client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points)

        if operation_info.status == UpdateStatus.COMPLETED:
            logger.info("Data inserted successfully!")
        else:
            logger.info("Failed to insert data")
        
        


def create_collection(collection_name):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "description": VectorParams(size=3072, distance=models.Distance.COSINE) # 3072 is the native embedding size of the text-embedding-3-large model
        }
    )


if __name__ == "__main__":
    # ==================================================
    # Configura il client per accedere al DB Qdrant
    #==================================================
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise ValueError("Parameter error. Make sure the QDRANT_URL parameter is correctly configured in the .env file")
    if not qdrant_api_key:
        raise ValueError("Parameter error. Make sure the QDRANT_API_KEY parameter is correctly configured in the .env file")

    qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    
    
    # qdrant_client.delete_collection(collection_name=collection_name)

    # ===========================================================
    # Crea se non esiste la collection in cui inserire i vettori
    # ===========================================================
    collections = qdrant_client.get_collections()
    if len(collections.collections) == 0:
        create_collection(collection_name)
    else:
        collection_already_exists = False
        for collection in collections.collections:
            if collections.collections[0].name == collection_name:
                collection_already_exists = True
                break
        if not collection_already_exists:
            create_collection(collection_name)
    collections = qdrant_client.get_collections()


    # =================================================================
    # Carica la classificazione da inserire nel DB
    # =================================================================   
    qdrant_points_df = pd.read_csv(input_filepath,
                                   sep="\t",
                                   encoding="utf8",
                                   low_memory=False,
                                   on_bad_lines="warn")
    
    # creo la colonna "cpa6_prodcom_descr" concatenando cpa6_descr con prodcom_descr
    qdrant_points_df["cpa6_prodcom_descr"] = qdrant_points_df["cpa6_descr"].astype(str) + " - " + qdrant_points_df["prodcom_descr"].astype(str)

    qdrant_points_as_dict_list = qdrant_points_df.to_dict(orient='records')


    # =================================================================
    # Caricamento nel DB in forma embeddata (tramite ChatGPT) dei dati
    # =================================================================

    add_data_to_collection(qdrant_points_as_dict_list, qdrant_client, collection_name)


    qdrant_client.close()

