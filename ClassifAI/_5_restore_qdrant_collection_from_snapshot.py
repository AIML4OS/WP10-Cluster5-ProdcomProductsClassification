from qdrant_client import QdrantClient
import os
import requests
from dotenv import load_dotenv

# =======================================================================================
# Questo programma effettua il restore di uno snapshot di una collection Qdrant
# evitandomi di ricalcolare da 0 gli embeddings
#
# Se non funziona il programma potrebbe essere necessario aggiornare i parametri relativi
# a Qdrant (url e apikey) nel file .env (vanno rigenerati dopo un certo numero di giorni di inutilizzo)
#
# The Restore function is typically used to move a collection to a different Qdrant instance, 
# but we can also use it to create a new collection on the same cluster; in this case it is just 
# going to have a different name, eg: test_collection_import. 
# IMPORTANT: We do not need to create a collection first, as it is going to be created automatically.
#
# https://qdrant.tech/documentation/database-tutorials/create-snapshot/
# =======================================================================================


# load qdrant info (url and apikey)
load_dotenv('.env')
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


# ===================
# input parameters
collection_name = "prodcom_2023_flattened" # collection to be restored (if empty or corrupted) or created and restored
QDRANT_NODES = [qdrant_url] # in the free tier I just have 1 node
local_snapshot_path = "./qdrant_snapshots/prodcom_2023_flattened-8762970021422419-2025-08-26-09-16-43.snapshot"
local_snapshot_paths = [local_snapshot_path]
# ===================

for node_url, snapshot_path in zip(QDRANT_NODES, local_snapshot_paths):
    snapshot_name = os.path.basename(snapshot_path)
    requests.post(
        f"{node_url}/collections/{collection_name}/snapshots/upload?priority=snapshot",
        headers={
            "api-key": qdrant_api_key,
        },
        files={"snapshot": (snapshot_name, open(snapshot_path, "rb"))},
    )


print("snapshot restored online!")

