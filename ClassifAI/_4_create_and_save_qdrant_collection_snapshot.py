from qdrant_client import QdrantClient
import os
import requests
from dotenv import load_dotenv

# =======================================================================================
# Questo programma crea uno snapshot di una collection Qdrant e la salva su disco
# in modo da poterla riutilizzare in futuro al posto di ricalcolare da 0 gli embeddings
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
collection_name = "prodcom_2023_flattened"
QDRANT_NODES = [qdrant_url]
qdrant_snapshot_folder = "qdrant_snapshots"
# ===================

# create a snapshot to be downloaded
snapshot_urls = []
for node_url in QDRANT_NODES:
    node_client = QdrantClient(node_url, api_key=qdrant_api_key)
    snapshot_info = node_client.create_snapshot(collection_name=collection_name)
    snapshot_url = f"{node_url}/collections/{collection_name}/snapshots/{snapshot_info.name}"
    snapshot_urls.append(snapshot_url)


print(snapshot_urls[0]) # ho un solo nodo quindi un solo url

# Create a directory to store snapshots
os.makedirs(qdrant_snapshot_folder, exist_ok=True)

# Download snapshot on disk
local_snapshot_paths = []
for snapshot_url in snapshot_urls:
    snapshot_name = os.path.basename(snapshot_url)
    local_snapshot_path = os.path.join(qdrant_snapshot_folder, snapshot_name)

    response = requests.get(
        snapshot_url, headers={"api-key": qdrant_api_key}
    )
    with open(local_snapshot_path, "wb") as f:
        response.raise_for_status()
        f.write(response.content)

    local_snapshot_paths.append(local_snapshot_path)

print("snapshot downloaded on disk!")