import requests
import mimetypes
import base64
import io
import os
import sys
import configparser
import time, random
from PIL import Image
from urllib.parse import unquote
from urllib.parse import urlparse


# =============================================================================
# INPUT VARS
# =============================================================================
img_temp_folder = ""
input_products_file = ""
output_dir = ""
output_file_name = ""
input_prodcom_2023_file = ""
image_passing_mode = 1 # passa direttamente l'url all'LLM
classif_strategy = 2 # Qdrant then Openai

# =============================================================================
# GLOBAL VARS
# =============================================================================
qdrant_client = None
qdrant_collection_name = ""
openai_client = None
openai_model = ""
openai_embedding_model = ""
default_image_content = {"type": "text", "text": ""} # da usare quando non viene fornita una immagine
default_openai_model_temperature = None
prodcom_df = None
sections = {}
logger = None
num_options_to_choose_from = None # num matches da far elaborare a qdrant e poi presentare a openai
sleep_secs_between_products = None
_initialized = False

MAX_PRODUCT_DESCRIPTION_CHARS_FOR_SELECTION = 3000
MAX_OPTION_DESCRIPTION_CHARS_FOR_SELECTION = 500
MAX_OPTIONS_FOR_SELECTION_PROMPT = 30
MAX_USER_PROMPT_CHARS = 12000


def truncate_text(text: str, max_chars: int) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _initialize():
    global _initialized
    if _initialized:
        return  
     
    global input_products_file, output_dir, output_file_name, input_prodcom_2023_file
    global qdrant_collection_name, num_options_to_choose_from
    global openai_model, openai_embedding_model, default_openai_model_temperature
    global img_temp_folder, image_passing_mode
    global sleep_secs_between_products, classif_strategy
    config_file = "classifai_cfg.ini"
    config = load_config(config_file)
    
    if 'paths' in config:
        input_products_file = config['paths'].get('input_products_file', '')
        output_dir = config['paths'].get('output_dir', '')
        output_file_name = config['paths'].get('output_file_name', '')
        input_prodcom_2023_file = config['paths'].get('input_prodcom_2023_file', "")
    else:
        print("Section 'paths' not found in the \"classifai_cfg.ini\" configuration file.")
        sys.exit(1)

    if 'qdrant' in config:
        qdrant_collection_name = config['qdrant'].get('qdrant_collection_name', '')
        num_options_to_choose_from = config['qdrant'].getint('num_options_to_choose_from', 5)
    else:
        print("Section 'qdrant' not found in the \"classifai_cfg.ini\" configuration file.")
        sys.exit(1)

    if 'openai' in config:
        openai_model = config['openai'].get('openai_model', '')
        openai_embedding_model = config['openai'].get('openai_embedding_model', '')
        default_openai_model_temperature = config['openai'].getfloat('default_openai_model_temperature', 1)
    else:
        print("Section 'openai' not found in the \"classifai_cfg.ini\" configuration file.")
        sys.exit(1)

    if 'img' in config:
        img_temp_folder = config['img'].get('img_temp_folder', '')
        image_passing_mode = config['img'].getint('image_passing_mode', 1)
    else:
        print("Section 'img' not found in the \"classifai_cfg.ini\" configuration file.")
        sys.exit(1)

    if 'behaviour' in config:
        sleep_secs_between_products = config['behaviour'].getfloat('sleep_secs_between_products', 0.1)
        classif_strategy = config['behaviour'].getint('classif_strategy', 2)
    else:
        print("Section 'behaviour' not found in the \"classifai_cfg.ini\" configuration file.")
        sys.exit(1)

    _initialized = True


def get_llm_best_selection(options: dict, product_generated_description:str, model=openai_model, temperature=default_openai_model_temperature):
    if len(options) == 1:
        key = next(iter(options))
        return str(key)

    short_product_description = truncate_text(
        str(product_generated_description),
        MAX_PRODUCT_DESCRIPTION_CHARS_FOR_SELECTION,
    )

    option_lines = []
    for code, description in list(options.items())[:MAX_OPTIONS_FOR_SELECTION_PROMPT]:
        short_description = truncate_text(
            str(description),
            MAX_OPTION_DESCRIPTION_CHARS_FOR_SELECTION,
        )
        option_lines.append(f"{code}: {short_description}")

    prompt = "Considera la seguente descrizione prodotto: \n"
    prompt = prompt + short_product_description + "\n\n"
    prompt = prompt + "Scegli tra le seguenti voci quella più appropriata per classificare il prodotto:\n"
    prompt = prompt + "\n".join(option_lines)
    prompt = prompt + "\n"
    prompt = prompt + "Restituisci solo il codice associato alla voce\n"
    #best_selection = invoke_chatgpt_api(prompt)
    best_selection = invoke_chatgpt_api_with_rate_limit(prompt)
    return best_selection


def invoke_chatgpt_api_with_rate_limit(user_prompt, image_content=default_image_content, model=None, temperature=default_openai_model_temperature, max_retries=5):
    
    if model == None:
        model = openai_model
    if temperature == None:
        temperature = default_openai_model_temperature

    retries = 0
    backoff = 1
    user_prompt = truncate_text(str(user_prompt), MAX_USER_PROMPT_CHARS)

    while True:
        try:
            # Usa la versione raw per accedere agli header
            raw_response = openai_client.chat.completions.with_raw_response.create(
                model=model,
                messages=[
                    {
                        "role": "system", 
                        "content": "Sei un assistente esperto nel classificare prodotti usando la classificazione PRODCOM."
                    },
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": user_prompt},
                            image_content,
                        ],
                    }
                ],
                temperature=temperature,
                seed=42
            )
            
            # Ottieni la risposta JSON
            response = raw_response.parse()

            # ---- Header del rate limit ----
            headers = raw_response.headers
            remaining_tokens = int(headers.get("x-ratelimit-remaining-tokens", 1))
            reset_tokens = float(to_seconds(keep_digits(headers.get("x-ratelimit-reset-tokens", 0))))

            logger.info(f"Token rimasti: {remaining_tokens}, reset in {reset_tokens:.3f}s - {headers.get('x-ratelimit-reset-tokens', 0)}")

            if remaining_tokens <= 0:
                wait_time = max(reset_tokens, 1.0)
                logger.info(f"Quota token esaurita. Attendo {wait_time:.2f}s...")
                time.sleep(wait_time)

            logger.debug(f"total token: {response.usage.total_tokens}")
            return response.choices[0].message.content

        except Exception as e:
            if "429" in str(e) and retries < max_retries:
                wait_time = backoff + random.uniform(0, 0.5)
                logger.info(f"Rate limit superato, attendo {wait_time:.2f}s e riprovo...")
                time.sleep(wait_time)
                backoff = backoff * 2
                retries = retries + 1
            else:
                logger.exception(f"Errore durante la chiamata alle API: {e}")
                return f"xxx ERRORE durante la chiamata API xxx {e}"

def keep_digits(s: str) -> str:
    # rimuove tutti i caratteri non numerici da una stringa
    return "".join(ch for ch in s if ch.isdigit())


def to_seconds(num_str: str) -> str:
    num_str = num_str.strip()
    
    # Se input contiene un punto assumo che sia già espresso in secondi
    if "." in num_str:
        return f"{float(num_str):.3f}"
    
    # Altrimenti se input è un intero in millisecondi lo converto in secondi
    ms = int(num_str)
    seconds = ms / 1000
    return f"{seconds:.3f}"


def invoke_chatgpt_api_standard(user_prompt, image_content=default_image_content, model=None, temperature=None):

    if model == None:
        model = openai_model
    if temperature == None:
        temperature = default_openai_model_temperature

    try:
        response = openai_client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system", 
                    "content": "Sei un assistente esperto nel classificare prodotti usando la classificazione PRODCOM."
                },
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        image_content,
                    ],
                }
            ],
            temperature=temperature,
            seed=42
        )
        logger.info(f"total token: {response.usage.total_tokens}")
        return response.choices[0].message.content
    except Exception as e:
        return f"Errore: {e}"
    

def get_text_embedding(text: str, model=None) -> list:
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
    if model is None:
        model = openai_embedding_model

    try:
        embedding = openai_client.embeddings.create(
            input=text,
            model=model
        ).data[0].embedding
        return embedding
    except openai_client.OpenAIError as e:
        raise e


def get_image_content(url: str, mode: int):
    """
    Restituisce un contenuto immagine compatibile con le API OpenAI.
    
    mode:
      1 -> usa direttamente l'URL pubblico
      2 -> salva su disco e poi carica in base64
      3 -> converte in base64 direttamente da requests.get (senza file)
    """

    if not is_valid_url(url):
        logger.warn(f"is_valid_url returned False for image url: {url}")
        return None
    
    if not is_image_url(url):
        logger.warn(f"is_image_url returned False for image url: {url}")
        return None

    # Decodifica URL
    url = unquote(url)
    # Headers per Reddit o siti che bloccano bot
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    if mode == 1:
        # --- Modalità 1: URL pubblico diretto
        return {
            "type": "image_url",
            "image_url": {"url": url}
        }

    elif mode == 2:
        # --- Modalità 2: salva su disco e poi codifica con conversione PNG
        try:
            filename = f"{img_temp_folder}/temp_image"
            response = requests.get(url, headers=headers, verify=False)
            if response.status_code != 200:
                logger.error(f"Errore download immagine: {response.status_code}")
                #raise ValueError(f"Errore download immagine: {response.status_code}")
                return None

            # Determino estensione dal Content-Type
            ext = response.headers.get("Content-Type", "image/png").split("/")[-1]
            filename_ext = f"{filename}.{ext}"

            with open(filename_ext, "wb") as f:
                f.write(response.content)

            # Faccio l'eventuale resize, apro con pillow e converto in PNG
            resized_filename = f"{img_temp_folder}/resized_temp_image.png"
            resize_to_max_400(filename_ext, resized_filename)
            image = Image.open(filename_ext)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Apro con Pillow e converto in PNG
            # image = Image.open(filename_ext)
            # buffer = io.BytesIO()
            # image.save(buffer, format="PNG")
            # image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            # Elimina il file temporaneo
            os.remove(filename_ext)
            os.remove(resized_filename)
            
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        except Exception as e:
            logger.error(f"Errore durante l'acquisizione dell'immagine {url}:", e)
            return None 
    elif mode == 3:
        # --- Modalità 3: senza file temporaneo, con Pillow per essere sicuri di passare sempre file png in uscita
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError(f"Errore download immagine: {response.status_code}")
        
        image = Image.open(io.BytesIO(response.content))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_base64}"}
        }

    else:
        raise ValueError("Valore non valido per image_passing_mode (usa 1, 2 o 3)")
    

def is_valid_url(url: str) -> bool:
    """Controlla se la stringa passata rappresenta un URL valido."""
    try:
        result = urlparse(url)
        # Deve avere almeno schema (http/https) e netloc (dominio)
        return all([result.scheme in ("http", "https"), result.netloc])
    except Exception:
        return False
    

def is_image_url(url: str) -> bool:
    try:
        # prima verifica semplice: estensione del file
        mime_type, _ = mimetypes.guess_type(url)
        if mime_type and mime_type.startswith("image/"):
            return True

        # verifica robusta: controllo degli header HTTP
        response = requests.head(url, allow_redirects=True, timeout=5)
        content_type = response.headers.get("Content-Type", "")
        return content_type.startswith("image/")
    except Exception as e:
        logger.error(f"Errore durante la verifica del link immagine {url}:", e)
        return False 
    

def resize_to_max_400(input_path: str, output_path: str):
    # Apri l'immagine
    img = Image.open(input_path)

    # Trova il lato più lungo
    max_side = max(img.width, img.height)

    # Se il lato più lungo è già <= 400, non serve ridimensionare
    if max_side <= 400:
        img.save(output_path, format="PNG")
        return

    # Calcola il fattore di ridimensionamento
    scale = 400 / max_side
    new_width = int(img.width * scale)
    new_height = int(img.height * scale)

    # Ridimensiona mantenendo il rapporto di forma
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)

    # Salva come PNG
    resized_img.save(output_path, format="PNG")


def get_prodcom_description_by_code(guessed_prodcom):
    #print("guessed_prodcom value to search = " + str(guessed_prodcom))
    prodcom_descr_by_code = prodcom_df.loc[prodcom_df["prodcom"] == str(guessed_prodcom), "prodcom_descr"].values
    description = ""
    try:
        description = prodcom_descr_by_code[0] 
    except IndexError as e:
        logger.warning(f"WARNING: prodcom code {guessed_prodcom} not found in {input_prodcom_2023_file}!", e)
    return description


def get_divisions_by_section(section):
    division_df = prodcom_df[prodcom_df["section"] == str(section)]
    division_df = division_df[["division","division_descr"]]
    division_df = division_df.drop_duplicates(subset=["division"]).reset_index(drop=True)
    my_dict = dict(zip(division_df.iloc[:, 0], division_df.iloc[:, 1]))
    return my_dict


def get_groups_by_division(division):
    group_df = prodcom_df[prodcom_df["division"] == str(division)]
    group_df = group_df[["group","group_descr"]]
    group_df = group_df.drop_duplicates(subset=["group"]).reset_index(drop=True)
    my_dict = dict(zip(group_df.iloc[:, 0], group_df.iloc[:, 1]))
    return my_dict


def get_classes_by_group(group):
    class_df = prodcom_df[prodcom_df["group"] == str(group)]
    class_df = class_df[["class","class_descr"]]
    class_df = class_df.drop_duplicates(subset=["class"]).reset_index(drop=True)
    my_dict = dict(zip(class_df.iloc[:, 0], class_df.iloc[:, 1]))
    return my_dict


def get_cpa5_by_class(classe):
    cpa5_df = prodcom_df[prodcom_df["class"] == str(classe)]
    cpa5_df = cpa5_df[["cpa5","cpa5_descr"]]
    cpa5_df = cpa5_df.drop_duplicates(subset=["cpa5"]).reset_index(drop=True)
    my_dict = dict(zip(cpa5_df.iloc[:, 0], cpa5_df.iloc[:, 1]))
    return my_dict


def get_cpa6_by_cpa5(cpa5):
    cpa6_df = prodcom_df[prodcom_df["cpa5"] == str(cpa5)]
    cpa6_df = cpa6_df[["cpa6","cpa6_descr"]]
    cpa6_df = cpa6_df.drop_duplicates(subset=["cpa6"]).reset_index(drop=True)
    my_dict = dict(zip(cpa6_df.iloc[:, 0], cpa6_df.iloc[:, 1]))
    return my_dict
    
    
def get_prodcoms_by_cpa6(cpa6):
    prodcoms_df = prodcom_df[prodcom_df["cpa6"] == str(cpa6)]
    prodcoms_df = prodcoms_df[["prodcom","prodcom_descr"]]
    prodcoms_df = prodcoms_df.drop_duplicates(subset=["prodcom"]).reset_index(drop=True)
    my_dict = dict(zip(prodcoms_df.iloc[:, 0], prodcoms_df.iloc[:, 1]))
    return my_dict


def load_config(config_file):
    config = configparser.ConfigParser()
    try:
        with open(config_file, 'r') as file:
            config.read_file(file)
    except (FileNotFoundError, configparser.Error) as e:
        print(f"Error loading configuration file: {e}")
        sys.exit(1)
    return config