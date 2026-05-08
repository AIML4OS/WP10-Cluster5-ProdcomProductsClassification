import os
import re
import sys
import logging
import time
import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from openai import OpenAI
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import context as ctx
import classif_qdrant_only
import classif_qdrant_openai
import classif_openai
import classif_openai_no_descr

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
PROGRAM_NAME = "classifai"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter("%(asctime)s - %(module)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S")
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setLevel(logging.DEBUG)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
now = datetime.now()
dateTime = now.strftime("%Y-%m-%d_%H_%M_%S")
LOG_FILE_NAME = "./logs/log_" + PROGRAM_NAME + "_" + dateTime + ".log"
fileHandler = logging.FileHandler(LOG_FILE_NAME)
fileHandler.setLevel(logging.DEBUG)
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ========================================================================
# ========================                        ========================
# ========================          MAIN          ========================
# ========================                        ========================
# ========================================================================

def main():
    load_dotenv('.env')
    ctx._initialize()
    os.makedirs(ctx.img_temp_folder, exist_ok=True)        
    ctx.logger = logging.getLogger("Main")

    # per far funzionare le chiamate a openai da rete istat decommentare le 2 righe sottostanti
    # os.environ['HTTP_PROXY'] = 'proxy.istat.it:3128'
    # os.environ['HTTPS_PROXY'] = 'proxy.istat.it:3128'

    # ==================================================
    # Inizializza il client Qdrant
    #==================================================
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if not qdrant_url:
        raise ValueError("Parameter error. Make sure the QDRANT_URL parameter is correctly configured in the .env file")
    if not qdrant_api_key:
        raise ValueError("Parameter error. Make sure the QDRANT_API_KEY parameter is correctly configured in the .env file")

    ctx.qdrant_client = QdrantClient(
        url=qdrant_url,
        api_key=qdrant_api_key
    )

    # ==================================================
    # Inizializza il client OpenAI
    # ==================================================
    ctx.openai_api_key = os.getenv("OPENAI_API_KEY")
    if not ctx.openai_api_key:
        raise ValueError("Chiave API non trovata. Assicurati che OPENAI_API_KEY sia nel file .env.")
    ctx.openai_client = OpenAI(api_key=ctx.openai_api_key)

    # ==================================================
    # Carico il file CSV la classificazione PRODCOM
    # ==================================================  
    ctx.prodcom_df = pd.read_csv(ctx.input_prodcom_2023_file, 
                              dtype=str,                # forza tutte le colonne a stringa
                              keep_default_na=False,    # non interpretare stringhe come NaN
                              na_filter=False,          # non convertire celle vuote in NaN
                              delimiter="\t")  
    ctx.sections = get_sections()

    # ==================================================
    # Carico il file CSV con i prodotti da classificare  
    # ==================================================  
    products_df = pd.read_csv(ctx.input_products_file, 
                              dtype=str,                # forza tutte le colonne a stringa
                              keep_default_na=False,    # non interpretare stringhe come NaN
                              na_filter=False,          # non convertire celle vuote in NaN
                              delimiter="\t")  
    
    # ==================================================
    # Creo e scrivo le righe nel file di output
    # ================================================== 
    output_file = get_generated_output_filepath(ctx.output_dir, ctx.output_file_name)
    with open(output_file, "w", encoding="utf-8") as f:
        
        # Scrivi intestazione
        header = ["name", "description", "id", "domain", "product_url", "product_img_url","source_file",
                  "code_type", "parent", "generated_description","code", "code_description"]
        f.write("\t".join(header) + "\n")

        # Iterazione sulle righe del DataFrame
        total_num_of_lines = products_df.shape[0]
        for i, (_, row) in enumerate(products_df.iterrows(), start=1):
            print("\n\n")
            print("============================================================================")
            ctx.logger.info(f"================>   Processing product/line number {i} of {total_num_of_lines}")
            time.sleep(ctx.sleep_secs_between_products)

            # ==================================================            
            # Classifica il prodotto
            # ==================================================
            row_dict = row.to_dict()
            get_product_generated_description_from_llm(row_dict)
            product_generated_description = row_dict["product_generated_description"]
            classif_strategy = ctx.classif_strategy

            if classif_strategy == 1:       # =====     Qdrant only     =====
                code, code_description, code_type, code_parent = classif_qdrant_only.get_qdrant_product_classification(product_generated_description)
            elif  classif_strategy == 2:    # =====     Qdrant then Openai      =====
                code, code_description, code_type, code_parent = classif_qdrant_openai.get_qdrant_openai_product_classification(product_generated_description, ctx.num_options_to_choose_from)
            elif  classif_strategy == 3:    # =====     Qpenai recursive on prodcom hierarchy    =====
                code, code_description, code_type, code_parent = classif_openai.get_openai_product_classification(product_generated_description)
            elif  classif_strategy == 4:    # =====     Qpenai recursive on prodcom hierarchy (NO generated description)   =====
                code, code_description, code_type, code_parent = classif_openai_no_descr.get_openai_product_classification_no_generated_descr(row)
            else:
                ctx.logger.info(f"Error: Invalid classification strategy selected!")
                raise ValueError("Valore non valido per classif_strategy (usa 1, 2, 3 o 4)")


            values = prepare_output_line_to_write(row_dict, code, code_description, code_type, code_parent)
            if values is not None:
                # Scrivi la riga originale più le colonne calcolate nel file di output
                f.write("\t".join(map(str, values)) + "\n")
                f.flush()

    ctx.qdrant_client.close()

def prepare_output_line_to_write(row_dict, code, code_description, code_type, code_parent):
    
    values = [
                row_dict["name"],
                str(row_dict["description"]).replace("\t", " ").replace("\n", " "),
                str(row_dict["id"]),
                row_dict["domain"],
                row_dict["product_url"],
                row_dict["product_img_url"],
                row_dict["source_file"],
                code_type,
                code_parent,
                row_dict["product_generated_description"].replace("\t", " ").replace("\n", " "),
                code.replace("\t", " ").replace("\n", " "),
                code_description.replace("\t", " ").replace("\n", " ")
            ]
    return values

def get_product_generated_description_from_llm(row):
    pre_prompt = """
            Dimmi in massimo 10 parole come classificheresti questo prodotto in base alla classificazione prodcom (non riportare il codice).

            Prodotto:

                    """  
    product_name = f"nome prodotto: {row['name']}"
    product_description = f"descrizione prodotto: {row['description']}"
    product_url = f"url prodotto: {row['product_url']}"
    product_textual_info_to_submit = get_product_textual_info_to_submit(product_name, product_description, product_url)
    user_prompt = pre_prompt + " \n " + product_textual_info_to_submit

    # Ottengo la descrizione generata dall'LLM per il prodotto specifico
    product_image_url = row['product_img_url']
    image_passing_mode = ctx.image_passing_mode
    image_content = ctx.get_image_content(product_image_url, image_passing_mode)
    if image_content is not None:
        #product_generated_description = ctx.invoke_chatgpt_api(user_prompt, image_content=image_content)
        product_generated_description = ctx.invoke_chatgpt_api_with_rate_limit(user_prompt, image_content=image_content)
    else:
        #product_generated_description = ctx.invoke_chatgpt_api(user_prompt)
        product_generated_description = ctx.invoke_chatgpt_api_with_rate_limit(user_prompt)
          
    logger.info("name: " + row['name'])
    logger.info("descr: " + row['description'])
    logger.info("url: " + row['product_url'])
    logger.info("image_url: " + product_image_url)
    logger.info("product_generated_description: " + product_generated_description)
    row["product_generated_description"] = clean_string(product_generated_description)
    return clean_string(product_generated_description)
    

def clean_string(s: str) -> str:
    """
    Rimuove tabulazioni, a capo, spazi multipli e spazi iniziali/finali.
    """
    # Rimuove tabulazioni e a capo
    s = re.sub(r"[\t\n\r]+", " ", s)
    
    # Sostituisce spazi multipli con uno singolo
    s = re.sub(r" +", " ", s)
    
    # Rimuove spazi iniziali e finali
    s = s.strip()
    
    return s


def get_product_textual_info_to_submit(product_name: str, product_description: str, product_url: str):
    if product_description.strip() != "" and len(product_description.strip()) >= 50:
        # esiste una descrizione probabilmente sensata del prodotto, evito di usare anche il nome
        return product_description + "\n" + product_url
    else:
        # prodotto senza descrizione, devo necessariamente usare il nome
        return product_name + "\n" + product_description + "\n" + product_url
        

def get_sections():
    sections_df = ctx.prodcom_df.drop_duplicates(subset=["section"]).reset_index(drop=True)
    sections_df = sections_df[["section","section_descr"]]
    my_dict = dict(zip(sections_df.iloc[:, 0], sections_df.iloc[:, 1]))
    return my_dict


def get_generated_output_filepath(output_dir:str, file_name:str, file_extension:str=".csv"):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    filename = f"{file_name}_{timestamp}{file_extension}"
    output_filepath = os.path.abspath(os.path.join(output_dir, filename))
    return output_filepath

if __name__ == "__main__":
    main()
