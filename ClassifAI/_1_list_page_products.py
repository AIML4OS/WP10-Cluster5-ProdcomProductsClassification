from scrapegraphai.graphs import SmartScraperGraph
import json
import csv
import nest_asyncio  # Import nest_asyncio module for asynchronous operations
nest_asyncio.apply()  # Apply nest_asyncio to resolve any issues with asyncio event loop
from dotenv import load_dotenv
import os

# ===========================================================================================================
# Questo programma legge un file CSV contenente una lista di URL aziendali in cui sono presenti prodotti
# per ogni riga/url estrae i prodotti presenti e salva nome e descrizione di ognuno in un file json
# ===========================================================================================================


# ========================================================================
# =====         INPUT
# ========================================================================
input_file_csv = "./input/url_containing_products.csv"
output_subfolder = "json_output"



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Chiave API non trovata. Assicurati che OPENAI_API_KEY sia nel file .env.")


my_custom_prompt = """Find all the products in the webpage with their descriptions (if available).
    Return the results exclusively in JSON format, with each found product represented as an object with three fields:
    - "name": the name of the product
    - "description": full description of the product (if available)
    - "image_url": absolute url of the image product (if available)
    If the page does not contain any products or descriptions, return an empty JSON array: []."""

def process_csv(file_input):
    """
    Reads a CSV file and for each line calls the get_products function.
    """
    try:
        with open(file_input, mode="r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file, delimiter="\t")  # Usa DictReader per leggere il CSV come un dizionario
            for line_num, row in enumerate(reader):
                get_products(row, line_num+2)  # line_num+2 assume che la prima riga riga utile del file csv sia la secoinda
    except FileNotFoundError:
        print(f"Error: file {file_input} not found.")
    except Exception as e:
        print(f"Error: {e}")


def get_products(row, line_num):

    row_id = row["ID"]
    row_dict = [
        f"ID: {row['ID']}\n",
        f"URL: {row['URL']}\n",
        f"Product Link: {row['Product_link']}\n"
    ]


    # OPENAI Configuration dictionary for the graph
    graph_config = {
        "llm": {
            "api_key": openai_api_key,  # Specify the output format as JSON
            "model": "openai/gpt-4o-mini",
        },
        "verbose": True,  # Enable verbose mode for debugging purposes
        "headless": False,
        "timeout": 30,
        "browser_args": [
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
        ],
    }

    smart_scraper_graph = SmartScraperGraph(
        prompt=my_custom_prompt,
        source=row['Product_link'],
        config=graph_config
    )

    try:
        result = smart_scraper_graph.run() # Run the SmartScraperGraph and store the result

        # json custom enrichment
        result["ID"] = row['ID']
        result["URL"] = row['URL']
        result["Product_link"] = row['Product_link']

        output = json.dumps(result, indent=2)  # Convert result to JSON format with indentation
        line_list = output.split("\n")  # Split the JSON string into lines

        print(row['Product_link'])
        print(line_list)

        file_name = f"./{output_subfolder}/{row_id}_at_row_{line_num}.json"
        with open(file_name, "w", encoding='utf-8') as file:
            for line in line_list:
                # print(line)
                file.writelines(line + "\n")
        print(f"File created: {file_name}")
    except Exception as e:
        print(e)


process_csv(input_file_csv)