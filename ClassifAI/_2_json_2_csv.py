import json
import csv
import os
import re
import glob

# ===========================================================================================================
# Questo programma legge tutti i file JSON contenuti in una cartella e riporta le informazioni presenti
# in ognuno sotto forma di riga in un file CSV che viene creato
# ===========================================================================================================

def normalizza_spazi(testo: str) -> str:
    """
    Sostituisce tabulazioni, a capo e spazi multipli con un singolo spazio.
    Rimuove anche eventuali spazi all'inizio e alla fine della stringa.
    
    Args:
        testo (str): La stringa di input da normalizzare.
    
    Returns:
        str: La stringa normalizzata.
    """
    if testo is None or testo == "NA":
        return ""
    # Sostituisce tab, newline e ritorni a capo con spazio
    testo = re.sub(r'[\t\n\r]+', ' ', testo)
    # Sostituisce spazi multipli con singolo spazio
    testo = re.sub(r'\s{2,}', ' ', testo)
    # Rimuove spazi iniziali e finali
    return testo.strip()


def json_2_csv(input_folder, output_file):

    # Campo di intestazione aggiornato con source_file
    fieldnames = ['name', 'description', 'id', 'domain', 'product_url', 'product_img_url', 'source_file']

    # Scrittura del file di output con tabulazione
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()

        # Trova tutti i file .json nella cartella
        json_files = glob.glob(os.path.join(input_folder, '*.json'))

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                product_id = data.get('ID', '')
                url = data.get('URL', '')
                product_link = data.get('Product_link', '')
                products = data.get('content', [])

                for product in products:
                    writer.writerow({
                        'name': normalizza_spazi(product.get('name', '')),
                        'description': normalizza_spazi(product.get('description', '')),
                        'id': product_id,
                        'domain': url,
                        'product_url': product_link,
                        'product_img_url': normalizza_spazi(product.get('image_url', '')),
                        'source_file': os.path.basename(json_file)
                    })

            except Exception as e:
                print(f"Errore nel file '{json_file}': {e}")


if __name__ == "__main__":
    
    input_folder = './json_output/'
    output_file = './output/test_products_extracted_from_urls.csv'
    
    json_2_csv(input_folder, output_file)
    print(f"File TSV '{output_file}' creato con successo.")