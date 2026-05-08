# ClassifAI - Automatic Product Classification System

ClassifAI is an automatic product classification system that uses **Qdrant** (vector search) and **OpenAI** to classify products according to the **Prodcom classification** (statistical product classification).

## Overview

The system processes a CSV file containing products (with name, description, image, URL) and automatically generates:
- A synthetic product description via LLM
- A Prodcom classification code
- The classification code description

It supports **4 different classification strategies** to adapt to various use cases.

---

## Key Features

- ✅ **Semantic classification**: uses vector embeddings for semantic search
- ✅ **Multimodal support**: processes text, images, and product URLs
- ✅ **Hierarchical classification**: supports Prodcom hierarchical structure
- ✅ **Complete logging**: records all operations to file and console
- ✅ **Flexible configuration**: adjustable parameters via INI and .env files
- ✅ **4 classification strategies**: from simple to sophisticated
- ✅ **Batch processing**: processes multiple products sequentially

---

## Prerequisites

- **Python 3.10+**
- **Conda** (recommended) or pip
- **Qdrant access** (running instance)
- **OpenAI API key**

---

## Installation

### 1. Create Conda Environment

```bash
conda create --name classifai python=3.10
conda activate classifai
```

### 2. Install Dependencies

All project dependencies are pinned in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

If you need to update the requirements file after installing new packages:

```bash
pip freeze > requirements.txt
```

**Manual installation** (if needed):
```bash
pip install python-dotenv
pip install qdrant-client==1.16.1  # ⚠️ Version 1.16.1 or later (uses query_points API)
pip install openai
pip install pandas
pip install requests
pip install sentence-transformers
pip install faiss-cpu
pip install certifi
pip install Pillow
pip install nest_asyncio
pip install scrapegraphai
pip install playwright
playwright install chromium
```

**Additional setup for ScrapeGraphAI**:
After installing `scrapegraphai`, the Playwright browser timeout has been patched internally to support longer load times (up to 180 seconds). This prevents timeouts when scraping slower websites.

**Note for ISTAT users**: if you are behind a proxy, use:
```bash
pip install --proxy http://proxy.istat.it:8080 -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file in the project root:

```ini
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
```

---

## Dependency Management

### Generating `requirements.txt`

To generate or update the requirements file with all installed packages and their versions:

```bash
# Activate the Conda environment
conda activate classifai

# Generate requirements.txt
pip freeze > requirements.txt

# Verify the file was created
cat requirements.txt
```

This ensures reproducibility and makes it easy to set up the project on other machines.

### Installing from `requirements.txt`

```bash
pip install -r requirements.txt
```

---

## Configuration File Management

### Environment Variables (`.env`)

Create a `.env` file in the project root to store sensitive credentials:

```ini
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

**⚠️ Important**: Never commit the `.env` file to version control. It contains sensitive API keys.

### Configuration File: `classifai_cfg.ini`

The main configuration file controls system behavior:

#### `[paths]` - File Paths
```ini
input_products_file = ./output/extracted_products.csv
output_dir = ./output
output_file_name = classified_products_classifai
input_prodcom_2023_file = ./input/prodcom_2023_classification.csv
```

#### `[qdrant]` - Qdrant Configuration
```ini
qdrant_collection_name = prodcom_2023_flattened
num_options_to_choose_from = 5  # Top-K results from Qdrant
```

#### `[openai]` - OpenAI Configuration
```ini
openai_model = gpt-4o-mini
openai_embedding_model = text-embedding-3-large
default_openai_model_temperature = 0.0  # 0.0 = deterministic
```

#### `[img]` - Image Handling
```ini
img_temp_folder = ./img_temp
image_passing_mode = 2  # 1=direct URL, 2=base64 file, 3=PIL in-memory
```

#### `[behaviour]` - Behavior
```ini
sleep_secs_between_products = 0.1
classif_strategy = 2  # See below
```

---

## Classification Strategies

The system supports 4 classification strategies selectable via `classif_strategy`:

| Strategy | ID | Description |
|----------|-----|-------------|
| **Qdrant only** | 1 | Searches for the most similar code in Qdrant (fast, less accurate) |
| **Qdrant + OpenAI** | 2 | Qdrant returns top-K, OpenAI chooses the best (recommended) |
| **OpenAI recursive** | 3 | OpenAI recursively navigates Prodcom hierarchy (slow, more accurate) |
| **OpenAI recursive (no descr)** | 4 | Like 3 but without LLM-generated description |

**Recommendation**: Strategies 2-3 offer the best balance between quality and speed.

---

## Project Structure

```
ClassifAI/
├── _7_classifai.py                    # Main script
├── context.py                          # Global configuration and utilities
├── classif_qdrant_only.py             # Qdrant-only classifier
├── classif_qdrant_openai.py           # Qdrant+OpenAI classifier
├── classif_openai.py                  # OpenAI recursive classifier
├── classif_openai_no_descr.py         # OpenAI classifier (no desc)
├── classifai_cfg.ini                  # Configuration file
├── .env                                # Environment variables
├── input/                              # Input folder (product CSVs)
├── output/                             # Output folder (results)
├── logs/                               # Log files
├── img_temp/                           # Temporary images
└── qdrant_snapshots/                  # Qdrant collection backups
```

---

## Usage

### Main Execution

```bash
conda activate classifai
python _7_classifai.py
```

### Input File Format

The input CSV file must contain these columns:
```
name                (str) - Product name
description         (str) - Product description
id                  (str) - Unique ID
domain              (str) - Domain/category
product_url         (str) - Product URL
product_img_url     (str) - Product image URL
source_file         (str) - Source file
```

**Delimiter**: TAB (`\t`)

### Output

The output file contains:
```
name | description | id | domain | product_url | product_img_url | source_file
| code_type | parent | generated_description | code | code_description
```

---

## Data Preparation Scripts

The project includes utility scripts for data preprocessing and collection management:

| Script | Purpose |
|--------|---------|
| `_1_list_page_products.py` | Scrapes products from URLs using ScrapeGraphAI and saves to JSON |
| `_2_json_2_csv.py` | Converts JSON scraping results into CSV format |

---

## Qdrant Collection Management

### Available Utility Scripts

| Script | Function |
|--------|----------|
| `_3_qdrant_populate_prodcom.py` | Populates Qdrant with Prodcom classification |
| `_4_create_and_save_qdrant_collection_snapshot.py` | Creates a collection backup |
| `_5_restore_qdrant_collection_from_snapshot.py` | Restores from backup |

---

## Logging

The system automatically generates log files with the following naming scheme:

```
logs/log_classifai_YYYY-MM-DD_HH_MM_SS.log
```

### Log Levels
- `INFO`: General execution information
- `WARNING`: Warnings (e.g., proxy, API rate limit)
- `ERROR`: Errors that block execution
- `EXCEPTION`: Exceptions with full stack trace

Logs are written to **both console and file** simultaneously.

---

## Advanced Configuration

### Qdrant Collection Snapshots

The system can backup and restore Qdrant collections using snapshots:

**Create a snapshot**:
```bash
python _4_create_and_save_qdrant_collection_snapshot.py
```

**Restore from snapshot**:
```bash
python _5_restore_qdrant_collection_from_snapshot.py
```

Snapshots are saved in `qdrant_snapshots/` folder and can be transferred to other Qdrant instances.

### Custom Prompt Templates

You can customize the system prompt and user prompts by editing:
- System prompt in `context.py` (line ~125)
- User prompt construction in `classif_*.py` files

Modify these to adapt to your specific classification needs.

---

## Troubleshooting

### Issue: `AttributeError: 'QdrantClient' object has no attribute 'search'`
**Cause**: Using an outdated version of `qdrant-client` (< 1.16.0).  
**Solution**: Upgrade to version 1.16.1 or later:
```bash
pip install --upgrade qdrant-client
```
Starting from version 1.16.0, the API changed from `.search()` to `.query_points()`. This project is compatible with both old and new versions.

### Issue: "QDRANT_URL not found"
**Solution**: Verify that the `.env` file is in the project root and contains:
```
QDRANT_URL=your_url
QDRANT_API_KEY=your_key
```

### Issue: OpenAI Context Length Exceeded
**Cause**: Prompt too long (over 128,000 tokens for GPT-4).  
**Solution**: The system automatically truncates descriptions and options. If it still occurs:
1. Reduce `num_options_to_choose_from` in `classifai_cfg.ini`
2. Use a simpler product description
3. Switch to a model with larger context window

### Issue: "Page.goto: Timeout 30000ms exceeded" (ScrapeGraphAI)
**Cause**: Website taking too long to load.  
**Solution**: The Playwright timeout has been patched to 180 seconds internally. If still failing:
1. Check the website is accessible manually
2. Increase `timeout` parameter in `context.py` if needed
3. Some sites may block automated browsers—try a different source

### Issue: OpenAI Rate Limit
**Solution**: Increase `sleep_secs_between_products` in `classifai_cfg.ini` (e.g., from 0.1 to 1.0).

### Issue: Images Not Loading
**Solution**: Verify:
1. `image_passing_mode` in configuration (1=URL, 2=base64 file, 3=PIL in-memory)
2. That image URLs are publicly accessible
3. That the `img_temp/` folder is writable
4. Network connectivity and proxy settings

### Issue: Incorrect Classification
**Solution**: Try changing `classif_strategy` (recommended: 2 or 3 for best accuracy).

---

## Building an Executable

To distribute the project as an `.exe` file (Windows):

```bash
pip install pyinstaller
pyinstaller _7_classifai.py --onefile --icon=icon.ico
```

The executable will be in `dist/`.

---

## Performance Tips

- **Strategy selection**: Strategy 2 (Qdrant + OpenAI) offers the best balance between quality and speed
- **Batch processing**: Process large product lists in multiple runs to avoid timeouts
- **Rate limiting**: Adjust `sleep_secs_between_products` based on OpenAI quota
- **Caching**: Qdrant caches embeddings; reusing collections avoids recalculation
- **Images**: Use `image_passing_mode=1` (direct URL) for faster processing when images are not critical

---

## Support and Feedback

For questions, issues, or feedback:
- **Email**: donato.summa@istat.it
- **Issues**: Report bugs with full log output from `logs/` folder

---

## Changelog

### Recent Updates (May 2026)
- ✅ Updated Qdrant API from `search()` to `query_points()` (v1.16.1+)
- ✅ Added prompt truncation to prevent OpenAI context overflow
- ✅ Fixed logging formatting errors
- ✅ Increased Playwright browser timeouts to 180 seconds
- ✅ Improved error handling and exception logging
- ✅ Added comprehensive requirements.txt management

---

## License and Attribution

Research project for integrating AI/ML in production environments.  
Part of the ISTAT research initiative on product classification automation.

---
