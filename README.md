# MediRag: Clinical Decision Support System (CDSS)# Medical-Graph-RAG

We build a Graph RAG System specifically for the medical domain.

MediRag is an AI-powered clinical decision support system that integrates medical knowledge graphs, semantic search, and large language models (LLMs) to provide evidence-based care recommendations for healthcare professionals.

Check our paper here: https://arxiv.org/abs/2408.04187

## Features

- **Semantic Search:** Retrieves relevant medical evidence using Qdrant vector database and Neo4j graph relationships.**⚠️ IMPORTANT: This project has been migrated from OpenAI to Google Gemini. See [GEMINI_MIGRATION.md](GEMINI_MIGRATION.md) for details.**

- **LLM Integration:** Uses Ollama (local LLM) to generate care options and clinical rationales based on patient context and evidence.

- **Robust Backend:** Flask-based API with error handling, degraded mode, and resilience to database outages.## Demo

- **Frontend Dashboard:** Interactive web UI for submitting clinical queries and viewing patient summaries, care options, and evidence mapping.a docker demo is here: https://hub.docker.com/repository/docker/jundewu/medrag-post/general

- **Configurable Environment:** All credentials and endpoints managed via `.env` file. 

**Note**: Demo will need updating for Gemini API:

## Technologies Used~~Use it by: docker run -it --rm --storage-opt size=10G -p 7860:7860 \ -e OPENAI_API_KEY= your_key -e NCBI_API_KEY= your_key medrag-post~~

- Python (Flask)Use it by: docker run -it --rm --storage-opt size=10G -p 7860:7860 \ -e GOOGLE_API_KEY= your_key -e NCBI_API_KEY= your_key medrag-post

- Neo4j (Medical Knowledge Graph)

- Qdrant (Vector Database)this demo used web-based searches on PubMed instead of locally storing medical papers and textbooks to detour the license wall.

- Ollama (Local LLM)

- HTML/CSS/JS (Frontend)## Quick Start (Hierarchical Medical Graph RAG)



## Getting Started1. **Setup Environment:**

   ```bash

### 1. Clone the Repository   conda env create -f medgraphrag.yml

```bash   # OR use the virtual environment

git clone https://github.com/<your-org>/MediRag.git   source venv/bin/activate

cd MediRag   pip install -r requirements.txt

```   ```



### 2. Environment Setup2. **Configure Environment Variables:**

Copy `.env.example` to `.env` and fill in your configuration:   ```bash

```bash   export GOOGLE_API_KEY=your_GOOGLE_API_KEY

cp .env.example .env   export NEO4J_URL=bolt://localhost:7687

```   export NEO4J_USERNAME=neo4j

Edit `.env` with your database, LLM, and API credentials.   export NEO4J_PASSWORD=your_password

   ```

### 3. Install Dependencies

```bash3. **Setup 3-Level Hierarchical Database:**

pip install -r requirements.txt   ```bash

```   python run.py -setup_hierarchy

   ```

### 4. Start Neo4j and Qdrant (Docker recommended)   

```bash   This creates:

docker-compose up neo4j qdrant   - **Bottom Level**: Medical Dictionary (UMLS concepts - diseases, symptoms, procedures)

```   - **Middle Level**: Medical Books and Papers (literature and knowledge)  

   - **Top Level**: Patient Records (clinical notes, test results)

### 5. Start Ollama (Local LLM)

Refer to [Ollama documentation](https://ollama.com/) for installation and model setup.4. **Test the Hybrid System:**

   ```bash

### 6. Run the Application   python run.py -test_hybrid

```bash   ```

python app.py

```5. **Add Patient Data (Runtime):**

Or use the provided shell script:   ```bash

```bash   python run.py -construct_graph -dataset mimic_ex -data_path ./dataset_test -grained_chunk -ingraphmerge -crossgraphmerge

./run_app.sh   ```

```

## Build from scratch (Complete Graph RAG flow in the paper)

### 7. Access the Web UI

Open [http://localhost:5000](http://localhost:5000) in your browser.### About the dataset

#### Paper Datasets

## Usage**Top-level Private data (user-provided)**: we used [MIMIC IV dataset](https://physionet.org/content/mimiciv/3.0/) as the private data.

- Upload or select a patient record.



## File StructureIn the code, we use the 'trinity' argument to enable the hierarchy graph linking function. If set to True, you must also provide a 'gid' (graph ID) to specify which graphs the top-level should link to. UMLS is largely structured as a graph, so minimal effort is required to construct it. However, MedC-K must be constructed as graph data. There are several methods you can use, such as the approach we used to process the top-level in this repo (open-source LLMs are recommended to keep costs down), or you can opt for non-learning-based graph construction algorithms (faster, cheaper, and generally noisier)

```

MediRag/#### Example Datasets

├── app.py                  # Flask app entry pointRecognizing that accessing and processing all the data mentioned may be challenging, we are working to provide simpler example dataset to demonstrate functionality. Currently, we are using the mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex) here as the Top-level data, which is the processed smaller dataset derived from MIMIC. For Medium-level and Bottom-level data, we are in the process of identifying suitable alternatives to simplify the implementation, welcome for any recommendations.

├── clinical_decision_engine.py  # Core backend logic

├── utils.py                # Utility functions (LLM, DB, embeddings)### 1. Prepare the environment, Neo4j and LLM

├── requirements.txt        # Python dependencies1. conda env create -f medgraphrag.yml

├── run_app.sh              # Shell script to run the app

├── templates/              # HTML templates (frontend)

├── static/                 # Static files (CSS, JS)2. prepare neo4j and LLM (using Google Gemini here for an example), you need to export:

├── sample_data/            # Example patient data

├── sample_fhir_data/       # Example FHIR dataexport GOOGLE_API_KEY = your_GOOGLE_API_KEY

├── medical_books_papers/   # Medical literature corpus

├── camel/                  # Core modules (agents, embeddings, etc.)export NEO4J_URL= your NEO4J_URL

├── uploads/                # Uploaded files

├── .env.example            # Environment variable templateexport NEO4J_USERNAME= your NEO4J_USERNAME

└── README.md               # Project documentation

```export NEO4J_PASSWORD= your NEO4J_PASSWORD



## Environment Variables### 2. Construct the graph (use "mimic_ex" dataset as an example)

See `.env.example` for all required keys:1. Download mimic_ex [here](https://huggingface.co/datasets/Morson/mimic_ex), put that under your data path, like ./dataset/mimic_ex

- `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBEDDING_MODEL`

- `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -grained_chunk -ingraphmerge -construct_graph

- `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME`

- `DATABASE_POOL_SIZE`, `MAX_RETRIES`, `ENVIRONMENT`, `LOG_LEVEL`### 3. Model Inference

1. put your prompt to ./prompt.txt

## Troubleshooting

- **LLM Fallback:** If Ollama is slow or unreachable, the backend will use semantic evidence to generate fallback care options.2. python run.py -dataset mimic_ex -data_path ./dataset/mimic_ex(where you put the dataset) -inference

- **Network Errors:** Ensure all services (Neo4j, Qdrant, Ollama) are running and accessible.

- **Frontend Issues:** Check browser console for JS errors; ensure API responses contain all required fields.## Acknowledgement

We are building on [CAMEL](https://github.com/camel-ai/camel), an awesome framework for construcing multi-agent pipeline.

## Contributing

Pull requests and issues are welcome! Please submit bug reports, feature requests, or improvements via GitHub.## Cite

~~~


