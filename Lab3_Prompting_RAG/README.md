# Lab 3 - Prompt Engineering and RAG

Choose **one LLM route**, then complete the two notebooks in order. Both routes have the
same exercises and submission requirements. The supplied `.env` file is the shared
configuration for both notebooks and the Docker setup.

## Your route through the lab

1. **Verify LLM access.** Choose one option:
   - **Ollama on your laptop:** install the Ollama application, or run Ollama with Docker
     Compose as described below. Work through `Get_Started_With_Ollama.ipynb` to learn
     chat, model comparison and embeddings.
   - **GCP Vertex AI:** use the instructor's GCP project and confirm access with the
     [official prompt-design Colab](https://colab.research.google.com/github/GoogleCloudPlatform/generative-ai/blob/main/gemini/prompts/intro_prompt_design.ipynb).
     Follow its authentication/setup instructions. Access to a Gemini Developer API key
     alone is not Vertex AI access.
2. Complete **`Lab3-1_Prompt_Engineering.ipynb`**: prompt patterns, sampling, chaining,
   structured output, function calling, ReAct and guardrails. Fill the TODOs and explain results.
3. Complete **`Lab3-2_RAG.ipynb`**: PDF ingestion, chunking, embeddings, FAISS
   retrieval, grounded answers and evaluation. Fill its TODOs and compare settings.
4. Submit the **two completed lab notebooks to dépôt DVL**. See the submission checklist below.

The onboarding notebook is preparation, not a third required submission.

## Files you need to keep together

| File | Purpose |
| --- | --- |
| `Get_Started_With_Ollama.ipynb` | Ollama onboarding and model/embedding experiments |
| `Lab3-1_Prompt_Engineering.ipynb` | First assessed notebook |
| `Lab3-2_RAG.ipynb` | Second assessed notebook |
| `notebook_utils.py` | Shared provider adapters, tools, schemas, RAG and environment helpers |
| `pyproject.toml` | Python dependencies, with `ollama` and `vertex` extras |
| `.env` / `.env.example` | Your shared settings / a clean template |
| `compose.yaml` | Optional Ollama server in Docker with persistent model storage |
| `compose.gpu.yaml` | Optional NVIDIA GPU override for compatible hosts |

## 1. Configure `.env`

The starter `.env` selects Ollama. If it is missing (e.g. hidden files were not copied), copy
`.env.example` to `.env`. Edit it with a text editor; no shell `source` command is required.

| Settings | What to change |
| --- | --- |
| `LLM_PROVIDER` | `ollama` or `vertex` |
| `OLLAMA_BASE_URL` | Laptop URL of the native or Docker server, usually `http://localhost:11434` |
| `OLLAMA_MODEL` | Chat model; use one supporting tools and structured JSON for the full prompt lab |
| `OLLAMA_EMBED_MODEL` | Dedicated embedding model, initially `all-minilm` |
| `VERTEX_PROJECT_ID`, `VERTEX_MODEL`, `VERTEX_LOCATION` | Assigned GCP project, chat model and supported region |
| `VERTEX_EMBED_MODEL`, `VERTEX_EMBED_LOCATION` | Embedding model and its supported region; can differ from generation |
| `AUTHENTICATE_IN_COLAB` | `true` for the Colab Vertex login flow; `false` for local ADC |
| `TEMPERATURE`, `MAX_OUTPUT_TOKENS`, `REQUEST_TIMEOUT_SECONDS` | Shared generation controls |
| `RAG_*` | PDF/index paths, download count, chunking, top-k and page/chunk limits |

The Vertex project and chat model are empty until you enter the instructor's values. Do not
put API keys, service-account JSON or passwords in `.env` or notebook cells. GCP authentication
uses Colab login or local Application Default Credentials (ADC).

Both notebooks read the `.env` beside `notebook_utils.py`. Existing process environment
variables take precedence; remove stale exported values if edits to `.env` appear ineffective.
Rerun setup/client cells after configuration changes. Restart the kernel after editing helper
code. Changing provider or embedding settings requires rebuilding the RAG index.

## 2. Create a local Python environment

Python **3.11 or 3.12 is recommended** for straightforward scientific-package installation.
Run these commands in a terminal **inside `Lab3_Prompting_RAG/`**.

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If your managed computer blocks PowerShell activation, use `.venv\Scripts\python.exe`
in place of `python`; activation is a convenience, not a requirement.

### Install the dependencies for your chosen route

For Ollama, whether native or Docker:

```bash
python -m pip install -e ".[ollama]"
```

For Vertex AI:

```bash
python -m pip install -e ".[vertex]"
```

Only if you intend to test both routes, install `python -m pip install -e ".[all]"`.
Start Jupyter from the same folder and select the virtual environment's Python kernel:

```bash
python -m ipykernel install --user --name lab5 --display-name "Python (Lab 3)"
python -m jupyter lab
```

VS Code users can select `.venv` directly as the notebook kernel. The project installs NumPy,
pandas, requests, python-dotenv, Pydantic, pypdf, the LangChain loader/splitter/vector-store
packages, FAISS CPU, JupyterLab and ipykernel. The extra adds only the selected provider SDK.
The RAG notebook uses provider embeddings; Sentence-Transformers, TensorFlow, tf-keras and
the legacy `google-generativeai` package are not required.

## 3A. Ollama installed directly on the laptop

Install/update [Ollama](https://ollama.com/download), then start the application. If no server
is already running, `ollama serve` starts it from a terminal. Pull the chat and embedding
models named in `.env` (these are the starter names):

```bash
ollama pull qwen3:0.6b
ollama pull all-minilm
ollama list
```

For the optional onboarding comparisons, also pull the additional models in
`OLLAMA_CHAT_MODELS` and `OLLAMA_EMBED_MODELS`, or reduce those lists to the models you want.
Not all chat models support native function calling. The default Qwen3 candidate is small;
choose a larger suitable model if your hardware permits and record the exact name.

Open `Get_Started_With_Ollama.ipynb`, then proceed to the two assessed notebooks.

## 3B. Ollama in Docker (alternative to 3A)

Install Docker Desktop, or Docker Engine with the Compose plugin, and start Docker.
**Jupyter still runs in the local virtual environment.** Compose runs only the model server,
so neither Python nor the notebooks need to be placed in a container.

From this folder:

```bash
docker compose pull ollama
docker compose up -d ollama
docker compose --profile setup run --rm pull-models
docker compose exec ollama ollama list
```

`pull-models` waits for the server health check and pulls the two model names from `.env`.
The server is published only on laptop loopback at the configured `OLLAMA_PORT`. Use
`OLLAMA_BASE_URL=http://localhost:11434` in the notebooks; `http://ollama:11434` is only the
internal container-to-container URL used by the setup service.

If native Ollama is already using port 11434, stop that server or set both
`OLLAMA_PORT=11435` and `OLLAMA_BASE_URL=http://localhost:11435` in `.env` before starting
Compose. Native and Docker installations have **separate model stores**: models pulled
natively are not automatically available in the container.

Useful commands:

```bash
docker compose logs --tail=50 ollama
docker compose stop ollama
docker compose up -d ollama
docker compose down
```

The named volume preserves downloaded models across container recreation and `down`.
Do not add `-v` to `down` unless you intend to delete the model volume. The default image
tag is `latest`; run `docker compose pull ollama` and `docker compose up -d ollama` to
refresh it, or set `OLLAMA_IMAGE_TAG` to an instructor-specified version for reproducibility.

The default configuration works on CPU. On a compatible NVIDIA host with the NVIDIA
Container Toolkit, use the override consistently:

```bash
docker compose -f compose.yaml -f compose.gpu.yaml up -d ollama
docker compose -f compose.yaml -f compose.gpu.yaml --profile setup run --rm pull-models
```

GPU passthrough is not available with Docker Desktop on macOS; native Ollama is usually
preferable there for GPU acceleration. See the [official Docker guide](https://docs.ollama.com/docker)
and [GPU FAQ](https://docs.ollama.com/faq#how-do-i-use-ollama-with-gpu-acceleration-in-docker).

## 3C. GCP Vertex AI, locally or in Colab

Use the instructor's project and an enabled model/region. The project needs billing,
the Vertex AI API and permission to invoke both the generation and embedding models.
GCP calls can incur costs; the supplied sampling/agent/indexing limits keep initial work small.

**Local notebook:** install the Google Cloud CLI and run:

```bash
gcloud auth application-default login
```

Set the Vertex fields in `.env`, `LLM_PROVIDER=vertex` and `AUTHENTICATE_IN_COLAB=false`.
Start Jupyter in your venv. No Docker or local Ollama server is required for this route.

**Online Colab:** first verify access in the linked official Colab. Then upload this folder's
three notebooks, `notebook_utils.py`, `pyproject.toml` and `.env.example` using the Colab Files
panel, all into the same working directory (normally `/content`). Run a temporary setup cell:

```python
%cd /content
!cp .env.example .env
%pip install -e ".[vertex]"
```

Copy the template only once, before editing: it overwrites an existing `.env`. Edit `.env`
in the Files panel: choose Vertex, fill in the project/model/regions, and set
`AUTHENTICATE_IN_COLAB=true`. Open the prompt lab, run its configuration and login cells,
then its smoke test. Repeat the setup/upload for a new Colab runtime when opening the RAG
notebook, or keep the files in a mounted Drive folder and work from that directory. Colab
runtime files are temporary; download your completed notebooks before the runtime expires.
A hosted Colab notebook cannot reach a laptop Ollama server through `localhost`.

## 4. Complete the prompt and RAG notebooks

Keep the student TODOs as your work areas; do not merely run the supplied demonstrations.
Shared helpers and new tool/schema implementations belong in `notebook_utils.py`.

The RAG downloader obtains up to ten fixed-version research PDFs and records their source
URLs and checksums. The starter limits extraction to 5 pages per PDF and indexing to 200
chunks. Limits can omit later papers: inspect the source list before choosing evaluation
questions. `RAG_CHUNK_SIZE` and `RAG_CHUNK_OVERLAP` are **characters**. If embeddings reject
a chunk as too long, reduce the size. Scanned PDFs need OCR, which is not included.

Both routes normalize embeddings and use FAISS inner product for cosine search. Index
persistence uses FAISS plus plain JSON metadata, not pickle. Use your own index; rebuild when
changing documents, chunking, provider, embedding model, embedding region or model weights.
Changing only the generation model does not require re-embedding the collection.

## 5. Submission to dépôt DVL

Submit these two required files:

1. **`Lab3-1_Prompt_Engineering.ipynb`** - completed exercises, representative outputs,
   model/provider identification, function-calling/ReAct traces and guardrail discussion.
2. **`Lab3-2_RAG.ipynb`** - completed pipeline, retrieved evidence/citations,
   evaluation questions/results, comparisons and failure analysis.

Before submitting, save both notebooks with useful outputs and written answers. Document
any edits to `notebook_utils.py` in a final Markdown appendix, including relevant code excerpts,
so the changes are visible in the two required notebook files. Include the names and settings
of the models you actually used. The provided notebook outputs are cleared for distribution;
your submitted versions should show your work.

Do not submit the onboarding notebook, `.env`, credentials, `.venv`, downloaded PDFs,
Docker model storage or vector indexes. Upload the two assessed notebooks to the **Lab 5
dépôt on DVL**, following any deadline and naming instructions given by the instructor.

## Optional setup checks

These checks use simulated API responses and need no server or cloud credentials. They do
not complete student exercises or establish live Vertex access. After installing your route:

```bash
python -m unittest discover -s tests -v
```

The inactive provider's adapter check is skipped when its SDK is not installed.
