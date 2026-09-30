# 🚀 Mattex Image Search Tool — Version 3 (SigLIP 2 & LanceDB)

Version 3 is a state-of-the-art multimodal image search engine powered by **Google SigLIP 2** foundation vision-language models and an embedded **LanceDB** vector store.

---

## 🌟 Key Features & Architectural Highlights

| Feature | Description |
| :--- | :--- |
| **Multimodal Vision-Language Engine** | Google **SigLIP 2 SO400M** (1152-dim, 384x384) & **SigLIP 2 Base** (768-dim, 256x256). |
| **Unified Shared Latent Space** | $L_2$-normalized vector space with 100% geometric alignment for text and image queries. |
| **Native Multilingual Support** | Pre-trained on WebLI covering **>100 languages** (English, Italian, French, Spanish, German, Chinese, Japanese, etc.). |
| **Embedded Columnar Vector Store** | **LanceDB** with zero-copy Apache Arrow backend and ACID transactional guarantees. |
| **SQL DataFusion Pre-Filtering** | Fast, indexed scalar pre-filtering (`.where("category = '...' AND subcategory = '...')`). |
| **Composed Multimodal Search** | Search by **Image + Text Modification Prompt** ($z_{\text{composed}} = \text{normalize}(\alpha z_I + \beta z_T)$). |
| **Fullscreen Lightbox Viewer** | Double-click preview modal with keyboard arrow navigation ($\leftarrow$ / $\rightarrow$ / `Esc`). |
| **Duplicate Image Finder** | Automated cluster detection for duplicate or near-identical images ($\ge 95\%$ cosine similarity). |
| **Result Exporter** | One-click export to copy top matching images to a chosen destination directory. |
| **LanceDB IVF-PQ Disk Indexing** | On-demand disk ANN index construction for scaling to 100,000+ images with sub-millisecond latency. |

---

## 📐 Mathematical Formulation

### Unified Latent Space & Dot Product Cosine Similarity

Given an input image $I$ and text query $T$, the SigLIP 2 encoder projects both modalities onto a unit hypersphere:

$$z_I = \frac{f_\theta(I)}{\|f_\theta(I)\|_2}, \quad z_T = \frac{g_\phi(T)}{\|g_\phi(T)\|_2}$$

Because $\|z_I\|_2 = \|z_T\|_2 = 1$, the cosine similarity coincides exactly with the vector dot product:

$$\text{sim}(z_I, z_T) = \sum_{k=1}^D z_{I,k} \cdot z_{T,k} = \cos(\theta)$$

### Composed Multimodal Search

To search for an image modified by a natural language instruction (e.g., base image of a red car + prompt *"in blue"*):

$$z_{\text{composed}} = \frac{\alpha z_I + \beta z_T}{\|\alpha z_I + \beta z_T\|_2}$$

---

## 📁 Project Structure

```
v3-SigLIP2-LanceDB/
├── venv/                 # Isolated Python virtual environment
├── main.py               # PyQt5 GUI Application with async workers & Lightbox
├── engine.py             # Core SigLIP 2 neural encoder & LanceDB manager
├── test_engine.py        # Automated test suite for all engine capabilities
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation (100% English)
└── dataset_lancedb/      # (auto-created) Embedded LanceDB database folder
```

---

## 🔧 Setup & Execution

### 1. Installation

```bash
cd v3-SigLIP2-LanceDB
./venv/bin/pip install -r requirements.txt
```

### 2. Running the GUI Application

```bash
./venv/bin/python3 main.py
```

### 3. Running the Verification Test Suite

```bash
./venv/bin/python3 test_engine.py
```

---

## 📖 User Guide

1. **Select Dataset Directory**:
   - Click **"📁 1. Select Dataset Folder"** to select your image directory.
2. **Start Indexing**:
   - Click **"🚀 2. Start Indexing"**. Incremental indexing automatically skips untouched files.
3. **Text Search**:
   - Enter a query in any language (e.g., *"red sports car"*, *"macchina rossa"*, *"voiture rouge"*).
   - Click **"Search by Text"** or press `Enter`.
4. **Visual Similarity Search**:
   - Select a target image and click **"Search Similar Images"**.
5. **Composed Search (Image + Text)**:
   - Select a base image, enter a text modification (e.g. *"in blue color"*), and click **"Search Composed"**.
6. **Preview & Lightbox**:
   - Double-click any result card to open the fullscreen Lightbox viewer. Use left/right arrow keys to cycle through top matches.
7. **Duplicate Finder**:
   - Click **"🔍 Find Duplicates"** in the header to find near-identical photo clusters ($\ge 95\%$).
8. **Export Results**:
   - Click **"💾 Export Results"** to copy top matching images into a target folder.

---

## 🖥️ Hardware Target & Memory Optimization

- **GPU Target**: NVIDIA GPU with 6–8 GB VRAM running `siglip2-so400m` in `bfloat16`/`float16`.
- **CPU / Lightweight**: Fallback to `siglip2-base` (768-dim) for low VRAM or CPU-only execution.

---

## 📄 License

MIT License — © 2026 Mattex  
Free to use, modify, and distribute for open-source and commercial applications.
