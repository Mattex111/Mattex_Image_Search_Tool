# 🧠 Mattex Image Search Tool

A smart desktop tool for **image similarity search** and **natural language text search**, powered by **SigLIP**, **OpenCLIP**, **MobileNetV4**, and **PyQt**.

---

## 📦 Available Versions

### 🚀 [`v3-SigLIP2-LanceDB`](./v3-SigLIP2-LanceDB) ⭐ *(RECOMMENDED 2026)*

State-of-the-art multimodal vector search engine featuring Google's SigLIP 2 and LanceDB:

- **Models**: **SigLIP 2 SO400M** (1152-dim, 384×384) & **SigLIP 2 Base** (768-dim, 256×256)
- **Latent Space**: Unified single-space vision-language encoder with native multilingual support (>100 languages)
- **Database**: **LanceDB Embedded Vector Store** (Apache Arrow zero-copy backend, ACID transactional)
- **Filtering**: **SQL DataFusion Pre-filtering** for categories and subcategories
- **GUI**: Modern dark theme PyQt5 interface with async thread pool, drag & drop, live progress, and cross-platform desktop integration

➡️ [View README for v3-SigLIP2-LanceDB →](./v3-SigLIP2-LanceDB/README.md)

---

### 🔹 [`v2-Clip-Integration`](./v2-Clip-Integration) *(Legacy)*

Experimental version with text search via OpenAI CLIP ViT-B/32 + MobileNetV2.

➡️ [View README for v2 →](./v2-Clip-Integration/README.md)

---

### 🔹 [`v1-MobileNet`](./v1-MobileNet) *(Legacy)*

Classic image similarity search using TensorFlow MobileNetV2.

➡️ [View README for v1 →](./v1-MobileNet/README.md)

---

## ⚖️ Version Comparison

| Goal | Recommended Version |
|---|---|
| **State-of-the-art Multilingual Text & Visual Search** | `v3-SigLIP2-LanceDB` (SigLIP 2 SO400M / Base) |
| **Transactional Vector DB & SQL Filtering** | `v3-SigLIP2-LanceDB` (LanceDB) |
| Legacy CLIP | `v2-Clip-Integration` |
| Legacy MobileNetV2 | `v1-MobileNet` |

---

## 📄 License

Released under the **MIT License**.  
© 2026 Mattex — Feel free to use, modify, and share with credit. ✌️
