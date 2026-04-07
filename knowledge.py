"""
Strategic knowledge system for the Factorio RL agent.
Scrapes Factorio guides, encodes them into embeddings, and produces
a strategy vector that conditions the CNN's decisions.

The CNN sees pixels. This module provides background knowledge.
Two systems, one agent.
"""

import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


CACHE_DIR = Path("knowledge_cache")
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension

# Sources to scrape
WIKI_URLS = [
    "https://wiki.factorio.com/Tutorial:Quick_start_guide",
    "https://wiki.factorio.com/Mining",
    "https://wiki.factorio.com/Furnace",
    "https://wiki.factorio.com/Assembling_machine",
    "https://wiki.factorio.com/Belt_transport_system",
    "https://wiki.factorio.com/Inserter",
]

REDDIT_SEARCH = "factorio beginner tips"
REDDIT_LIMIT = 5

# Additional sources for knowledge refresh when agent is stuck
REFRESH_WIKI_URLS = [
    "https://wiki.factorio.com/Power_production",
    "https://wiki.factorio.com/Enemies",
    "https://wiki.factorio.com/Wall",
    "https://wiki.factorio.com/Turret",
    "https://wiki.factorio.com/Oil_processing",
    "https://wiki.factorio.com/Logistic_network",
    "https://wiki.factorio.com/Railway",
]

REFRESH_REDDIT_QUERIES = [
    "factorio tips when stuck early game",
    "factorio what to do after smelting",
    "factorio beginner automation guide",
]

# Stage-specific queries for selecting relevant knowledge
STAGE_QUERIES = {
    "Exploration": [
        "how to move around and explore in factorio",
        "basic controls and navigation in factorio",
        "what to do first in a new factorio game",
    ],
    "Gathering": [
        "how to mine iron and copper ore in factorio",
        "picking up resources and filling inventory",
        "building your first stone furnace for smelting",
        "early game resource gathering strategy",
    ],
    "Automation": [
        "how to build inserters and belts for automation",
        "setting up automated smelting production lines",
        "crafting machines and assemblers in factorio",
        "power generation with steam engines and boilers",
    ],
}


# ---------------------------------------------------------------------------
# Scraping
# ---------------------------------------------------------------------------

def _url_hash(url):
    return hashlib.md5(url.encode()).hexdigest()[:12]


def _scrape_wiki(url):
    """Scrape text content from a Factorio wiki page."""
    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "FactorioRL/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Get main content area
        content = soup.find("div", {"id": "mw-content-text"})
        if not content:
            content = soup.find("main") or soup.find("body")

        # Remove scripts, styles, nav
        for tag in content.find_all(["script", "style", "nav", "footer"]):
            tag.decompose()

        text = content.get_text(separator="\n", strip=True)
        # Clean up excessive whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception as e:
        print(f"  Failed to scrape {url}: {e}")
        return ""


def _scrape_reddit(query, limit=5):
    """Scrape top Reddit results for a search query."""
    texts = []
    try:
        url = f"https://www.reddit.com/search.json?q={query}&sort=relevance&limit={limit}"
        resp = requests.get(url, timeout=15, headers={"User-Agent": "FactorioRL/1.0"})
        resp.raise_for_status()
        data = resp.json()

        for post in data.get("data", {}).get("children", []):
            p = post.get("data", {})
            title = p.get("title", "")
            selftext = p.get("selftext", "")
            if title:
                texts.append(f"{title}\n{selftext}".strip())
    except Exception as e:
        print(f"  Failed to scrape Reddit: {e}")
    return texts


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_text(text, max_chars=500):
    """Split text into chunks suitable for embedding."""
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current) + len(para) > max_chars and current:
            chunks.append(current.strip())
            current = para
        else:
            current = current + "\n" + para if current else para

    if current.strip():
        chunks.append(current.strip())

    # Filter out very short chunks (navigation artifacts etc)
    return [c for c in chunks if len(c) > 50]


# ---------------------------------------------------------------------------
# Knowledge base
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """Scrapes, embeds, and serves strategic knowledge for the RL agent."""

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        CACHE_DIR.mkdir(exist_ok=True)
        self.chunks = []
        self.embeddings = None  # (N, 384)
        self.model = None
        self.model_name = model_name
        self._current_strategy = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        self._current_stage = None

    def _load_model(self):
        if self.model is None:
            print(f"  Loading sentence transformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def build(self, force_refresh=False):
        """Scrape all sources and build the embedding index."""
        cache_file = CACHE_DIR / "knowledge.npz"
        chunks_file = CACHE_DIR / "chunks.json"

        if not force_refresh and cache_file.exists() and chunks_file.exists():
            print("  Loading cached knowledge base...")
            self.chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
            data = np.load(cache_file)
            self.embeddings = data["embeddings"]
            print(f"  Loaded {len(self.chunks)} chunks, embeddings {self.embeddings.shape}")
            return

        print("  Building knowledge base from scratch...")
        all_chunks = []

        # Scrape wiki
        for url in WIKI_URLS:
            print(f"  Scraping: {url}")
            text = _scrape_wiki(url)
            if text:
                chunks = _chunk_text(text)
                print(f"    -> {len(chunks)} chunks")
                all_chunks.extend(chunks)
            time.sleep(1)  # Be polite

        # Scrape Reddit
        print(f"  Scraping Reddit: '{REDDIT_SEARCH}'")
        reddit_texts = _scrape_reddit(REDDIT_SEARCH, REDDIT_LIMIT)
        for rt in reddit_texts:
            chunks = _chunk_text(rt, max_chars=300)
            all_chunks.extend(chunks)
        print(f"    -> {len(reddit_texts)} posts, {sum(len(_chunk_text(rt, 300)) for rt in reddit_texts)} chunks")

        if not all_chunks:
            print("  WARNING: No content scraped. Using fallback knowledge.")
            all_chunks = [
                "In Factorio, you start by mining iron and copper ore by hand.",
                "Build stone furnaces to smelt ore into plates.",
                "Use plates to craft machines like inserters and belts.",
                "Automate production with assembling machines.",
                "Build power with boilers and steam engines using coal and water.",
                "Explore to find resource patches of iron, copper, coal, and stone.",
                "Biters will attack your base over time. Build walls and turrets.",
                "The goal is to launch a rocket by building a full production chain.",
            ]

        self.chunks = all_chunks

        # Embed all chunks
        model = self._load_model()
        print(f"  Embedding {len(self.chunks)} chunks...")
        self.embeddings = model.encode(self.chunks, show_progress_bar=True,
                                       convert_to_numpy=True, normalize_embeddings=True)

        # Cache
        np.savez_compressed(cache_file, embeddings=self.embeddings)
        chunks_file.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        print(f"  Knowledge base built: {self.embeddings.shape}")

    def refresh(self, stage_name):
        """Expand the knowledge base with additional sources. Called when stuck."""
        print("\n  Refreshing knowledge base with additional sources...")
        new_chunks = []

        for url in REFRESH_WIKI_URLS:
            print(f"  Scraping: {url}")
            text = _scrape_wiki(url)
            if text:
                chunks = _chunk_text(text)
                new_chunks.extend(chunks)
            time.sleep(0.5)

        for query in REFRESH_REDDIT_QUERIES:
            print(f"  Reddit: '{query}'")
            texts = _scrape_reddit(query, limit=3)
            for t in texts:
                new_chunks.extend(_chunk_text(t, max_chars=300))
            time.sleep(0.5)

        if not new_chunks:
            print("  No new content found.")
            return

        # Deduplicate against existing chunks
        existing = set(self.chunks)
        new_chunks = [c for c in new_chunks if c not in existing]
        print(f"  {len(new_chunks)} new unique chunks found")

        if not new_chunks:
            return

        # Embed and append
        model = self._load_model()
        new_embs = model.encode(new_chunks, convert_to_numpy=True, normalize_embeddings=True)
        self.chunks.extend(new_chunks)
        self.embeddings = np.vstack([self.embeddings, new_embs])

        # Update cache
        cache_file = CACHE_DIR / "knowledge.npz"
        chunks_file = CACHE_DIR / "chunks.json"
        np.savez_compressed(cache_file, embeddings=self.embeddings)
        chunks_file.write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")
        print(f"  Knowledge base expanded: {self.embeddings.shape}")

        # Force strategy recalculation
        self._current_stage = None

    def get_strategy_vector(self, stage_name):
        """Get a strategy embedding for the current curriculum stage.

        Returns a (384,) float32 vector — the mean of the top-K most relevant
        knowledge chunks for this stage.
        """
        if stage_name == self._current_stage:
            return self._current_strategy

        model = self._load_model()
        queries = STAGE_QUERIES.get(stage_name, STAGE_QUERIES["Exploration"])

        # Encode stage queries
        query_embs = model.encode(queries, convert_to_numpy=True, normalize_embeddings=True)

        # Mean query embedding
        query_mean = query_embs.mean(axis=0)
        query_mean = query_mean / (np.linalg.norm(query_mean) + 1e-8)

        # Cosine similarity against all knowledge chunks
        similarities = self.embeddings @ query_mean  # (N,)

        # Top-K relevant chunks
        top_k = min(10, len(self.chunks))
        top_indices = np.argsort(similarities)[-top_k:]

        # Weighted mean of top chunk embeddings (weighted by similarity)
        top_sims = similarities[top_indices]
        weights = np.maximum(top_sims, 0)
        weights = weights / (weights.sum() + 1e-8)

        strategy = (self.embeddings[top_indices] * weights[:, None]).sum(axis=0)
        strategy = strategy / (np.linalg.norm(strategy) + 1e-8)

        self._current_strategy = strategy.astype(np.float32)
        self._current_stage = stage_name

        # Log top chunks
        print(f"\n  Strategy vector updated for stage: {stage_name}")
        print(f"  Top relevant knowledge:")
        for idx in reversed(top_indices[-3:]):
            snippet = self.chunks[idx][:100].replace("\n", " ")
            print(f"    [{similarities[idx]:.3f}] {snippet}...")

        return self._current_strategy

    @property
    def embedding_dim(self):
        return EMBEDDING_DIM


def main():
    print("=== Knowledge Base Test ===\n")

    kb = KnowledgeBase()
    kb.build(force_refresh=True)

    print(f"\nKnowledge base: {len(kb.chunks)} chunks, embeddings {kb.embeddings.shape}")

    for stage in ["Exploration", "Gathering", "Automation"]:
        vec = kb.get_strategy_vector(stage)
        print(f"\n  {stage} vector: shape={vec.shape}, norm={np.linalg.norm(vec):.3f}, "
              f"range=[{vec.min():.3f}, {vec.max():.3f}]")

    print("\nDone.")


if __name__ == "__main__":
    main()
