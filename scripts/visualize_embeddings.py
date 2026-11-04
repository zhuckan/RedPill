"""Визуализация эмбеддингов из Qdrant с помощью UMAP и t-SNE.

Достаёт все векторы из коллекции knowledge_memory, сжимает 1024-мерное
пространство до 2D и строит scatter plot. Точки окрашены по источнику
(статья учителя vs исторические события) — так видно, как кластеризуются
документы.

Сохраняет:
- knowledge_base/umap.png  — UMAP-визуализация
- knowledge_base/tsne.png  — t-SNE-визуализация (альтернатива)

Запуск:
    python -m scripts.visualize_embeddings
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from qdrant_client import QdrantClient
from sklearn.manifold import TSNE

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Внимание: umap-learn не установлен, будет только t-SNE.")
    print("Установи: uv pip install umap-learn")

from config import settings

OUTPUT_DIR = "knowledge_base"
UMAP_PATH = os.path.join(OUTPUT_DIR, "umap.png")
TSNE_PATH = os.path.join(OUTPUT_DIR, "tsne.png")

# Цвета для разных источников
COLORS = {
    "mentor_article_2026-10-28": "#2E86AB",  # синий
    "historical_events": "#E63946",           # красный
}
DEFAULT_COLOR = "#888888"


def main() -> None:
    """Загружает эмбеддинги, сжимает и строит графики."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    client = QdrantClient(settings.QDRANT_HOST)
    print("Загружаю эмбеддинги из knowledge_memory...")

    records, _ = client.scroll(
        collection_name=settings.QDRANT_KNOWLEDGE_COLLECTION,
        limit=10000,
        with_vectors=True,
        with_payload=True,
    )

    if len(records) < 5:
        print(f"Слишком мало точек для визуализации: {len(records)}")
        return

    print(f"Получено {len(records)} точек.")

    # Собираем векторы, источники и превью (для подсказок)
    vectors = np.array([rec.vector for rec in records])
    sources = [rec.payload.get("source", "") for rec in records]

    print(f"Размерность исходных векторов: {vectors.shape}")

    # --- UMAP ---
    if HAS_UMAP:
        print("Запускаю UMAP (это может занять 10-30 секунд)...")
        reducer = umap.UMAP(n_neighbors=10, min_dist=0.3, random_state=42)
        coords_2d = reducer.fit_transform(vectors)

        _plot(
            coords_2d, sources,
            title="UMAP: 80 чанков knowledge_memory",
            output_path=UMAP_PATH,
        )
        print(f"UMAP сохранён: {UMAP_PATH}")

    # --- t-SNE ---
    print("Запускаю t-SNE (это может занять 10-30 секунд)...")
    # perplexity должна быть < числа точек
    perplexity = min(30, max(5, len(records) // 3))
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca")
    tsne_coords = tsne.fit_transform(vectors)

    _plot(
        tsne_coords, sources,
        title=f"t-SNE: 80 чанков knowledge_memory (perplexity={perplexity})",
        output_path=TSNE_PATH,
    )
    print(f"t-SNE сохранён: {TSNE_PATH}")

    print("\nГотово. Открой knowledge_base/umap.png или tsne.png для просмотра.")


def _plot(coords: np.ndarray, sources: list, title: str, output_path: str) -> None:
    """Строит scatter plot и сохраняет в файл."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Группируем точки по источнику, чтобы добавить легенду
    unique_sources = sorted(set(sources))

    for source in unique_sources:
        mask = [i for i, s in enumerate(sources) if s == source]
        color = COLORS.get(source, DEFAULT_COLOR)
        label = _short_label(source)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            label=label,
            s=60,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _short_label(source: str) -> str:
    """Короткое название источника для легенды."""
    if source == "mentor_article_2026-10-28":
        return "Статья учителя (43)"
    if source == "historical_events":
        return "Исторические события (37)"
    return source or "Unknown"


if __name__ == "__main__":
    main()