import os
import shutil
import tempfile
from PIL import Image, ImageDraw

from engine import SigLIP2Engine, LanceDBManager, IndexManager, SUPPORTED_MODELS

def create_sample_dataset(root_dir: str):
    cats = {
        "nature": ["mountains.png", "forest.jpg"],
        "vehicles": ["red_car.png", "blue_boat.jpg"]
    }

    for cat, files in cats.items():
        cat_dir = os.path.join(root_dir, cat, "photos")
        os.makedirs(cat_dir, exist_ok=True)
        for fname in files:
            fpath = os.path.join(cat_dir, fname)
            img = Image.new("RGB", (256, 256), color=(255 if "red" in fname else 50, 200 if "forest" in fname or "mountain" in fname else 50, 255 if "blue" in fname else 50))
            draw = ImageDraw.Draw(img)
            draw.text((20, 20), fname, fill=(255, 255, 255))
            img.save(fpath)

    # Create an exact duplicate file to test duplicate detection
    dup_path = os.path.join(root_dir, "vehicles", "photos", "red_car_copy.png")
    shutil.copy(os.path.join(root_dir, "vehicles", "photos", "red_car.png"), dup_path)

def test_pipeline():
    print("=== Starting v3 Engine Verification Test Suite ===")
    tmp_dir = tempfile.mkdtemp(prefix="mattex_test_")
    db_dir = os.path.join(tmp_dir, "lancedb_data")
    dataset_dir = os.path.join(tmp_dir, "dataset")

    try:
        print("1. Creating synthetic dataset (5 images total including 1 duplicate)...")
        create_sample_dataset(dataset_dir)

        print("2. Initializing SigLIP2Engine (siglip2-base for rapid testing)...")
        engine = SigLIP2Engine(model_key="siglip2-base")
        assert engine.dim == 768, f"Expected dimension 768, got {engine.dim}"

        print("3. Initializing LanceDBManager...")
        db_manager = LanceDBManager(db_path=db_dir, vector_dim=engine.dim)

        print("4. Indexing synthetic dataset...")
        index_manager = IndexManager(engine, db_manager)

        def progress_cb(current, total, filename):
            print(f"   Indexed {current}/{total}: {filename}")

        count = index_manager.index_directory(dataset_dir, batch_size=2, progress_callback=progress_cb)
        print(f"   Total indexed: {count}")
        assert count == 5, f"Expected 5 images indexed, got {count}"

        print("5. Incremental indexing test (should skip all existing images)...")
        count_reindex = index_manager.index_directory(dataset_dir, batch_size=2)
        print(f"   Re-indexed count: {count_reindex}")
        assert count_reindex == 0, f"Expected 0 new images, got {count_reindex}"

        print("6. Testing Text Search ('red car')...")
        text_vec = engine.encode_text_query("red car")
        text_results = db_manager.search_vector(text_vec, top_k=2)
        print("   Text Results:")
        for r in text_results:
            print(f"     - {r['rel_path']} | Category: {r['category']} | cos(θ): {r['raw_cosine']:.4f} | Score: {r['score']:.2f}%")
        assert len(text_results) > 0

        print("7. Testing Visual Image Search (mountains.png)...")
        sample_img = os.path.join(dataset_dir, "nature", "photos", "mountains.png")
        img_vec = engine.encode_images_batch([sample_img])[0]
        img_results = db_manager.search_vector(img_vec, top_k=2, is_text_search=False)
        print("   Visual Results:")
        for r in img_results:
            print(f"     - {r['rel_path']} | Category: {r['category']} | Score: {r['score']:.2f}%")
        assert img_results[0]["rel_path"].endswith("mountains.png")

        print("8. Testing Composed Image Search (red_car.png + 'blue')...")
        base_img = os.path.join(dataset_dir, "vehicles", "photos", "red_car.png")
        comp_vec = engine.encode_composed_query(base_img, text_modification="blue")
        comp_results = db_manager.search_vector(comp_vec, top_k=2, is_text_search=True)
        print("   Composed Results:")
        for r in comp_results:
            print(f"     - {r['rel_path']} | Score: {r['score']:.2f}%")

        print("9. Testing Duplicate Finder (threshold >= 95%)...")
        duplicates = db_manager.find_duplicates(similarity_threshold=0.95)
        print(f"   Found {len(duplicates)} duplicate clusters.")
        for cluster in duplicates:
            print(f"     - Primary: {cluster['primary_rel_path']}")
            for m in cluster["matches"]:
                print(f"       ↳ Duplicate: {m['rel_path']} ({m['similarity']:.1f}% match)")
        assert len(duplicates) > 0, "Expected at least 1 duplicate cluster"

        print("10. Testing SQL Category Pre-filtering ('vehicles')...")
        filtered_results = db_manager.search_vector(text_vec, top_k=5, category="vehicles")
        print("   Filtered Results:")
        for r in filtered_results:
            print(f"     - {r['rel_path']} | Category: {r['category']} | Score: {r['score']:.2f}%")
        print(f"11. Testing Disk Index status check (has_disk_index)... Has index: {db_manager.has_disk_index()}")

        print("\n✅ ALL ENGINE VERIFICATION TESTS PASSED SUCCESSFULLY!")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

if __name__ == "__main__":
    test_pipeline()
