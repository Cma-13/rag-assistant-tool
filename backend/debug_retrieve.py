from pipeline import retrieve_chunks

results = retrieve_chunks("सेडाको सिरानी मुनि के भेटियो?", top_k=5, source_file="story.pdf")
for (chunk_id, source_file, chunk_text, distance) in results:
    print(f"Distance: {distance:.4f}")
    print(f"Text: {chunk_text[:150]}")
    print()