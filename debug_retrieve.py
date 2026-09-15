from pipeline import retrieve_chunks

results = retrieve_chunks("मेसिन लर्निङ के हो?", top_k=5, source_file="samplee.pdf")
for (chunk_id, source_file, chunk_text, distance) in results:
    print(f"Distance: {distance:.4f} | Source: {source_file}")
    print(f"Text: {chunk_text[:100]}")
    print()