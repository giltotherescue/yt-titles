from load_pinecone import query_pinecone

def print_results(results: list) -> None:
    """Print formatted search results."""
    print("\nTop performing similar titles:")
    print("-" * 60)
    for idx, result in enumerate(results, 1):
        metadata = result['metadata']
        print(f"\n{idx}. Title: {metadata['title']}")
        print(f"Views: {metadata['total_view_count']:,}")
        print(f"Type: {'Short' if metadata['is_short'] else 'Regular video'}")
        print(f"Similarity Score: {result['similarity_score']:.3f}")
        print(f"Hybrid Score: {result['hybrid_score']:.3f}")

def main():
    # Test specific queries
    test_queries = [
        "funny cat videos",
        "gaming tutorial guide",
        "workout motivation",
    ]

    for query in test_queries:
        print(f"\n\nTesting query: '{query}'")
        print("=" * 60)
        results = query_pinecone(query, top_k=3, view_count_weight=0.3)
        print_results(results)

    # Test view count weight variations
    query = "funny cat videos"
    for weight in [0.0, 0.5, 1.0]:
        print(f"\n\nTesting with view_count_weight = {weight}")
        print("=" * 60)
        results = query_pinecone(query, top_k=3, view_count_weight=weight)
        print_results(results)

if __name__ == "__main__":
    main() 