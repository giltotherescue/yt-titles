"""
Core functionality for handling embeddings and vector operations with Pinecone.
"""
import pandas as pd
from pinecone import Pinecone
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np

# Load environment variables
load_dotenv()

# Initialize clients
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index("yt-titles-1")

def get_embedding(text: str) -> list:
    """Convert text to vector embedding using OpenAI's API."""
    print(f"Getting embedding for text: '{text[:50]}...'")
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    print("Successfully got embedding")
    return response.data[0].embedding

def load_csv_to_pinecone(csv_path: str) -> None:
    """Load YouTube title data from CSV into Pinecone index."""
    print(f"Reading CSV file from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from CSV")
    
    for idx, row in df.iterrows():
        print(f"\nProcessing row {idx + 1}/{len(df)}")
        text_content = f"{row['title']} {'#shorts' if row['is_short'] else ''}"
        print(f"Title: {text_content}")
        
        try:
            vector = get_embedding(text_content)
            metadata = {
                'yt_video_id': row['yt_video_id'],
                'title': row['title'],
                'total_view_count': int(row['total_view_count']),
                'is_short': bool(row['is_short'])
            }
            
            index.upsert(vectors=[(row['yt_video_id'], vector, metadata)])
            print(f"Successfully processed video: {row['yt_video_id']}")
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            continue

def query_pinecone(query_text: str, top_k: int = 5, view_count_weight: float = 0.3) -> list:
    """
    Query similar titles with hybrid scoring (similarity + view count).
    
    Args:
        query_text: The title to find similar matches for
        top_k: Number of results to return
        view_count_weight: Weight given to view count in scoring (0-1)
    """
    print(f"\nProcessing query: '{query_text}'")
    query_vector = get_embedding(query_text)
    
    initial_results = index.query(
        vector=query_vector,
        top_k=top_k * 2,
        include_metadata=True
    )
    
    processed_results = []
    for match in initial_results.matches:
        view_count = match.metadata['total_view_count']
        similarity_score = match.score
        
        processed_results.append({
            'metadata': match.metadata,
            'similarity_score': similarity_score,
            'view_count': view_count,
            'hybrid_score': (1 - view_count_weight) * similarity_score + 
                          view_count_weight * (np.log1p(view_count) / 10)
        })
    
    processed_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return processed_results[:top_k]