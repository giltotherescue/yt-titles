# YouTube Title Vector Search

## Simple Explanation
This tool helps you find successful YouTube video titles that are similar to any title you input. It's like having a smart assistant that can tell you "here are some high-performing videos with similar titles to what you're thinking of using."

For example, if you input "funny cat fails", it might show you similar successful titles like "hilarious feline mishaps" or "cats being clumsy compilation" along with their view counts.

## Technical Explanation
This project uses semantic search to find similar YouTube titles and ranks them based on a hybrid scoring system that considers both:
- Semantic similarity (using OpenAI's text-embedding-3-small model)
- Performance metrics (view counts)

The system:
1. Converts titles into vector embeddings using OpenAI's API
2. Stores these vectors in a Pinecone vector database
3. Performs similarity searches with a custom scoring algorithm that balances semantic similarity with view count performance

### Architecture
- OpenAI API: Converts text to vector embeddings
- Pinecone: Vector database for efficient similarity search
- Python: Core implementation using pandas for data handling
- Hybrid scoring: Combines cosine similarity with log-normalized view counts

## Setup

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key
- A Pinecone index named "yt-titles-1" using:
  - Metric: cosine
  - Dimensions: 1536
  - Model: text-embedding-3-small

### Installation

1. Clone this repository:
    git clone [repository-url]
    cd youtube-title-vector

2. Create a virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install dependencies:
    pip install pandas pinecone-client openai python-dotenv numpy

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Fill in your API keys in `.env`

### Data Format
Your `youtube_data.csv` should have these columns:
- `yt_video_id`: Unique identifier for the video
- `title`: The video title
- `total_view_count`: Number of views
- `is_short`: Boolean indicating if it's a YouTube Short

### Usage

1. Load data into Pinecone:
    python load_data.py

2. Run test queries:
    python test_queries.py

## Example Output

Testing query: 'funny cat videos'
============================================================
Top performing similar titles:
------------------------------------------------------------
1. Title: Funny cats and dogs - New Funny animal videos 2024🤣
Views: 872
Type: Regular video
Similarity Score: 0.856
Hybrid Score: 0.742

## Files
- `load_pinecone.py`: Core functionality for embeddings and vector operations
- `load_data.py`: Script to load CSV data into Pinecone
- `test_queries.py`: Script to test similarity searches
- `.env.example`: Template for environment variables

## Security Notes
- Never commit your `.env` file
- Regenerate API keys if they're ever exposed
- Use virtual environment to manage dependencies

## Contributing
Feel free to submit issues and enhancement requests!
