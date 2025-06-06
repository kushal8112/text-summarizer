# AI Text Summarizer
A modern web application that provides both extractive and abstractive text summarization powered by AI. Built with React, FastAPI, and Transformers.

## 🚀 Features
- **Dual Summarization Modes**:
  - **Quick Summary**: Fast, extractive summarization using ranking method
  - **Smart Summary**: AI-powered abstractive summarization using T5 model
- **Adjustable Length**: Customize summary length (short, medium, long)
- **Rich Text Editing**: Edit and format generated summaries
- **Dark Mode**: Eye-friendly dark theme
- **Copy to Clipboard**: One-click copy of generated summaries

## 🛠️ Tech Stack

### Frontend
- React 18
- Tailwind CSS for styling
- React Icons
- React Toastify for notifications
- Axios for API calls

### Backend
- FastAPI
- Python 3.9
- NLTK for text processing
- Transformers (Hugging Face) for AI summarization
- NumPy/SciPy for computations

### DevOps
- Docker & Docker Compose
- Uvicorn ASGI server

## 📦 Prerequisites
- Node.js (v16+)
- Python 3.9+
- Docker (optional)

## 🚀 Getting Started

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/kushal8112/ai-text-summarizer.git
   cd ai-text-summarizer

## Usage
1. Enter your text in the input area
2. Select the type of summary
3. Adjust the summary length
4. Click "Generate Summary"
5. Format the summary using the formatting buttons
6. Copy the summary to clipboard if needed

## Development
- Frontend runs on: http://127.0.0.1:3000
- Backend API runs on: http://127.0.0.1:8000
- API Docs: http://127.0.0.1:8000/docs

## API Endpoints
  ### `POST /summarize`
  Generate a summary of the provided text using either quick (extractive) or smart (abstractive) method.

  #### Request Body
```json
  {
    "text": "The full text to be summarized...",
    "summary_type": "quick|smart",
    "length": "short|medium|long"
  }
  ```

  #### Response
  ```json
  {
    "summary": "The generated summary..."
  }
  ```

## Project Structure

windsurf-project/
│
├── app/                         # Backend (FastAPI)
│   ├── __init__.py
│   ├── main.py                  # FastAPI app and routes
│   ├── summarizers.py           # Core summarization logic
│   └── requirements.txt
│
├── frontend/                    # Frontend (React)
│   ├── public/
│   │   └── index.html
│   └── src/
│       ├── App.js               # Main React component
│       ├── index.js             # Entry point
│       ├── index.css            # Global styles
│       └── assets/              # Static assets
│
├── docker-compose.yml           # Docker Compose config
└── Dockerfile                   # Backend Dockerfile

## License
MIT License

