# ChromaDB Cloud Setup Instructions

## Overview
This application uses ChromaDB Cloud for vector storage instead of local persistence. Follow these steps to set up your ChromaDB Cloud instance.

## Step 1: Access Your ChromaDB Cloud Dashboard

Visit: **https://www.trychroma.com/**

Sign up for a ChromaDB Cloud account and access your personalized dashboard.

## Step 2: Create a Database

1. **Sign in** to your ChromaDB Cloud account
2. **Create a new database** named `rag_documents` (or use the name specified in your `.env` file)
3. **Copy your API credentials** from the dashboard

## Step 3: Configure Environment Variables

Your `.env` file is already configured with the following variables:

```env
# ChromaDB Cloud Configuration
CHROMA_API_KEY = "your_api_key_here"
CHROMA_TENANT = "your_tenant_name"
CHROMA_DATABASE = "rag_documents"
```

**Make sure these values match your actual ChromaDB Cloud setup:**

- `CHROMA_API_KEY`: Your API key from the ChromaDB Cloud dashboard
- `CHROMA_TENANT`: Your tenant ID (usually matches your username)
- `CHROMA_DATABASE`: The database name you created

## Step 4: Verify Connection

Test your connection by running:

```bash
uv run python -c "from src.vector_store import VectorStore; vs = VectorStore(); print('Connection successful!')"
```

## Step 5: Run Tests

Test the vector store functionality:

```bash
uv run python tests/run_tests.py
```

Or run specific vector store tests:

```bash
uv run python -m unittest tests.test_vector_store
```

## Troubleshooting

### Common Issues:

1. **Invalid API Key**: Make sure your API key is correct and has not expired
2. **Wrong Tenant**: Verify the tenant name matches your ChromaDB Cloud account
3. **Database Not Found**: Ensure the database exists in your ChromaDB Cloud dashboard
4. **Network Issues**: Check your internet connection and firewall settings

### Error Messages:

- `CHROMA_API_KEY environment variable is required` → Check your `.env` file
- `CHROMA_TENANT environment variable is required` → Add tenant to `.env` file
- `CHROMA_DATABASE environment variable is required` → Add database name to `.env` file
- `Vector store initialization failed` → Check your credentials and network connection

## Features Available

With ChromaDB Cloud setup, you can:

- ✅ Store document chunks with embeddings
- ✅ Perform similarity search across documents
- ✅ Filter by document categories
- ✅ Delete individual documents or entire categories
- ✅ Get collection statistics and health checks
- ✅ No local storage or disk space requirements

## Security Notes

- Keep your API key secure and never commit it to version control
- The `.env` file is already added to `.gitignore`
- Your data is stored securely in ChromaDB Cloud with enterprise-grade security

## Getting Started

Once setup is complete, you can start using the application:

```bash
uv run streamlit run main.py
```

The application will automatically connect to your ChromaDB Cloud instance and create collections as needed.