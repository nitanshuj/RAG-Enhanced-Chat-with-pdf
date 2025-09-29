# Multi-Category Document Understanding - Q&A Chatbot
-------

**IMPORTANT: This application only accepts PDF documents. All input must be in PDF format.**

`Keywords`: document understanding, Q&A chatbot, multi-category processing, research papers, articles, books, receipts, terms and conditions (all in PDF format)

`Tech Stack`:
- Python
- LangChain
- Chroma DB Cloud (vector store and retrieval)
- PDF parsers (exclusively)
- Embeddings (type to be decided)
- AI/ML API for LLM (GPT 4 nano)
- Streamlit Frontend (easy to deploy and share)

### `Note`
- We will not use Ollama or any kind of local hosting of LLMs, rather we will use AI/ML API for LLM (GPT 4 nano).
- We will use good industry standards for embeddings, chunking, and vector stores.
- We will make a good production-grade code.
- We will be using Agentic RAG approach, since it will be a chatbot - the memory of chat will be only present for the session.
- We will use a good prompt engineering approach to make the chatbot work well.
- We will use a good chunking approach for each document type.
- We will use a good metadata approach for each document type.
- We will use a good retrieval approach for each document type.
- 
-------

## Document Categories and Unique Processing

The chatbot supports multiple document types, each with specialized processing approaches:

### Document Categories (User-Selected via Dropdown):
- **Research Paper**: Academic papers, journals, conference proceedings (PDF format)
- **Article**: News articles, blog posts, magazine content (PDF format)
- **Book**: Textbooks, novels, reference materials (PDF format)
- **Other**: General documents not fitting specific categories (PDF format)
- **Receipts**: Purchase receipts, invoices, financial documents (PDF format)
- **Terms & Conditions**: Legal documents, contracts, policy documents (PDF format)

**Note**: Users will select the document category from a dropdown menu during upload.

### Category-Specific Processing:

#### Research Paper Processing
- **Structure-aware chunking**: By sections (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
- **Metadata extraction**: Authors, journal, publication date, citations
- **Specialized prompts**: Focus on methodology, findings, statistical significance
- **Citation tracking**: Link references and maintain academic context

#### Article Processing
- **Topic-based chunking**: By themes and key points
- **Timeline extraction**: Date references and chronological events
- **Fact verification**: Cross-reference claims and sources
- **Summary-focused**: Emphasis on main points and conclusions

#### Book Processing
- **Chapter-based segmentation**: Hierarchical structure (chapters, sections, subsections)
- **Character/concept tracking**: Maintain consistency across long narratives
- **Progressive context**: Build understanding as story/content develops
- **Index utilization**: Leverage table of contents and index for navigation

#### Other Document Processing
- **Adaptive chunking**: Dynamic approach based on document structure
- **General metadata**: Basic document information and properties
- **Flexible prompting**: Generic question-answering approach
- **Content-type detection**: Identify specific patterns for better processing

#### Receipts Processing
- **Structured data extraction**: Items, prices, dates, vendors
- **Financial categorization**: Expense types and tax information
- **OCR optimization**: Enhanced text recognition for printed receipts
- **Transaction tracking**: Link related purchases and patterns

#### Terms & Conditions Processing
- **Legal clause identification**: Rights, obligations, limitations
- **Risk assessment**: Highlight important terms and conditions
- **Plain language translation**: Simplify complex legal language
- **Compliance checking**: Identify key regulatory requirements

---------------

## Agentic RAG - Step-by-Step Agentic RAG Chatbot Workflow

### 1. Document Ingestion and Category-Specific Chunking

- **PDF-only input**: Only PDF documents are accepted for processing
- **User-selected classification**: Users select document category from dropdown during upload (Research Paper, Article, Book, Other, Receipts, Terms & Conditions)
- **Category-specific parsing**: Extract text from PDF using appropriate methods for each user-selected document type
- **Intelligent chunking**: Apply category-specific chunking strategies:
  - Research Papers: Section-based (Abstract, Methods, Results, etc.)
  - Articles: Topic and paragraph-based
  - Books: Chapter and section-based
  - Receipts: Line-item and transaction-based
  - Terms & Conditions: Clause and section-based
  - Other: Adaptive paragraph-based
- **Enhanced metadata**: Store category-specific metadata (citations for papers, transaction details for receipts, legal clauses for T&C)

### 2. Chunk Embedding and Vector Store
- Embed all chunks: Generate embeddings of each chunk using a semantic model.
- Persist in Chroma DB Cloud: Store all embeddings and metadata in Chroma Cloud for scalable vector storage and retrieval.

### 3. Chatbot Session Initialization
- Session memory/context: For each user/chatbot session, store:
- Chat history (previous queries and answers)
- Any insight or facts the agent has already found

### 4. User Question/Input
- User asks any question (can be fact-based, complex, or multi-step):
- E.g., “Summarize the methods, then compare findings with discussion.”

### 5. Agentic Intent Recognition
The agent parses the query:
- Checks for complex requests, reasoning, chaining, or multiple subtasks.
- Splits the main question into subtasks if needed:
    - Task 1: Summarize methods
    - Task 2: Extract findings from results
    - Task 3: Compare findings with discussion

### 6. Contextual Retrieval (for Each Subtask)

For each subtask:
- Embed sub-task question
- Search Chroma DB Cloud for relevant chunks (use metadata filtering to target specific sections and user-selected categories)
- Retrieve top N relevant chunks using similarity search

### 7. Dynamic Prompt Construction
For each agent step:

- Construct user prompt:

```
Use this context from the 'Methods' section to summarize the methodology:
[methods chunks...]
```

- For chaining:
    - Store result in session context/memory
    - Use outputs as inputs for next step

- Example prompt for chaining:


```
First, summarize the Methods based on this context.
Then, using Results and Discussion sections, compare findings.
If you have already found important details in earlier steps, refer to them.
```

### 8. Multi-Step LLM Invocation via AI/ML API
Send prompt to cloud-based LLM (GPT-4 nano) via AI/ML API (http://aimlapi.com/) for each reasoning step.

**API Integration Features:**
- **Cloud-based processing**: No local compute requirements, scalable and reliable
- **Model selection**: Access to GPT-4 nano and other state-of-the-art models
- **Rate limiting**: Built-in request management and optimization
- **Error handling**: Robust retry mechanisms and fallback strategies

Agent can ask the LLM for clarification or more evidence if its output is not sufficient ("Was this methodology unique compared to the discussion section's comments?").

Store each output as part of chatbot context for user reference and follow-ups.

### 9. Output and Conversation Continuation

Present results to the user in the chat interface.

Allow follow-ups:
User can clarify, ask further, or “dig deeper”—previous context/history is considered.
Agent maintains context to avoid repeating work, reason with earlier answers, and chain queries.


### 10. Conversation Memory / Multi-Turn QA
As the conversation progresses, the agent:
Uses history for context to improve retrieval and answer quality.
Can refer back to earlier steps or answers.
Adapts retrieval to cover missing information or fill knowledge gaps.

-------

## Key Differences From Simple RAG
1. **Chaining**: Agent splits questions into subtasks, sequences retrievals, and composes answers from multi-step reasoning.
2. **Memory**: Each chat session keeps context, earlier answers, and facts.
3. **Interaction**: System asks clarifying questions, follows up, adapts approach based on user responses.
4. **Multi-ste**p: The final answer may be composed from several LLM calls, each focused on different sections or tasks.

-------

### Example Chatbot Agentic RAG Use Cases by Document Type

#### Research Paper Example
`User`: "Summarize the methods. Then, are any findings challenged in the discussion?"

`Agent`:
1. Retrieves and summarizes methods section using academic-focused prompts
2. Retrieves results and discussion sections
3. Searches for challenges/contradictions to findings in discussion
4. Responds: "The methods used were X... Findings Y were challenged in discussion by Z..."

#### Receipt Example
`User`: "What did I spend on groceries last month and which store was cheapest?"

`Agent`:
1. Filters receipts by date range and grocery categories
2. Extracts vendor information and totals
3. Performs comparative analysis across stores
4. Responds: "You spent $X on groceries. Store Y was cheapest at $Z per visit on average..."

#### Terms & Conditions Example
`User`: "What are my cancellation rights and any associated fees?"

`Agent`:
1. Identifies cancellation-related clauses using legal pattern recognition
2. Extracts fee structures and timeframes
3. Translates legal language to plain English
4. Responds: "You can cancel within X days. Fees apply as follows: Y..."

#### Book Example
`User`: "How does the main character develop throughout the story?"

`Agent`:
1. Tracks character mentions across chapters chronologically
2. Identifies key character development moments
3. Analyzes character arc progression
4. Responds: "The character evolves from X to Y, with key changes in chapters Z..."

-------

### Benefits

- Handles complex, multi-step questions and dialogue.
- Leverages prior context—more like a true research assistant.
- Supports clarifications, corrections, deep dives, and sophisticated QA flows.

-------

This is a full agentic loop—multi-hop, conversation-aware, and with chaining—using cloud-based LLMs via AI/ML API as the final answer engine. Perfect for multi-category document analytics with production-grade reliability!