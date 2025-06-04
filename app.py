import chainlit as cl
import asyncio
import os
from dotenv import load_dotenv
from web_scraper import WebScraper
from text_chunker import TextChunker
from neo4j_manager import Neo4jManager
from llm_client import LLMClient
import json
from typing import Optional

# Load environment variables
load_dotenv()

# Initialize global components
scraper = WebScraper()
chunker = TextChunker(max_chunk_size=500, overlap_size=50)
neo4j_manager = None
llm_client = None

def initialize_components():
    """Initialize Neo4j and LLM clients"""
    global neo4j_manager, llm_client
    
    try:
        # Initialize Neo4j
        neo4j_manager = Neo4jManager(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        )
        
        # Initialize LLM client
        llm_client = LLMClient(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile"
        )
        
        return True
    except Exception as e:
        print(f"Error initializing components: {e}")
        return False

@cl.on_chat_start
async def start():
    """Initialize the chat session"""
    await cl.Message(
        content="# 🚀 Web Content RAG System\n\nWelcome! I can help you analyze web content using a sophisticated RAG (Retrieval-Augmented Generation) system.\n\n## How it works:\n1. **Provide a URL** - I'll scrape and process the content\n2. **Ask questions** - I'll find relevant information and provide detailed answers\n\n## Features:\n- 📄 Smart content extraction and chunking\n- 🧠 Semantic search with embeddings\n- 🔗 Graph-based chunk relationships in Neo4j\n- 💬 Context-aware responses powered by Groq\n\n**To get started, please provide a URL you'd like me to analyze!**"
    ).send()
    
    # Initialize components
    if not initialize_components():
        await cl.Message(
            content="⚠️ **Error**: Failed to initialize system components. Please check your configuration in the `.env` file.",
            author="System"
        ).send()
        return
    
    # Test connections
    connection_status = []
    
    if neo4j_manager and neo4j_manager.test_connection():
        connection_status.append("✅ Neo4j Database")
    else:
        connection_status.append("❌ Neo4j Database")
    
    if llm_client and llm_client.test_connection():
        connection_status.append("✅ Groq API")
    else:
        connection_status.append("❌ Groq API")
    
    status_message = "## System Status:\n" + "\n".join(connection_status)
    await cl.Message(content=status_message, author="System").send()
    
    # Store initial state
    cl.user_session.set("processed_urls", set())
    cl.user_session.set("current_url", None)

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    user_input = message.content.strip()
    
    # Check if input looks like a URL
    if is_url(user_input):
        await process_url(user_input)
    else:
        # Handle as query
        await process_query(user_input)

async def process_url(url: str):
    """Process a URL and store its content"""
    try:
        # Show processing message
        processing_msg = await cl.Message(content="🔄 **Processing URL...**", author="System").send()
        
        # Check if URL already processed
        processed_urls = cl.user_session.get("processed_urls", set())
        if url in processed_urls:
            doc_info = neo4j_manager.get_document_info(url)
            if doc_info:
                # Create new message instead of updating
                await cl.Message(
                    content=f"✅ **URL Already Processed**: {doc_info['title']}\n\n*Total chunks: {doc_info['total_chunks']}*\n\nYou can now ask questions about this content!",
                    author="System"
                ).send()
                cl.user_session.set("current_url", url)
                return
        
        # Step 1: Scrape content
        await cl.Message(content="🔄 **Step 1/4**: Scraping web content...", author="System").send()
        
        scrape_result = await asyncio.to_thread(scraper.scrape_url, url)
        
        if scrape_result['status'] != 'success':
            await cl.Message(
                content=f"❌ **Scraping Failed**: {scrape_result['message']}",
                author="System"
            ).send()
            return
        
        title = scrape_result['title']
        content = scrape_result['content']
        
        if not content.strip():
            await cl.Message(
                content="❌ **Error**: No content could be extracted from the URL.",
                author="System"
            ).send()
            return
        
        # Step 2: Chunk content
        await cl.Message(content="🔄 **Step 2/4**: Creating logical chunks...", author="System").send()
        
        chunks = await asyncio.to_thread(chunker.chunk_text, content, title, url)
        
        if not chunks:
            await cl.Message(
                content="❌ **Error**: Failed to create content chunks.",
                author="System"
            ).send()
            return
        
        # Step 3: Store in Neo4j
        await cl.Message(content="🔄 **Step 3/4**: Storing in graph database...", author="System").send()
        
        success = await asyncio.to_thread(neo4j_manager.store_document_chunks, chunks, url, title)
        
        if not success:
            await cl.Message(
                content="❌ **Error**: Failed to store chunks in database.",
                author="System"
            ).send()
            return
        
        # Step 4: Complete
        await cl.Message(content="🔄 **Step 4/4**: Finalizing...", author="System").send()
        
        # Get chunk statistics
        stats = chunker.get_chunk_stats(chunks)
        
        # Update processed URLs
        processed_urls.add(url)
        cl.user_session.set("processed_urls", processed_urls)
        cl.user_session.set("current_url", url)
        
        # Success message
        success_content = f"""✅ **Processing Complete!**

**Document**: {title}
**URL**: {url}
**Total Chunks**: {stats['total_chunks']}
**Average Tokens per Chunk**: {stats['avg_tokens_per_chunk']:.0f}
**Total Tokens**: {stats['total_tokens']}

🎯 **Ready for Questions!** You can now ask me anything about this content."""

        await cl.Message(content=success_content, author="System").send()
        
        # Show available documents
        await show_available_documents()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Unexpected Error**: {str(e)}",
            author="System"
        ).send()

async def process_query(query: str):
    """Process a user query and provide response"""
    try:
        current_url = cl.user_session.get("current_url")
        processed_urls = cl.user_session.get("processed_urls", set())
        
        if not processed_urls:
            await cl.Message(
                content="📄 **Please provide a URL first** so I can analyze its content and answer your questions."
            ).send()
            return
        
        # Show thinking message
        thinking_msg = await cl.Message(content="🤔 **Thinking...**", author="Assistant").send()
        
        # Step 1: Analyze query intent
        await cl.Message(content="🤔 **Analyzing your question...**", author="Assistant").send()
        intent = await asyncio.to_thread(llm_client.analyze_query_intent, query)
        
        # Step 2: Retrieve relevant chunks
        await cl.Message(content="🔍 **Finding relevant information...**", author="Assistant").send()
        
        chunks = await asyncio.to_thread(
            neo4j_manager.semantic_search, 
            query, 
            current_url, 
            7
        )
        
        if not chunks:
            await cl.Message(
                content="❌ **No relevant information found** for your query. Try rephrasing your question or check if the content contains information about this topic.",
                author="Assistant"
            ).send()
            return
        
        # Step 3: Generate response
        await cl.Message(content="💭 **Generating response...**", author="Assistant").send()
        
        # Generate full response first
        full_response = ""
        try:
            # Get the generator and convert to list
            print("chunks:",chunks)
            response_generator = llm_client.generate_streaming_response(query, chunks)
            for chunk in response_generator:
                if isinstance(chunk, (list, tuple)):
                    for token in chunk:
                        full_response += str(token)
                else:
                    full_response += str(chunk)
        except Exception as e:
            print(f"Error generating response: {e}")
            full_response = "Sorry, I encountered an error generating the response."
        
        # Add source information
        source_info = f"\n\n---\n**📚 Sources**: {len(chunks)} relevant chunks found"
        if current_url:
            doc_info = neo4j_manager.get_document_info(current_url)
            if doc_info:
                source_info += f" from '{doc_info['title']}'"
        
        # Send the complete response
        await cl.Message(
            content=full_response + source_info,
            author="Assistant"
        ).send()
        
    except Exception as e:
        await cl.Message(
            content=f"❌ **Error processing query**: {str(e)}",
            author="System"
        ).send()

async def show_available_documents():
    """Show list of available documents"""
    try:
        documents = await asyncio.to_thread(neo4j_manager.get_all_documents)
        
        if not documents:
            return
        
        doc_list = ["## 📚 Available Documents:"]
        for i, doc in enumerate(documents[:5], 1):  # Show max 5
            doc_list.append(f"{i}. **{doc['title']}** ({doc['total_chunks']} chunks)")
        
        if len(documents) > 5:
            doc_list.append(f"*... and {len(documents) - 5} more documents*")
        
        await cl.Message(
            content="\n".join(doc_list),
            author="System"
        ).send()
        
    except Exception as e:
        print(f"Error showing documents: {e}")

def is_url(text: str) -> bool:
    """Check if text looks like a URL"""
    return (
        text.startswith(('http://', 'https://')) or
        ('.' in text and ' ' not in text and len(text) > 5)
    )

@cl.on_chat_end
async def end():
    """Clean up when chat ends"""
    print("Chat session ended")

if __name__ == "__main__":
    # Verify environment variables
    required_vars = ['NEO4J_PASSWORD', 'GROQ_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing required environment variables: {', '.join(missing_vars)}")
        print("Please check your .env file")
        exit(1)
    
    print("🚀 Starting Web Content RAG System...")
    print("📊 Initializing components...")
    
    # Run the Chainlit app
    cl.run()