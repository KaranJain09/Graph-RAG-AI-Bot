# 🚀 Graph-RAG-AI-Bot

**Chat with any website using AI-powered, graph-based retrieval.**
Transform static web pages into dynamic conversational experiences with the power of LLMs and knowledge graphs.([medium.com][1])

---

## 📚 Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Architecture](#architecture)
* [Installation](#installation)
* [Usage](#usage)
* [Configuration](#configuration)
* [Examples](#examples)
* [Troubleshooting](#troubleshooting)
* [Contributors](#contributors)
* [License](#license)

---

## 🧠 Introduction

**Graph-RAG-AI-Bot** is an innovative AI assistant that allows users to interact with any website through natural language conversations. Instead of manually reading through extensive web content, users can simply input a website URL, and the bot will provide concise, accurate answers to their queries by leveraging advanced AI techniques.

---

## ✨ Features

* **Web Scraping**: Automatically extracts content from any given website URL.
* **Markdown Conversion**: Converts scraped HTML content into structured Markdown format while preserving the original layout.
* **Logical Chunking**: Breaks down content into meaningful sections for efficient processing.
* **Knowledge Graph Storage**: Stores content chunks in a Neo4j graph database, capturing relationships between different sections.
* **Semantic Search**: Retrieves relevant content chunks based on user queries.
* **LLM Integration**: Utilizes Groq's LLaMA 3.3 model to generate accurate and context-aware responses.
* **Interactive Chat Interface**: Provides a user-friendly interface for seamless interactions.

---

## 🏗️ Architecture

1. **Input**: User provides a website URL.
2. **Scraping**: The `web_scraper.py` module uses BeautifulSoup to extract HTML content.
3. **Conversion**: HTML content is converted to Markdown format, preserving structure.
4. **Chunking**: The `text_chunker.py` module divides the content into logical chunks.
5. **Graph Storage**: Chunks are stored in Neo4j using the `neo4j_manager.py` module, establishing relationships between them.
6. **Query Processing**: User queries are processed, and relevant chunks are retrieved from the graph database.
7. **Response Generation**: The `llm_client.py` module sends the retrieved information to the LLaMA 3.3 model via Groq's API to generate responses.
8. **Output**: The generated response is presented to the user through the chat interface.([arxiv.org][2])

---

## 🛠️ Installation

### Prerequisites

* **Python 3.8+**
* **Neo4j Database**
* **Groq API Access** (for LLaMA 3.3 model)

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/KaranJain09/Graph-RAG-AI-Bot.git
   cd Graph-RAG-AI-Bot
   ```



2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```



3. **Configure Environment Variables**

   Create a `.env` file in the root directory and add the following:

   ```env
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   GROQ_API_KEY=your_groq_api_key
   ```



4. **Run the Application**

   ```bash
   python app.py
   ```



---

## 💡 Usage

1. **Start the Application**

   Ensure Neo4j is running and execute:

   ```bash
   chainlit run app.py
   ```



2. **Interact with the Bot**

   * Open your browser and navigate to `http://localhost:8000` (or the specified port).
   * Enter the URL of the website you wish to interact with.
   * Ask questions in natural language; the bot will provide concise answers based on the website's content.

---

## ⚙️ Configuration

All configurations are managed via the `config.py` file and environment variables. Ensure that the Neo4j credentials and Groq API key are correctly set in the `.env` file.

---

## 🧪 Examples

* **Input**: `https://example.com`
* **User Query**: "What services does this company offer?"
* **Bot Response**: "The company offers web development, mobile app development, and digital marketing services."

---

## 🛠️ Troubleshooting

* **Neo4j Connection Errors**: Ensure Neo4j is running and the credentials in the `.env` file are correct.
* **Groq API Issues**: Verify that your API key is valid and has the necessary permissions.
* **Module Errors**: Ensure all dependencies are installed by running `pip install -r requirements.txt`.

---

## 👥 Contributors

* **Karan Jain** - [GitHub](https://github.com/KaranJain09)

---

