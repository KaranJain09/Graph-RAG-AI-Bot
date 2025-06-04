from groq import Groq
from typing import List, Dict, Optional
import json

class LLMClient:
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        self.client = Groq(api_key=api_key)
        self.model = model
        
    def generate_response(self, query: str, chunks: List[Dict], context: Optional[str] = None) -> str:
        """
        Generate response using retrieved chunks and query
        
        Args:
            query: User query
            chunks: Retrieved chunks from Neo4j
            context: Additional context
            
        Returns:
            Generated response
        """
        try:
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build user message with context
            user_message = self._build_user_message(query, chunks, context)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                stream=False
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def generate_streaming_response(self, query: str, chunks: List[Dict], context: Optional[str] = None):
        """
        Generate streaming response for real-time display
        
        Args:
            query: User query
            chunks: Retrieved chunks from Neo4j
            context: Additional context
            
        Yields:
            Response chunks
        """
        try:
            print("Chunks:" , chunks)
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build user message with context
            user_message = self._build_user_message(query, chunks, context)
            
            # Make streaming API call
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=2048,
                temperature=0.3,
                top_p=0.9,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            print(f"Error in streaming response: {e}")
            yield "I apologize, but I encountered an error while generating the response. Please try again."
    
    def _build_system_prompt(self) -> str:
        """Build comprehensive system prompt"""
        return """You are an intelligent assistant that provides clear, direct answers to user questions based on relevant information from documents.

Your task is to:
1. Answer the user's question directly and naturally
2. Provide comprehensive information based on the content provided
3. Use a conversational and professional tone
4. Structure your response clearly and logically
5. Include relevant details that add value to your answer

Guidelines:
- Give direct, natural answers without mentioning "chunks" or technical retrieval details
- Write as if you're an expert explaining the topic to someone interested
- Use markdown formatting sparingly and only when it enhances readability
- If you don't have enough information, briefly mention what's missing
- Focus on being helpful and informative rather than technical
- Don't use bullet points or structured lists unless the question specifically asks for them
- Write in flowing paragraphs that feel natural to read

Remember: Provide authoritative, well-informed answers that directly address what the user wants to know."""
    
    def _build_user_message(self, query: str, chunks: List[Dict], context: Optional[str] = None) -> str:
        """Build user message with query and context chunks"""
        message_parts = []
        
        # Add context if provided
        if context:
            message_parts.append(f"Additional Context: {context}\n")
        
        # Add retrieved content in a natural way
        if chunks:
            message_parts.append("Here is the relevant information from the document:\n")
            
            for i, chunk in enumerate(chunks, 1):
                # Don't mention chunk numbers or similarity scores in the content
                content_section = chunk['content'].strip()
                if chunk.get('header') and chunk['header'] != 'Unknown':
                    message_parts.append(f"From section '{chunk['header']}':")
                message_parts.append(content_section)
                message_parts.append("")  # Add spacing
        else:
            message_parts.append("No relevant information was found in the document for this query.")
        
        # Add user query
        message_parts.append(f"Question: {query}")
        
        # Add simple instructions
        message_parts.append("""
Please provide a clear, direct answer to this question based on the information above. Write naturally and conversationally, as if you're explaining this to someone who's genuinely curious about the topic. Don't mention technical details about how you found the information - just focus on giving a helpful, informative response.""")
        
        return "\n".join(message_parts)
    
    def summarize_chunks(self, chunks: List[Dict]) -> str:
        """Generate a summary of the provided chunks"""
        try:
            if not chunks:
                return "No content chunks available for summarization."
            
            # Combine all chunk content
            combined_content = "\n\n---\n\n".join([
                f"Section: {chunk.get('header', 'Unknown')}\n{chunk['content']}" 
                for chunk in chunks
            ])
            
            system_prompt = """You are a helpful assistant that creates concise, informative summaries of text content. 
            Your task is to summarize the provided content chunks while preserving key information and maintaining readability."""
            
            user_message = f"""Please provide a concise summary of the following content:

{combined_content}

Summary requirements:
- Capture the main points and key information
- Maintain logical flow and structure
- Use bullet points or numbered lists where appropriate
- Keep it informative but concise
- Highlight any important conclusions or insights"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Unable to generate summary due to an error."
    
    def analyze_query_intent(self, query: str) -> Dict:
        """Analyze query to understand user intent"""
        try:
            system_prompt = """Analyze the user's query and determine their intent. Respond with a JSON object containing:
            - intent_type: question, request_summary, request_specific_info, comparison, analysis
            - key_topics: list of main topics/keywords
            - specificity: high, medium, low
            - expected_response_type: detailed_answer, summary, list, comparison, analysis"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Query: {query}"}
                ],
                max_tokens=256,
                temperature=0.1
            )
            
            try:
                return json.loads(response.choices[0].message.content.strip())
            except json.JSONDecodeError:
                return {
                    "intent_type": "question",
                    "key_topics": query.split(),
                    "specificity": "medium",
                    "expected_response_type": "detailed_answer"
                }
                
        except Exception as e:
            print(f"Error analyzing query intent: {e}")
            return {
                "intent_type": "question",
                "key_topics": query.split(),
                "specificity": "medium",
                "expected_response_type": "detailed_answer"
            }
    
    def test_connection(self) -> bool:
        """Test Groq API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello, this is a test."}],
                max_tokens=10
            )
            return True
        except Exception as e:
            print(f"Groq API test failed: {e}")
            return False