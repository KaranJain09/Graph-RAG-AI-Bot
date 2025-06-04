import re
import tiktoken
from typing import List, Dict
import hashlib

class TextChunker:
    def __init__(self, max_chunk_size: int = 500, overlap_size: int = 50):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, content: str, title: str = "", url: str = "") -> List[Dict]:
        """
        Chunk text content logically based on structure and semantic meaning
        
        Args:
            content (str): Text content to chunk
            title (str): Document title
            url (str): Source URL
            
        Returns:
            List[Dict]: List of chunk dictionaries with metadata
        """
        # First, split by major sections (headers)
        sections = self._split_by_headers(content)
        
        chunks = []
        chunk_id = 0
        
        for section in sections:
            section_chunks = self._process_section(section, chunk_id)
            
            for chunk_data in section_chunks:
                chunk_data.update({
                    'title': title,
                    'url': url,
                    'chunk_id': f"{url}_{chunk_id}",
                    'hash': self._generate_hash(chunk_data['content'])
                })
                chunks.append(chunk_data)
                chunk_id += 1
        
        return chunks
    
    def _split_by_headers(self, content: str) -> List[Dict]:
        """Split content by markdown headers"""
        sections = []
        current_section = {
            'header': '',
            'level': 0,
            'content': '',
            'start_pos': 0
        }
        
        lines = content.split('\n')
        current_content = []
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)', line)
            
            if header_match:
                # Save previous section
                if current_content or current_section['header']:
                    current_section['content'] = '\n'.join(current_content)
                    if current_section['content'].strip():
                        sections.append(current_section.copy())
                
                # Start new section
                level = len(header_match.group(1))
                header_text = header_match.group(2)
                
                current_section = {
                    'header': header_text,
                    'level': level,
                    'content': '',
                    'start_pos': i
                }
                current_content = [line]
            else:
                current_content.append(line)
        
        # Add last section
        if current_content:
            current_section['content'] = '\n'.join(current_content)
            if current_section['content'].strip():
                sections.append(current_section)
        
        # If no headers found, treat entire content as one section
        if not sections:
            sections = [{
                'header': 'Main Content',
                'level': 1,
                'content': content,
                'start_pos': 0
            }]
        
        return sections
    
    def _process_section(self, section: Dict, start_chunk_id: int) -> List[Dict]:
        """Process a section and create chunks"""
        content = section['content']
        header = section['header']
        level = section['level']
        
        # Count tokens
        token_count = len(self.tokenizer.encode(content))
        
        if token_count <= self.max_chunk_size:
            # Section fits in one chunk
            return [{
                'content': content,
                'header': header,
                'level': level,
                'token_count': token_count,
                'chunk_index': 0,
                'total_chunks': 1
            }]
        
        # Section needs to be split
        return self._split_large_section(section, start_chunk_id)
    
    def _split_large_section(self, section: Dict, start_chunk_id: int) -> List[Dict]:
        """Split large section into smaller chunks"""
        content = section['content']
        header = section['header']
        level = section['level']
        
        # Split by paragraphs first
        paragraphs = re.split(r'\n\s*\n', content)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            para_tokens = len(self.tokenizer.encode(paragraph))
            
            # If paragraph is too large, split it further
            if para_tokens > self.max_chunk_size:
                # Save current chunk if not empty
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk), 
                        header, 
                        level, 
                        chunk_index, 
                        current_tokens
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_tokens = 0
                
                # Split large paragraph by sentences
                sentence_chunks = self._split_by_sentences(paragraph, header, level, chunk_index)
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                
            elif current_tokens + para_tokens > self.max_chunk_size:
                # Current chunk is full, start new one
                if current_chunk:
                    chunks.append(self._create_chunk_dict(
                        '\n\n'.join(current_chunk), 
                        header, 
                        level, 
                        chunk_index, 
                        current_tokens
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_content(current_chunk)
                current_tokens = len(self.tokenizer.encode('\n\n'.join(current_chunk))) if current_chunk else 0
                current_chunk.append(paragraph)
                current_tokens += para_tokens
                
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                '\n\n'.join(current_chunk), 
                header, 
                level, 
                chunk_index, 
                current_tokens
            ))
        
        # Update total chunks count
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks
        
        return chunks
    
    def _split_by_sentences(self, text: str, header: str, level: int, start_index: int) -> List[Dict]:
        """Split text by sentences when paragraphs are too large"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_index = start_index
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_tokens = len(self.tokenizer.encode(sentence))
            
            if current_tokens + sentence_tokens > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(self._create_chunk_dict(
                    ' '.join(current_chunk), 
                    header, 
                    level, 
                    chunk_index, 
                    current_tokens
                ))
                chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_sentences(current_chunk)
                current_tokens = len(self.tokenizer.encode(' '.join(current_chunk))) if current_chunk else 0
            
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Add final chunk
        if current_chunk:
            chunks.append(self._create_chunk_dict(
                ' '.join(current_chunk), 
                header, 
                level, 
                chunk_index, 
                current_tokens
            ))
        
        return chunks
    
    def _get_overlap_content(self, chunks: List[str]) -> List[str]:
        """Get overlap content from previous chunks"""
        if not chunks:
            return []
        
        # Take last paragraph for overlap
        overlap_text = chunks[-1]
        overlap_tokens = len(self.tokenizer.encode(overlap_text))
        
        if overlap_tokens <= self.overlap_size:
            return [overlap_text]
        
        # If last paragraph is too long, take last few sentences
        sentences = re.split(r'(?<=[.!?])\s+', overlap_text)
        return self._get_overlap_sentences(sentences)
    
    def _get_overlap_sentences(self, sentences: List[str]) -> List[str]:
        """Get overlap sentences"""
        if not sentences:
            return []
        
        overlap = []
        tokens = 0
        
        # Take sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            sentence_tokens = len(self.tokenizer.encode(sentence))
            if tokens + sentence_tokens > self.overlap_size:
                break
            overlap.insert(0, sentence)
            tokens += sentence_tokens
        
        return overlap
    
    def _create_chunk_dict(self, content: str, header: str, level: int, chunk_index: int, token_count: int) -> Dict:
        """Create chunk dictionary with metadata"""
        return {
            'content': content,
            'header': header,
            'level': level,
            'token_count': token_count,
            'chunk_index': chunk_index,
            'total_chunks': 1  # Will be updated later
        }
    
    def _generate_hash(self, content: str) -> str:
        """Generate hash for chunk content"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_chunk_stats(self, chunks: List[Dict]) -> Dict:
        """Get statistics about chunks"""
        if not chunks:
            return {}
        
        total_tokens = sum(chunk['token_count'] for chunk in chunks)
        avg_tokens = total_tokens / len(chunks)
        
        return {
            'total_chunks': len(chunks),
            'total_tokens': total_tokens,
            'avg_tokens_per_chunk': avg_tokens,
            'max_tokens': max(chunk['token_count'] for chunk in chunks),
            'min_tokens': min(chunk['token_count'] for chunk in chunks)
        }