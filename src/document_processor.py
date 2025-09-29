import PyPDF2
from typing import List, Dict, Any, Union
from io import BytesIO
import re
from datetime import datetime


class DocumentProcessor:
    """PDF-only document processor with category-specific processing"""

    def __init__(self):
        """Initialize the document processor"""
        self.supported_types = ['pdf']

    def extract_text(self, file_buffer: Union[BytesIO, bytes]) -> str:
        """
        Extract text from PDF file buffer

        Args:
            file_buffer: PDF file buffer from Streamlit upload

        Returns:
            Extracted text content
        """
        try:
            # Handle different input types
            if hasattr(file_buffer, 'read'):
                # Already a file-like object
                pdf_buffer = file_buffer
            elif isinstance(file_buffer, (bytes, memoryview)):
                # Convert bytes or memoryview to BytesIO
                if isinstance(file_buffer, memoryview):
                    pdf_buffer = BytesIO(file_buffer.tobytes())
                else:
                    pdf_buffer = BytesIO(file_buffer)
            else:
                # Try to convert to BytesIO
                pdf_buffer = BytesIO(bytes(file_buffer))

            pdf_reader = PyPDF2.PdfReader(pdf_buffer)
            text_content = ""

            # Extract text from all pages
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        text_content += f"\n--- Page {page_num + 1} ---\n"
                        text_content += page_text
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num + 1}: {e}")
                    continue

            if not text_content.strip():
                raise ValueError("No text could be extracted from the PDF")

            return text_content.strip()

        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

    def simple_chunk(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Simple text chunking with overlap (used for non-research paper documents)

        Args:
            text: Text to chunk
            chunk_size: Target size of each chunk (in characters)
            overlap: Character overlap between chunks

        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []

        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())

        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If we're not at the end, try to break at sentence or word boundary
            if end < len(text):
                # Look for sentence boundary
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + chunk_size // 2:
                    end = sentence_end + 1
                else:
                    # Look for word boundary
                    word_end = text.rfind(' ', start, end)
                    if word_end > start + chunk_size // 2:
                        end = word_end

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def heading_based_chunk(self, text: str, category: str, max_chunk_size: int = 2000, overlap: int = 50) -> List[Dict[str, str]]:
        """
        Advanced heading-based chunking for research papers with fallback to simple chunking

        Args:
            text: Text to chunk
            category: Document category
            max_chunk_size: Maximum size for a single chunk before sub-chunking
            overlap: Character overlap between chunks

        Returns:
            List of chunk dictionaries with metadata
        """
        if category != "Research Paper":
            # Use simple chunking for non-research papers
            simple_chunks = self.simple_chunk(text, chunk_size=500, overlap=overlap)
            return [{"content": chunk, "section": "content", "chunk_type": "simple"} for chunk in simple_chunks]

        # Research paper specific heading-based chunking
        return self._research_paper_chunking(text, max_chunk_size, overlap)

    def _research_paper_chunking(self, text: str, max_chunk_size: int, overlap: int) -> List[Dict[str, str]]:
        """
        Specialized chunking for research papers using advanced heading detection
        """
        # Detect all potential headings using multiple heuristics
        potential_headings = self._detect_headings(text)

        # Create sections based on detected headings
        sections = self._create_sections_from_headings(text, potential_headings)

        # If no headings were found, fall back to content-based detection
        if len(sections) <= 1:
            sections = self._fallback_content_based_sections(text)

        chunks = []

        # Process each section
        for section_info in sections:
            section_content = section_info['content']
            section_name = section_info['section']

            # If section is small enough, keep as single chunk
            if len(section_content) <= max_chunk_size:
                chunks.append({
                    'content': section_content,
                    'section': section_name,
                    'chunk_type': 'section',
                    'size': len(section_content),
                    'confidence': section_info.get('confidence', 0.5)
                })
            else:
                # Section is too large, sub-chunk it with overlap
                sub_chunks = self._sub_chunk_large_section(section_content, section_name, max_chunk_size, overlap)
                chunks.extend(sub_chunks)

        # Add overlap between different sections
        chunks_with_overlap = self._add_inter_section_overlap(chunks, overlap)

        return chunks_with_overlap

    def _detect_headings(self, text: str) -> List[Dict[str, Any]]:
        """
        Advanced heading detection using multiple heuristics
        """
        lines = text.split('\n')
        potential_headings = []

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            heading_score = self._calculate_heading_score(line_stripped, i, lines)

            if heading_score > 0.5:  # Threshold for considering as heading
                potential_headings.append({
                    'text': line_stripped,
                    'line_index': i,
                    'confidence': heading_score,
                    'type': self._classify_heading_type(line_stripped)
                })

        return potential_headings

    def _calculate_heading_score(self, line: str, line_index: int, all_lines: List[str]) -> float:
        """
        Calculate confidence score that a line is a heading using multiple heuristics
        """
        score = 0.0

        # 1. Length heuristic: Headings are typically shorter
        if 5 <= len(line) <= 100:
            score += 0.2
        elif len(line) > 200:
            score -= 0.3

        # 2. Capitalization patterns
        words = line.split()
        if not words:
            return 0.0

        # All caps (strong indicator)
        if line.isupper() and len(words) <= 8:
            score += 0.4

        # Title case (moderate indicator)
        title_case_count = sum(1 for word in words if word[0].isupper() and len(word) > 1)
        if title_case_count >= len(words) * 0.7:  # At least 70% title case
            score += 0.3

        # 3. Numbered sections
        if re.match(r'^\d+\.?\s+', line) or re.match(r'^[A-Z]\.?\s+', line):
            score += 0.4

        # 4. Roman numerals
        if re.match(r'^[IVX]+\.?\s+', line):
            score += 0.3

        # 5. Structural indicators
        # Check if line is surrounded by empty lines (strong heading indicator)
        prev_empty = (line_index == 0 or
                     (line_index > 0 and not all_lines[line_index - 1].strip()))
        next_empty = (line_index == len(all_lines) - 1 or
                     (line_index < len(all_lines) - 1 and not all_lines[line_index + 1].strip()))

        if prev_empty and next_empty:
            score += 0.3
        elif prev_empty or next_empty:
            score += 0.1

        # 6. Check if followed by content (not another potential heading)
        if line_index < len(all_lines) - 2:
            next_line = all_lines[line_index + 1].strip()
            next_next_line = all_lines[line_index + 2].strip()

            if next_line == "" and len(next_next_line) > 50:  # Empty line followed by content
                score += 0.2

        # 7. Common heading patterns
        # Question headings
        if line.endswith('?'):
            score += 0.2

        # Colon endings (less common but possible)
        if line.endswith(':'):
            score += 0.1

        # 8. Academic paper specific patterns
        academic_keywords = [
            'introduction', 'methodology', 'results', 'discussion', 'conclusion',
            'abstract', 'background', 'related work', 'literature review',
            'experimental', 'evaluation', 'analysis', 'implementation',
            'future work', 'limitations', 'references', 'acknowledgments'
        ]

        line_lower = line.lower()
        if any(keyword in line_lower for keyword in academic_keywords):
            score += 0.3

        # 9. Avoid false positives
        # Sentences with punctuation in middle are less likely to be headings
        if ',' in line or ';' in line:
            score -= 0.2

        # Very long lines are unlikely to be headings
        if len(line) > 150:
            score -= 0.3

        # Lines starting with lowercase (except 'and', 'or', 'the', etc.)
        if line[0].islower() and not line.lower().startswith(('and ', 'or ', 'the ', 'a ', 'an ')):
            score -= 0.2

        return max(0.0, min(1.0, score))  # Clamp between 0 and 1

    def _classify_heading_type(self, line: str) -> str:
        """
        Classify the type of heading based on its format
        """
        if re.match(r'^\d+\.?\s+', line):
            return 'numbered'
        elif line.isupper():
            return 'all_caps'
        elif re.match(r'^[IVX]+\.?\s+', line):
            return 'roman_numeral'
        elif line.endswith('?'):
            return 'question'
        else:
            return 'title_case'

    def _create_sections_from_headings(self, text: str, headings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Create sections based on detected headings
        """
        if not headings:
            return []

        lines = text.split('\n')
        sections = []

        # Sort headings by line index
        headings.sort(key=lambda x: x['line_index'])

        for i, heading in enumerate(headings):
            heading_line = heading['line_index']
            section_name = heading['text']

            # Determine section content boundaries
            start_line = heading_line + 1  # Start after heading

            # Find end line (before next heading or end of document)
            if i < len(headings) - 1:
                end_line = headings[i + 1]['line_index']
            else:
                end_line = len(lines)

            # Extract section content
            section_lines = lines[start_line:end_line]
            section_content = '\n'.join(section_lines).strip()

            if section_content:  # Only add non-empty sections
                sections.append({
                    'section': section_name,
                    'content': section_content,
                    'line_start': start_line,
                    'line_end': end_line - 1,
                    'confidence': heading['confidence'],
                    'heading_type': heading['type']
                })

        return sections

    def _fallback_content_based_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Fallback method using content-based heuristics when no clear headings are found
        """
        # Try to find sections based on content patterns and paragraph structure
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        if len(paragraphs) <= 3:
            # Very short document, treat as single section
            return [{
                'section': 'Full Document',
                'content': text.strip(),
                'line_start': 0,
                'line_end': len(text.split('\n')) - 1,
                'confidence': 0.3
            }]

        # Use traditional academic patterns as last resort
        sections = []
        current_section = 'Introduction'
        current_content = []

        section_patterns = [
            (r'\b(?:ABSTRACT|Abstract)\b', 'Abstract'),
            (r'\b(?:INTRODUCTION|Introduction)\b', 'Introduction'),
            (r'\b(?:METHODOLOGY|Methods|METHODS)\b', 'Methodology'),
            (r'\b(?:RESULTS|Results)\b', 'Results'),
            (r'\b(?:DISCUSSION|Discussion)\b', 'Discussion'),
            (r'\b(?:CONCLUSION|Conclusion)\b', 'Conclusion'),
        ]

        text_lines = text.split('\n')
        for i, line in enumerate(text_lines):
            line_stripped = line.strip()

            # Check for traditional patterns
            found_section = None
            for pattern, section_name in section_patterns:
                if re.search(pattern, line_stripped):
                    found_section = section_name
                    break

            if found_section and current_content:
                section_text = '\n'.join(current_content).strip()
                if section_text:
                    sections.append({
                        'section': current_section,
                        'content': section_text,
                        'line_start': max(0, i - len(current_content)),
                        'line_end': i - 1,
                        'confidence': 0.4
                    })

                current_section = found_section
                current_content = [line]
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            section_text = '\n'.join(current_content).strip()
            if section_text:
                sections.append({
                    'section': current_section,
                    'content': section_text,
                    'line_start': len(text_lines) - len(current_content),
                    'line_end': len(text_lines) - 1,
                    'confidence': 0.4
                })

        return sections if sections else [{
            'section': 'Full Document',
            'content': text.strip(),
            'line_start': 0,
            'line_end': len(text.split('\n')) - 1,
            'confidence': 0.2
        }]

    def _sub_chunk_large_section(self, content: str, section_name: str, max_chunk_size: int, overlap: int) -> List[Dict[str, str]]:
        """
        Sub-chunk a large section while maintaining readability
        """
        chunks = []

        # Try to split by paragraphs first
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

        if not paragraphs:
            # Fallback to sentence splitting
            sentences = [s.strip() + '.' for s in content.split('.') if s.strip()]
            paragraphs = sentences

        current_chunk = ""
        chunk_number = 1

        for para in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(para) > max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'content': current_chunk.strip(),
                    'section': f"{section_name} (Part {chunk_number})",
                    'chunk_type': 'sub_section',
                    'size': len(current_chunk.strip())
                })

                # Start new chunk with overlap
                if overlap > 0:
                    # Take last 'overlap' characters from previous chunk
                    overlap_text = current_chunk[-overlap:].strip()
                    current_chunk = overlap_text + " " + para
                else:
                    current_chunk = para

                chunk_number += 1
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para

        # Add the final chunk
        if current_chunk.strip():
            chunks.append({
                'content': current_chunk.strip(),
                'section': f"{section_name} (Part {chunk_number})" if chunk_number > 1 else section_name,
                'chunk_type': 'sub_section' if chunk_number > 1 else 'section',
                'size': len(current_chunk.strip())
            })

        return chunks

    def _add_inter_section_overlap(self, chunks: List[Dict[str, str]], overlap: int) -> List[Dict[str, str]]:
        """
        Add overlap between different sections to maintain context
        """
        if overlap <= 0 or len(chunks) <= 1:
            return chunks

        enhanced_chunks = []

        for i, chunk in enumerate(chunks):
            content = chunk['content']

            # Add overlap from previous chunk
            if i > 0 and overlap > 0:
                prev_content = chunks[i-1]['content']
                if len(prev_content) >= overlap:
                    overlap_text = prev_content[-overlap:].strip()
                    content = f"[...{overlap_text}]\n\n{content}"

            # Add overlap to next chunk preview (optional context)
            if i < len(chunks) - 1 and overlap > 0:
                next_content = chunks[i+1]['content']
                if len(next_content) >= overlap:
                    next_preview = next_content[:overlap].strip()
                    content = f"{content}\n\n[{next_preview}...]"

            enhanced_chunks.append({
                **chunk,
                'content': content,
                'has_overlap': i > 0 or i < len(chunks) - 1
            })

        return enhanced_chunks

    def extract_metadata(self, text: str, category: str, filename: str) -> Dict[str, Any]:
        """
        Extract category-specific metadata from text

        Args:
            text: Extracted text content
            category: Document category
            filename: Original filename

        Returns:
            Dictionary containing metadata
        """
        base_metadata = {
            'filename': filename,
            'category': category,
            'processed_at': datetime.now().isoformat(),
            'text_length': len(text),
            'estimated_pages': text.count('--- Page') if '--- Page' in text else 1
        }

        # Category-specific metadata extraction
        if category == "Research Paper":
            base_metadata.update(self._extract_research_metadata(text))
        elif category == "Article":
            base_metadata.update(self._extract_article_metadata(text))
        elif category == "Book":
            base_metadata.update(self._extract_book_metadata(text))
        elif category == "Receipts":
            base_metadata.update(self._extract_receipt_metadata(text))
        elif category == "Terms & Conditions":
            base_metadata.update(self._extract_legal_metadata(text))
        else:  # "Other"
            base_metadata.update(self._extract_general_metadata(text))

        return base_metadata

    def _extract_research_metadata(self, text: str) -> Dict[str, Any]:
        """Extract research paper specific metadata"""
        metadata = {}

        # Look for common research paper sections
        sections = ['abstract', 'introduction', 'methodology', 'methods', 'results', 'discussion', 'conclusion', 'references']
        found_sections = []

        text_lower = text.lower()
        for section in sections:
            if section in text_lower:
                found_sections.append(section)

        metadata['detected_sections'] = found_sections
        metadata['is_academic'] = len(found_sections) >= 3

        return metadata

    def _extract_article_metadata(self, text: str) -> Dict[str, Any]:
        """Extract article specific metadata"""
        metadata = {}

        # Look for date patterns
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b',
            r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        ]

        dates_found = []
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dates_found.extend(matches)

        metadata['detected_dates'] = dates_found[:5]  # Limit to first 5 dates
        metadata['has_timeline'] = len(dates_found) > 0

        return metadata

    def _extract_book_metadata(self, text: str) -> Dict[str, Any]:
        """Extract book specific metadata"""
        metadata = {}

        # Look for chapter indicators
        chapter_patterns = [
            r'\bchapter\s+\d+\b',
            r'\bchapter\s+[ivx]+\b',
            r'\b\d+\.\s*[A-Z]'
        ]

        chapters_found = []
        for pattern in chapter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            chapters_found.extend(matches)

        metadata['detected_chapters'] = len(set(chapters_found))
        metadata['has_structure'] = len(chapters_found) > 0

        return metadata

    def _extract_receipt_metadata(self, text: str) -> Dict[str, Any]:
        """Extract receipt specific metadata"""
        metadata = {}

        # Look for monetary amounts
        money_pattern = r'\$\d+\.?\d*|\d+\.\d{2}'
        amounts = re.findall(money_pattern, text)

        # Look for common receipt keywords
        receipt_keywords = ['total', 'subtotal', 'tax', 'receipt', 'purchase', 'sale', 'item', 'qty', 'quantity']
        found_keywords = [kw for kw in receipt_keywords if kw.lower() in text.lower()]

        metadata['detected_amounts'] = len(amounts)
        metadata['receipt_indicators'] = found_keywords
        metadata['is_financial'] = len(found_keywords) >= 2

        return metadata

    def _extract_legal_metadata(self, text: str) -> Dict[str, Any]:
        """Extract legal document specific metadata"""
        metadata = {}

        # Look for legal terms
        legal_terms = ['terms', 'conditions', 'agreement', 'contract', 'liability', 'warranty', 'rights', 'obligations', 'clause']
        found_terms = [term for term in legal_terms if term.lower() in text.lower()]

        # Look for section numbers
        section_pattern = r'\b(?:section|clause|article)\s*\d+\b'
        sections = re.findall(section_pattern, text, re.IGNORECASE)

        metadata['legal_terms'] = found_terms
        metadata['detected_sections'] = len(sections)
        metadata['is_legal'] = len(found_terms) >= 3

        return metadata

    def _extract_general_metadata(self, text: str) -> Dict[str, Any]:
        """Extract general metadata for uncategorized documents"""
        metadata = {}

        # Basic text statistics
        words = text.split()
        sentences = text.count('.') + text.count('!') + text.count('?')

        metadata['word_count'] = len(words)
        metadata['sentence_count'] = sentences
        metadata['avg_sentence_length'] = len(words) / max(sentences, 1)

        return metadata

    def process_document(self, file_buffer: Union[BytesIO, bytes], category: str, filename: str) -> Dict[str, Any]:
        """
        Main processing pipeline for PDF documents

        Args:
            file_buffer: PDF file buffer
            category: Document category selected by user
            filename: Original filename

        Returns:
            Dictionary containing processed text chunks and metadata
        """
        try:
            # Extract text from PDF
            extracted_text = self.extract_text(file_buffer)

            if not extracted_text:
                raise ValueError("No text could be extracted from the document")

            # Create chunks using appropriate method based on category
            if category == "Research Paper":
                chunk_data = self.heading_based_chunk(extracted_text, category, max_chunk_size=2000, overlap=50)
                chunks = [chunk['content'] for chunk in chunk_data]
                chunk_metadata = chunk_data
                chunk_method = 'heading_based'
            else:
                chunks = self.simple_chunk(extracted_text, chunk_size=500, overlap=50)
                chunk_metadata = [{"content": chunk, "section": "content", "chunk_type": "simple"} for chunk in chunks]
                chunk_method = 'simple_overlap'

            if not chunks:
                raise ValueError("No text chunks could be created")

            # Extract document metadata
            document_metadata = self.extract_metadata(extracted_text, category, filename)

            # Add chunking-specific metadata for research papers
            if category == "Research Paper":
                sections_found = list(set([chunk.get('section', 'Unknown') for chunk in chunk_metadata]))
                document_metadata['sections_found'] = sections_found
                document_metadata['sections_count'] = len(sections_found)
                document_metadata['has_structured_sections'] = len(sections_found) > 1

            # Create result structure
            result = {
                'success': True,
                'filename': filename,
                'category': category,
                'extracted_text': extracted_text,
                'chunks': chunks,
                'chunk_metadata': chunk_metadata,
                'chunk_count': len(chunks),
                'metadata': document_metadata,
                'processing_info': {
                    'processor_version': '2.0',
                    'chunk_method': chunk_method,
                    'chunk_size': 2000 if category == "Research Paper" else 500,
                    'overlap': 50,
                    'specialized_processing': category == "Research Paper"
                }
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'filename': filename,
                'category': category
            }