"""
Utility functions for the RAG-Enhanced Chat application.

This module provides helper functions including document category auto-detection.
"""

from typing import List, Tuple
import re


def auto_detect_category(text_sample: str, max_chars: int = 2000) -> str:
    """
    Auto-detect document category using rule-based heuristics.

    This function analyzes a text sample and returns the most likely document category
    based on keyword patterns and content structure. Fast, free, and accurate for most cases.

    Args:
        text_sample: Text sample from document (typically first 2000 characters)
        max_chars: Maximum characters to analyze (default: 2000)

    Returns:
        Detected category string: "Research Paper", "Article", "Book", "Receipts",
        "Terms & Conditions", or "Other"

    Examples:
        >>> auto_detect_category("ABSTRACT\\n\\nThis paper presents...")
        'Research Paper'
        >>> auto_detect_category("TOTAL: $45.99\\nSUBTOTAL: $42.00")
        'Receipts'
    """
    if not text_sample or not text_sample.strip():
        return "Other"

    # Truncate to max_chars for performance
    if len(text_sample) > max_chars:
        text_sample = text_sample[:max_chars]

    text_lower = text_sample.lower()

    # Category detection patterns (order matters - most specific first)

    # 1. Research Paper - Strong academic indicators
    research_keywords = [
        'abstract', 'methodology', 'literature review', 'references',
        'introduction', 'conclusion', 'discussion', 'results',
        'hypothesis', 'experimental', 'study', 'research',
        'findings', 'analysis', 'methods', 'citation'
    ]
    research_score = sum(1 for kw in research_keywords if kw in text_lower)

    # Strong research paper indicators (high confidence)
    if research_score >= 4:
        return "Research Paper"
    if 'abstract' in text_lower and any(kw in text_lower for kw in ['methodology', 'results', 'conclusion']):
        return "Research Paper"

    # 2. Receipts - Financial transaction indicators
    receipt_keywords = [
        'total', 'subtotal', 'tax', 'receipt', 'purchase',
        'qty', 'quantity', 'item', 'sale', 'payment',
        'credit', 'debit', 'invoice', 'transaction'
    ]
    receipt_score = sum(1 for kw in receipt_keywords if kw in text_lower)

    # Check for monetary amounts (strong signal)
    money_pattern = r'\$\s*\d+\.?\d*|\d+\.\d{2}\s*(?:usd|eur|gbp)?'
    money_matches = len(re.findall(money_pattern, text_lower))

    if receipt_score >= 3 and money_matches >= 2:
        return "Receipts"
    if money_matches >= 5:  # Many price tags = likely receipt
        return "Receipts"

    # 3. Terms & Conditions - Legal document indicators
    legal_keywords = [
        'terms', 'conditions', 'agreement', 'contract',
        'liability', 'warranty', 'rights', 'obligations',
        'clause', 'party', 'herein', 'thereof', 'hereby',
        'legal', 'binding', 'jurisdiction', 'dispute'
    ]
    legal_score = sum(1 for kw in legal_keywords if kw in text_lower)

    # Strong legal indicators
    if legal_score >= 5:
        return "Terms & Conditions"
    if 'terms' in text_lower and 'conditions' in text_lower and legal_score >= 3:
        return "Terms & Conditions"

    # 4. Book - Chapter-based structure indicators
    book_keywords = [
        'chapter', 'preface', 'foreword', 'epilogue',
        'prologue', 'appendix', 'part i', 'part ii',
        'table of contents', 'acknowledgments'
    ]
    book_score = sum(1 for kw in book_keywords if kw in text_lower)

    # Check for chapter numbering patterns
    chapter_pattern = r'chapter\s+\d+|chapter\s+[ivxlcdm]+'
    chapter_matches = len(re.findall(chapter_pattern, text_lower))

    if book_score >= 2 or chapter_matches >= 2:
        return "Book"

    # 5. Article - News/blog article indicators
    article_keywords = [
        'author:', 'published', 'by:', 'reporter',
        'editor', 'news', 'article', 'story', 'posted',
        'update:', 'breaking', 'source:', 'correspondent'
    ]
    article_score = sum(1 for kw in article_keywords if kw in text_lower)

    # Date patterns common in articles
    date_pattern = r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b'
    date_matches = len(re.findall(date_pattern, text_lower))

    if article_score >= 2 and date_matches >= 1:
        return "Article"

    # 6. Fallback logic - use research paper if medium confidence
    if research_score >= 2:
        return "Research Paper"

    # Default to "Other" if no clear category detected
    return "Other"


def get_category_confidence(text_sample: str, detected_category: str, max_chars: int = 2000) -> Tuple[str, float]:
    """
    Get confidence score for auto-detected category.

    Args:
        text_sample: Text sample from document
        detected_category: Category detected by auto_detect_category()
        max_chars: Maximum characters to analyze

    Returns:
        Tuple of (category, confidence_score) where confidence is 0.0-1.0
    """
    if not text_sample or not text_sample.strip():
        return detected_category, 0.3

    if len(text_sample) > max_chars:
        text_sample = text_sample[:max_chars]

    text_lower = text_sample.lower()

    # Calculate confidence based on keyword density and pattern matches
    confidence = 0.5  # Base confidence

    if detected_category == "Research Paper":
        keywords = ['abstract', 'methodology', 'results', 'conclusion', 'references']
        matches = sum(1 for kw in keywords if kw in text_lower)
        confidence = min(0.95, 0.5 + (matches * 0.15))

    elif detected_category == "Receipts":
        money_pattern = r'\$\s*\d+\.?\d*'
        money_count = len(re.findall(money_pattern, text_lower))
        confidence = min(0.95, 0.5 + (money_count * 0.08))

    elif detected_category == "Terms & Conditions":
        legal_keywords = ['terms', 'conditions', 'agreement', 'liability', 'warranty']
        matches = sum(1 for kw in legal_keywords if kw in text_lower)
        confidence = min(0.95, 0.5 + (matches * 0.12))

    elif detected_category == "Book":
        chapter_pattern = r'chapter\s+\d+'
        chapter_count = len(re.findall(chapter_pattern, text_lower))
        confidence = min(0.95, 0.5 + (chapter_count * 0.2))

    elif detected_category == "Article":
        confidence = 0.6  # Medium confidence for articles

    else:  # "Other"
        confidence = 0.4  # Low confidence for fallback category

    return detected_category, confidence


# Document category constants (shared across the application)
DOCUMENT_CATEGORIES = [
    "Research Paper",
    "Article",
    "Book",
    "Other",
    "Receipts",
    "Terms & Conditions"
]
