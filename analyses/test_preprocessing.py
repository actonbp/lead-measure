#!/usr/bin/env python3
"""
Test preprocessing functions on sample leadership items

This script demonstrates how the stem removal and gender-neutral language
conversion work on sample leadership items.
"""

import re

# Stem patterns to remove (common leader references)
STEM_PATTERNS = [
    r'^my (leader|manager|supervisor|boss|superior)',
    r'^the (leader|manager|supervisor|boss|superior)',
    r'^our (leader|manager|supervisor|boss|superior)',
    r'^this (leader|manager|supervisor|boss|superior)',
    r'^he/she',
    r'^s?he',
    r'^they',
    r'^my department (leader|manager|supervisor|boss|superior)',
    r'^the department (leader|manager|supervisor|boss|superior)',
]

# Gendered terms to replace
GENDERED_TERMS = {
    'he': 'they',
    'she': 'they',
    'his': 'their',
    'her': 'their',
    'himself': 'themselves',
    'herself': 'themselves',
    'him': 'them',
    'chairman': 'chairperson',
    'foreman': 'supervisor',
    'policeman': 'police officer',
    'fireman': 'firefighter',
    'businessman': 'business person',
    'salesman': 'salesperson',
    'mailman': 'mail carrier',
    'stewardess': 'flight attendant',
    'waitress': 'server',
    'mankind': 'humanity',
    'man-made': 'artificial',
    'manpower': 'workforce',
    'man': 'person',
    'men': 'people',
    'woman': 'person',
    'women': 'people'
}

def remove_stems(text):
    """Remove common leadership item stems from text."""
    lower_text = text.lower()
    
    # Check and remove each stem pattern
    for pattern in STEM_PATTERNS:
        match = re.match(pattern, lower_text, re.IGNORECASE)
        if match:
            # Remove the stem and clean up
            text = text[match.end():].strip()
            # Remove leading punctuation if present
            text = re.sub(r'^[,.: ]+', '', text)
            # Capitalize first letter
            if text:
                text = text[0].upper() + text[1:]
            break
    
    return text

def remove_gendered_language(text):
    """Replace gendered terms with gender-neutral alternatives."""
    # Convert to lowercase for matching
    words = text.split()
    result_words = []
    
    for word in words:
        # Extract any punctuation
        prefix_punct = ''
        suffix_punct = ''
        
        match = re.match(r'^([^\w]*)(.+?)([^\w]*)$', word)
        if match:
            prefix_punct, word_core, suffix_punct = match.groups()
        else:
            word_core = word
        
        # Check if the word (case-insensitive) matches any gendered term
        word_lower = word_core.lower()
        if word_lower in GENDERED_TERMS:
            # Replace with gender-neutral term, preserve case
            if word_core.isupper():
                replacement = GENDERED_TERMS[word_lower].upper()
            elif word_core[0].isupper():
                replacement = GENDERED_TERMS[word_lower].capitalize()
            else:
                replacement = GENDERED_TERMS[word_lower]
            
            # Reassemble with punctuation
            result_words.append(f"{prefix_punct}{replacement}{suffix_punct}")
        else:
            # Keep original word
            result_words.append(word)
    
    return ' '.join(result_words)

# Sample leadership items with different stem patterns
sample_items = [
    "My leader encourages others to have high expectations at work",
    "The manager provides clear instructions about my job",
    "My supervisor gives me the authority I need to make decisions",
    "He/she inspires me to do more than I thought I could do",
    "My department leader clearly defines my role and responsibilities",
    "She regularly supports her team members in their work",
    "He makes sure that the interests of his subordinates are considered",
    "The foreman reviews all work completed by his team",
    "My boss gives me special recognition when my work is very good",
    "Our leader demonstrates high integrity and ethical behavior",
]

print("Testing preprocessing on sample leadership items:\n")

print("-" * 80)
print("ORIGINAL ITEM → STEM REMOVED → GENDER-NEUTRAL LANGUAGE\n")

for item in sample_items:
    # Apply preprocessing steps
    stem_removed = remove_stems(item)
    gender_neutral = remove_gendered_language(stem_removed)
    
    # Print the transformation
    print(f"ORIGINAL: {item}")
    print(f"NO STEM:  {stem_removed}")
    print(f"NEUTRAL:  {gender_neutral}")
    print()

print("-" * 80)
print("Note: This is just a test on sample items. Run preprocess_leadership_data.py")
print("to process the entire dataset and save the results.") 