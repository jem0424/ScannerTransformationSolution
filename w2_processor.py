#!/usr/bin/env python3
"""
W-2 Document Processor
Extracts employee names from Box 1 and tax year to rename files automatically.
Designed for batch processing of scanned W-2 documents.
"""

import os
import re
import shutil
from pathlib import Path
import logging
from datetime import datetime
import pytesseract
from PIL import Image
import pdf2image
import argparse
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('w2_processing.log'),
        logging.StreamHandler()
    ]
)

class W2Processor:
    def __init__(self, input_dir, output_dir, tesseract_path=None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.processed_count = 0
        self.failed_count = 0
        self.failed_files = []
        
        # Set Tesseract path if provided (Windows users may need this)
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Supported file extensions
        self.supported_extensions = {'.pdf', '.png', '.jpg', '.jpeg', '.tiff', '.bmp'}
    
    def detect_and_correct_orientation(self, image):
        """Detect document orientation and rotate if needed"""
        try:
            # Get orientation info from Tesseract
            osd = pytesseract.image_to_osd(image, config='--psm 0')
            
            # Extract rotation angle
            rotation_match = re.search(r'Rotate: (\d+)', osd)
            if rotation_match:
                rotation = int(rotation_match.group(1))
                
                # Rotate image if needed
                if rotation != 0:
                    logging.info(f"Detected rotation: {rotation} degrees, correcting...")
                    image = image.rotate(-rotation, expand=True)
                    
            return image
        except Exception as e:
            logging.warning(f"Could not detect orientation, proceeding with original: {e}")
            return image
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF using OCR with orientation correction"""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=600)
            
            # Use OCR on first page (W-2s are typically one page)
            if images:
                image = images[0]
                
                # Correct orientation
                corrected_image = self.detect_and_correct_orientation(image)
                
                # Extract text with better OCR settings for documents
                #                 text = pytesseract.image_to_string(
                #     corrected_image,
                #     config='--psm 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-() -c preserve_interword_spaces=1'
                # )
                text = pytesseract.image_to_string(image)
                print(f"RAW_TEXT: {text}")
                return text
            return ""
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""
    
    def extract_text_from_image(self, image_path):
        """Extract text from image using OCR with orientation correction"""
        try:
            image = Image.open(image_path)
            
            # Correct orientation
            corrected_image = self.detect_and_correct_orientation(image)
            
            # Extract text with better OCR settings for documents
            text = pytesseract.image_to_string(
                corrected_image, 
                config='--psm 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,-()'
            )
            return text
        except Exception as e:
            logging.error(f"Error extracting text from image {image_path}: {e}")
            return ""
    
    def extract_employee_name(self, text):
        """Extract employee name from W-2 text - specifically looking at W-2 structure"""
        lines = text.split('\n')
        print(f"DEBUG: Text has {len(lines)} lines")
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line.strip()}'")
        
        # W-2 specific extraction - look for patterns around SSN and employee info
        
        # Method 1: Look for lines with SSN pattern and extract nearby name
        ssn_pattern = r'\b\d{9}\b|\b\d{3}-?\d{2}-?\d{4}\b'
        
        for i, line in enumerate(lines):
            # If line contains SSN, look around it for employee name
            if re.search(ssn_pattern, line):
                print(f"DEBUG: Found SSN pattern in line {i}: '{line.strip()}'")
                
                # Look in surrounding lines for employee name
                search_lines = lines[max(0, i-2):min(len(lines), i+3)]
                
                for search_line in search_lines:
                    # Look for name patterns in these lines
                    name_matches = self.find_name_in_line(search_line)
                    if name_matches:
                        print(f"DEBUG: Found name '{name_matches}' near SSN")
                        return name_matches
        
        # Method 2: Look for specific W-2 structure patterns
        for i, line in enumerate(lines):
            # Look for lines that might contain employee information
            # Often after employer info but before wage amounts
            if i > 0 and i < len(lines) - 1:
                name_matches = self.find_name_in_line(line)
                if name_matches and not self.is_employer_line(line):
                    print(f"DEBUG: Found potential employee name in line {i}: '{name_matches}'")
                    return name_matches
        
        # Method 3: Look for capitalized names not in employer/address context
        all_potential_names = []
        for i, line in enumerate(lines):
            names = self.find_name_in_line(line)
            if names:
                # Score the name based on position and context
                score = self.score_name_likelihood(line, i, len(lines))
                all_potential_names.append((names, score, i))
                print(f"DEBUG: Potential name '{names}' in line {i} with score {score}")
        
        # Return highest scoring name
        if all_potential_names:
            all_potential_names.sort(key=lambda x: x[1], reverse=True)
            best_name = all_potential_names[0][0]
            print(f"DEBUG: Selected best name: '{best_name}'")
            return best_name
        
        return None
    
    def find_name_in_line(self, line):
        """Find potential names in a single line"""
        # Clean the line
        cleaned_line = re.sub(r'[^\w\s]', ' ', line)
        
        # Look for name patterns
        patterns = [
            r'\b([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,15})\b',  # Title case
            r'\b([A-Z]{2,15}\s+[A-Z]{2,15})\b',  # All caps
            r'\b([A-Z][a-zA-Z\']{1,15}\s+[A-Z][a-zA-Z\']{1,15})\b',  # Mixed with apostrophes
        ]
        
        excluded_words = {
            'EMPLOYEE', 'WAGES', 'TIPS', 'FEDERAL', 'STATE', 'SOCIAL', 'SECURITY',
            'INCOME', 'MEDICARE', 'WITHHOLDING', 'COMPENSATION', 'BENEFITS',
            'CONTROL', 'NUMBER', 'EMPLOYER', 'ADDRESS', 'CITY', 'ZIP',
            'FORM', 'TAX', 'STATEMENT', 'YEAR', 'THREAD', 'THREADING', 'SPA',
            'WASHINGTON', 'STREET', 'BRAINTREE', 'CAMBRIDGE', 'BOSTON'
        }
        
        for pattern in patterns:
            matches = re.findall(pattern, cleaned_line)
            for match in matches:
                name_parts = match.split()
                if (len(name_parts) == 2 and 
                    len(name_parts[0]) >= 2 and len(name_parts[1]) >= 2 and
                    not any(word.upper() in excluded_words for word in name_parts) and
                    not re.search(r'\d', match)):  # No digits in names
                    return match
        
        return None
    
    def is_employer_line(self, line):
        """Check if line likely contains employer information"""
        employer_indicators = ['LLC', 'INC', 'CORP', 'COMPANY', 'THREADING', 'SPA', 'STREET', 'AVENUE']
        return any(indicator in line.upper() for indicator in employer_indicators)
    
    def score_name_likelihood(self, line, line_index, total_lines):
        """Score how likely a line contains the employee name"""
        score = 0
        
        # Employee names typically appear in specific positions on W-2
        relative_position = line_index / total_lines
        
        # Prefer names that appear in the middle section (employee info area)
        if 0.2 < relative_position < 0.6:
            score += 10
        
        # Avoid lines with employer indicators
        if self.is_employer_line(line):
            score -= 20
        
        # Prefer lines with fewer numbers (addresses have zip codes, etc.)
        number_count = len(re.findall(r'\d', line))
        score -= number_count * 2
        
        # Prefer shorter lines (names typically on their own line)
        if len(line.strip()) < 50:
            score += 5
        
        return score
    
    def extract_tax_year(self, text):
        """Extract tax year from W-2 text"""
        # Look for 4-digit years in expected range
        year_patterns = [
            r'\b(20[1-3]\d)\b',  # 2010-2039
            r'(?:Tax\s+Year|Year|Form\s+W-?2)\s*:?\s*(20[1-3]\d)',  # After "Tax Year" etc.
            r'(20[1-3]\d)(?=\s*(?:Tax|W-?2|Form))',  # Before "Tax", "W2", "Form"
        ]
        
        all_years = []
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                year = match if isinstance(match, str) else match[0] if match else None
                if year and 2015 <= int(year) <= 2030:  # Reasonable range
                    all_years.append(year)
                    print(f"DEBUG: Found year: '{year}'")
        
        if all_years:
            # Return most common year, or first if tie
            from collections import Counter
            year_counts = Counter(all_years)
            return year_counts.most_common(1)[0][0]
        
        return None
    
    def clean_filename(self, name):
        """Clean name for use in filename"""
        # Replace spaces with underscores, remove special characters
        name = re.sub(r'[^\w\s-]', '', name)
        name = re.sub(r'[-\s]+', '_', name)
        return name.strip('_')
    
    def process_document(self, file_path):
        """Process a single document"""
        try:
            logging.info(f"Processing: {file_path.name}")
            
            # Extract text based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self.extract_text_from_pdf(file_path)
            else:
                text = self.extract_text_from_image(file_path)
            
            # DEBUG: Print extracted text
            print(f"\n=== DEBUG: Extracted text from {file_path.name} ===")
            print(text)
            print(f"=== END DEBUG for {file_path.name} ===\n")
            
            if not text.strip():
                logging.warning(f"No text extracted from {file_path.name}")
                return False
            
            # Extract employee name and tax year
            employee_name = self.extract_employee_name(text)
            tax_year = self.extract_tax_year(text)
            
            # DEBUG: Show what was extracted
            print(f"DEBUG: Extracted name: '{employee_name}'")
            print(f"DEBUG: Extracted year: '{tax_year}'")
            
            if not employee_name or not tax_year:
                logging.warning(f"Could not extract name ({employee_name}) or year ({tax_year}) from {file_path.name}")
                return False
            
            # Create new filename
            clean_name = self.clean_filename(employee_name)
            new_filename = f"{clean_name}_W2_{tax_year}{file_path.suffix}"
            new_path = self.output_dir / new_filename
            
            # Handle duplicate filenames
            counter = 1
            while new_path.exists():
                new_filename = f"{clean_name}_W2_{tax_year}_{counter}{file_path.suffix}"
                new_path = self.output_dir / new_filename
                counter += 1
            
            # Copy file to new location with new name
            shutil.copy2(file_path, new_path)
            
            logging.info(f"Renamed: {file_path.name} -> {new_filename}")
            self.processed_count += 1
            return True
            
        except Exception as e:
            logging.error(f"Error processing {file_path.name}: {e}")
            return False
    
    def process_batch(self):
        """Process all documents in the input directory"""
        logging.info(f"Starting batch processing from {self.input_dir}")
        logging.info(f"Output directory: {self.output_dir}")
        
        # Get all supported files
        files_to_process = []
        for ext in self.supported_extensions:
            files_to_process.extend(self.input_dir.glob(f"*{ext}"))
            files_to_process.extend(self.input_dir.glob(f"*{ext.upper()}"))
        
        total_files = len(files_to_process)
        logging.info(f"Found {total_files} files to process")
        
        if total_files == 0:
            logging.warning("No supported files found in input directory")
            return
        
        # Process each file
        for i, file_path in enumerate(files_to_process, 1):
            print(f"Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
            
            if not self.process_document(file_path):
                self.failed_count += 1
                self.failed_files.append(file_path.name)
        
        # Summary
        logging.info(f"Processing complete!")
        logging.info(f"Successfully processed: {self.processed_count}")
        logging.info(f"Failed: {self.failed_count}")
        
        if self.failed_files:
            logging.info("Failed files:")
            for failed_file in self.failed_files:
                logging.info(f"  - {failed_file}")

def main():
    parser = argparse.ArgumentParser(description="Process W-2 documents and rename based on employee name and tax year")
    parser.add_argument("input_dir", help="Directory containing scanned W-2 documents")
    parser.add_argument("output_dir", help="Directory to save renamed documents")
    parser.add_argument("--tesseract-path", help="Path to Tesseract executable (if needed)")
    
    args = parser.parse_args()
    
    # Validate directories
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return
    
    # Create processor and run
    processor = W2Processor(args.input_dir, args.output_dir, args.tesseract_path)
    processor.process_batch()

if __name__ == "__main__":
    main()