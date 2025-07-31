#!/usr/bin/env python3
"""
W-2 Document Processor
Extracts employee names from Box 1 and tax year to rename files automatically.
Designed for batch processing of scanned W-2 documents.
Supports multi-page PDFs with multiple W-2 documents.
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
        """Extract text from PDF using OCR with orientation correction - handles multiple pages"""
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path, dpi=600)

            extracted_texts = []

            # Process each page/image
            for page_num, image in enumerate(images):
                print(f"DEBUG: Processing page {page_num + 1} of {len(images)}")

                # Correct orientation
                corrected_image = self.detect_and_correct_orientation(image)

                # Extract text
                text = pytesseract.image_to_string(corrected_image)

                if text.strip():  # Only add non-empty text
                    extracted_texts.append(text)

            return extracted_texts if extracted_texts else [""]
        except Exception as e:
            logging.error(f"Error extracting text from PDF {pdf_path}: {e}")
            return [""]

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
            return [text] if text.strip() else [""]  # Return as list for consistency
        except Exception as e:
            logging.error(f"Error extracting text from image {image_path}: {e}")
            return [""]

    def extract_employee_name(self, text):
        """Extract employee name from W-2 text - specifically looking at W-2 structure"""
        lines = text.split('\n')
        print(f"DEBUG: Text has {len(lines)} lines")
        for i, line in enumerate(lines):
            print(f"Line {i}: '{line.strip()}'")

        # Method 1: Look for the specific W-2 structure
        # Find "Employee name, address, and ZIP code" and get the name from the next line(s)
        for i, line in enumerate(lines):
            line_upper = line.upper().strip()
            if any(phrase in line_upper for phrase in [
                "EMPLOYEE NAME, ADDRESS, AND ZIP CODE",
                "EMPLOYEE NAME ADDRESS AND ZIP CODE",
                "EMPLOYEE NAME",
                "EMPLOYEE'S NAME"
            ]):
                print(f"DEBUG: Found employee section marker at line {i}: '{line.strip()}'")

                # Look in the next few lines for the actual name
                for j in range(i + 1, min(i + 5, len(lines))):
                    if j < len(lines):
                        print(f"DEBUG: Checking line {j} after marker: '{lines[j].strip()}'")
                        potential_name = self.find_name_in_line(lines[j])
                        if potential_name and not self.is_address_or_employer_line(lines[j]):
                            print(f"DEBUG: Found employee name after marker: '{potential_name}'")
                            return potential_name

                        # Also check if MAYNARD ST or similar appears - the name might be just before it
                        if self.looks_like_address(lines[j]):
                            print(f"DEBUG: Found address at line {j}, checking previous lines for name")
                            # Check a few lines back from the address
                            for k in range(max(i + 1, j - 2), j):
                                if k < len(lines):
                                    prev_name = self.find_name_in_line(lines[k])
                                    if prev_name and not self.is_address_or_employer_line(lines[k]):
                                        print(f"DEBUG: Found name before address: '{prev_name}'")
                                        return prev_name

        # Method 2: Look for names that appear before address patterns
        # Addresses typically contain numbers, street indicators, or ZIP codes
        for i, line in enumerate(lines):
            potential_name = self.find_name_in_line(line)
            if potential_name and not self.is_government_agency(potential_name):
                # Check if the next line looks like an address
                if i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if self.looks_like_address(next_line):
                        print(f"DEBUG: Found name before address pattern: '{potential_name}'")
                        return potential_name

        # Method 3: Look for SSN pattern and find name in nearby lines (but avoid employer area)
        ssn_pattern = r'\b\d{9}\b|\b\d{3}-?\d{2}-?\d{4}\b'

        for i, line in enumerate(lines):
            if re.search(ssn_pattern, line):
                print(f"DEBUG: Found SSN pattern in line {i}: '{line.strip()}'")

                # Look in surrounding lines, but avoid lines too early (employer section)
                # Employee info typically comes after line 5-10
                search_start = max(0, i - 3)
                search_end = min(len(lines), i + 3)

                for j in range(search_start, search_end):
                    if j >= 5:  # Avoid employer section which is typically at the top
                        search_line = lines[j]
                        name_match = self.find_name_in_line(search_line)
                        if (name_match and
                                not self.is_address_or_employer_line(search_line) and
                                not self.is_government_agency(name_match)):
                            print(f"DEBUG: Found name near SSN (avoiding employer section): '{name_match}'")
                            return name_match

        # Method 4: Score all potential names and pick best one (improved scoring)
        all_potential_names = []
        for i, line in enumerate(lines):
            names = self.find_name_in_line(line)
            if names and not self.is_government_agency(names):
                score = self.score_employee_name_likelihood(line, i, len(lines))
                all_potential_names.append((names, score, i))
                print(f"DEBUG: Potential name '{names}' in line {i} with score {score}")

        # Return highest scoring name that's not a government agency
        if all_potential_names:
            all_potential_names.sort(key=lambda x: x[1], reverse=True)
            best_name = all_potential_names[0][0]
            print(f"DEBUG: Selected best name: '{best_name}'")
            return best_name

        return None

    def find_name_in_line(self, line):
        """Find potential names in a single line"""
        print(f"DEBUG find_name_in_line: Processing line: '{line}'")

        # Clean the line more aggressively
        cleaned_line = re.sub(r'[^\w\s]', ' ', line)
        cleaned_line = re.sub(r'\s+', ' ', cleaned_line.strip())
        print(f"DEBUG find_name_in_line: Cleaned line: '{cleaned_line}'")

        # Look for name patterns - including 3-part names and better handling
        patterns = [
            # Three-part names (First Middle Last) - all caps
            r'\b([A-Z]{2,15}\s+[A-Z]{1,15}\s+[A-Z]{2,15})\b',
            # Two-part names - all caps
            r'\b([A-Z]{2,15}\s+[A-Z]{2,15})\b',
            # Three-part names - title case
            r'\b([A-Z][a-z]{1,15}\s+[A-Z][a-z]{0,15}\s+[A-Z][a-z]{1,15})\b',
            # Two-part names - title case
            r'\b([A-Z][a-z]{1,15}\s+[A-Z][a-z]{1,15})\b',
            # Mixed case with apostrophes
            r'\b([A-Z][a-zA-Z\']{1,15}\s+[A-Z][a-zA-Z\']{1,15})\b',
        ]

        excluded_words = {
            'EMPLOYEE', 'WAGES', 'TIPS', 'FEDERAL', 'STATE', 'SOCIAL', 'SECURITY',
            'INCOME', 'MEDICARE', 'WITHHOLDING', 'COMPENSATION', 'BENEFITS',
            'CONTROL', 'NUMBER', 'EMPLOYER', 'ADDRESS', 'CITY', 'ZIP',
            'FORM', 'TAX', 'STATEMENT', 'YEAR', 'THREAD', 'THREADING', 'SPA',
            'WASHINGTON', 'STREET', 'BRAINTREE', 'CAMBRIDGE', 'BOSTON', 'MAYNARD'
        }

        for pattern in patterns:
            matches = re.findall(pattern, cleaned_line)
            print(f"DEBUG find_name_in_line: Pattern '{pattern}' found matches: {matches}")

            for match in matches:
                name_parts = match.split()
                print(f"DEBUG find_name_in_line: Checking name parts: {name_parts}")

                # Check if it's a valid name
                if (len(name_parts) >= 2 and len(name_parts) <= 3 and
                        all(len(part) >= 1 for part in name_parts) and
                        not any(word.upper() in excluded_words for word in name_parts) and
                        not re.search(r'\d', match) and  # No digits in names
                        not any(word.upper() in ['ST', 'AVE', 'RD', 'DR', 'LN', 'CT'] for word in
                                name_parts)):  # Not street abbreviations

                    print(f"DEBUG find_name_in_line: Found valid name: '{match}'")
                    return match
                else:
                    print(f"DEBUG find_name_in_line: Rejected '{match}' - failed validation")

        print(f"DEBUG find_name_in_line: No valid name found in line")
        return None

    def is_government_agency(self, name):
        """Check if the name is likely a government agency or department"""
        government_terms = {
            'TREASURY', 'INTERNAL', 'REVENUE', 'SERVICE', 'DEPARTMENT',
            'FEDERAL', 'GOVERNMENT', 'ADMINISTRATION', 'AGENCY', 'BUREAU',
            'SOCIAL', 'SECURITY', 'IRS', 'SSA', 'TREASURY INTERNAL'
        }

        name_upper = name.upper()
        return any(term in name_upper for term in government_terms)

    def looks_like_address(self, line):
        """Check if a line looks like an address"""
        address_indicators = [
            r'\d+\s+[A-Z][a-z]+\s+(ST|STREET|AVE|AVENUE|RD|ROAD|BLVD|BOULEVARD|DR|DRIVE|LN|LANE|CT|COURT)',
            r'\b\d{5}(-\d{4})?\b',  # ZIP code
            r'\d+\s+[A-Z]',  # Number followed by letter (like "24 MAYNARD")
            r'(SUITE|APT|APARTMENT|UNIT)\s*\d+',
        ]

        return any(re.search(pattern, line, re.IGNORECASE) for pattern in address_indicators)

    def is_address_or_employer_line(self, line):
        """Enhanced check for address or employer information"""
        # Original employer indicators
        employer_indicators = ['LLC', 'INC', 'CORP', 'COMPANY', 'THREADING', 'SPA']

        # Address indicators
        address_indicators = ['STREET', 'AVENUE', 'ROAD', 'DRIVE', 'LANE', 'COURT', 'BLVD', 'ST', 'AVE', 'RD', 'DR']

        # Government/tax indicators
        government_indicators = ['TREASURY', 'INTERNAL', 'REVENUE', 'SERVICE', 'FEDERAL', 'IRS', 'SSA']

        line_upper = line.upper()

        return (any(indicator in line_upper for indicator in employer_indicators) or
                any(indicator in line_upper for indicator in address_indicators) or
                any(indicator in line_upper for indicator in government_indicators) or
                self.looks_like_address(line))

    def score_employee_name_likelihood(self, line, line_index, total_lines):
        """Improved scoring for employee name likelihood"""
        score = 0

        # Employee names typically appear after the employer section
        # but before the wage amounts (usually in the first third to half)
        relative_position = line_index / total_lines

        # Prefer names that appear in the employee info section
        if 0.15 < relative_position < 0.6:
            score += 15
        elif 0.6 <= relative_position < 0.8:
            score += 5  # Could still be employee info
        else:
            score -= 10  # Too early (employer) or too late (tax amounts)

        # Strong penalty for government agencies
        if self.is_government_agency(line):
            score -= 50

        # Penalty for employer/address lines
        if self.is_address_or_employer_line(line):
            score -= 30

        # Penalty for lines with many numbers (addresses, amounts)
        number_count = len(re.findall(r'\d', line))
        score -= number_count * 3

        # Bonus for lines that are likely just names (shorter, cleaner)
        if 10 <= len(line.strip()) <= 40:  # Reasonable name length
            score += 10

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

    def extract_single_page_pdf(self, pdf_path, page_index, output_path):
        """Extract a single page from a PDF and save it as a separate PDF"""
        try:
            import PyPDF2

            with open(pdf_path, 'rb') as input_file:
                pdf_reader = PyPDF2.PdfReader(input_file)
                pdf_writer = PyPDF2.PdfWriter()

                # Add the specific page
                if page_index < len(pdf_reader.pages):
                    pdf_writer.add_page(pdf_reader.pages[page_index])

                    # Write to output file
                    with open(output_path, 'wb') as output_file:
                        pdf_writer.write(output_file)

                    logging.info(f"Extracted page {page_index + 1} to {output_path}")
                else:
                    logging.error(f"Page {page_index + 1} not found in {pdf_path}")

        except ImportError:
            logging.warning("PyPDF2 not available, using pdf2image method instead")
            self.extract_single_page_pdf_alternative(pdf_path, page_index, output_path)
        except Exception as e:
            logging.error(f"Error extracting page {page_index + 1} from {pdf_path}: {e}")
            # Fallback to copying original file
            shutil.copy2(pdf_path, output_path)

    def extract_single_page_pdf_alternative(self, pdf_path, page_index, output_path):
        """Alternative method to extract single page using pdf2image"""
        try:
            # Convert specific page to image then back to PDF
            images = pdf2image.convert_from_path(pdf_path, dpi=300, first_page=page_index + 1, last_page=page_index + 1)

            if images:
                # Save as PDF
                images[0].save(output_path, "PDF", resolution=300.0)
                logging.info(f"Extracted page {page_index + 1} to {output_path} using image conversion")
            else:
                logging.error(f"Could not extract page {page_index + 1} from {pdf_path}")
                shutil.copy2(pdf_path, output_path)

        except Exception as e:
            logging.error(f"Error in alternative page extraction: {e}")
            shutil.copy2(pdf_path, output_path)

    def process_document(self, file_path):
        """Process a single document - now handles multiple W-2s in one file"""
        try:
            logging.info(f"Processing: {file_path.name}")

            # Extract text based on file type - now returns list of texts
            if file_path.suffix.lower() == '.pdf':
                texts = self.extract_text_from_pdf(file_path)
            else:
                texts = self.extract_text_from_image(file_path)

            if not texts or all(not text.strip() for text in texts):
                logging.warning(f"No text extracted from {file_path.name}")
                return False

            successful_extractions = 0

            # Process each document/page separately
            for doc_index, text in enumerate(texts):
                if not text.strip():
                    continue

                doc_suffix = f"_doc{doc_index + 1}" if len(texts) > 1 else ""

                # DEBUG: Print extracted text
                print(f"\n=== DEBUG: Extracted text from {file_path.name}{doc_suffix} ===")
                print(text)
                print(f"=== END DEBUG for {file_path.name}{doc_suffix} ===\n")

                # Extract employee name and tax year
                employee_name = self.extract_employee_name(text)
                tax_year = self.extract_tax_year(text)

                # DEBUG: Show what was extracted
                print(f"DEBUG: Document {doc_index + 1} - Extracted name: '{employee_name}'")
                print(f"DEBUG: Document {doc_index + 1} - Extracted year: '{tax_year}'")

                if not employee_name or not tax_year:
                    logging.warning(
                        f"Could not extract name ({employee_name}) or year ({tax_year}) from {file_path.name}{doc_suffix}")
                    continue

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

                # For multi-document files, we need to create separate files
                if len(texts) > 1:
                    # For PDFs with multiple pages, split them
                    if file_path.suffix.lower() == '.pdf':
                        self.extract_single_page_pdf(file_path, doc_index, new_path)
                    else:
                        # For images, just copy the original (single page anyway)
                        shutil.copy2(file_path, new_path)
                else:
                    # Single document, just copy
                    shutil.copy2(file_path, new_path)

                logging.info(f"Renamed: {file_path.name}{doc_suffix} -> {new_filename}")
                successful_extractions += 1

            if successful_extractions > 0:
                self.processed_count += successful_extractions
                return True
            else:
                return False

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
            print(f"Progress: {i}/{total_files} ({i / total_files * 100:.1f}%)")

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