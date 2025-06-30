#!/usr/bin/env python3
"""
PDF Converter Utility
Converts text files to PDF format for the RAG system
"""

import os
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import argparse

class PDFConverter:
    def __init__(self, output_dir="documents/pdf"):
        self.output_dir = output_dir
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles for better formatting"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        )
        
        # Heading style
        self.heading_style = ParagraphStyle(
            'CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue
        )
        
        # Subheading style
        self.subheading_style = ParagraphStyle(
            'CustomSubheading',
            parent=self.styles['Heading3'],
            fontSize=12,
            spaceAfter=8,
            spaceBefore=12,
            textColor=colors.darkgreen
        )
        
        # Body text style
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=6,
            alignment=TA_JUSTIFY
        )
        
        # List item style
        self.list_style = ParagraphStyle(
            'CustomList',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=4,
            leftIndent=20,
            alignment=TA_LEFT
        )
    
    def convert_text_to_pdf(self, input_file, output_file=None):
        """Convert a text file to PDF format"""
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        if output_file is None:
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            output_file = os.path.join(self.output_dir, f"{base_name}.pdf")
        
        # Read the text file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Create PDF document
        doc = SimpleDocTemplate(output_file, pagesize=letter)
        story = []
        
        # Split content into lines
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            
            if not line:  # Empty line
                story.append(Spacer(1, 6))
                continue
            
            # Determine the style based on content
            if line.isupper() and len(line) < 100:  # Likely a title
                story.append(Paragraph(line, self.title_style))
            elif line.endswith(':') and len(line) < 50:  # Likely a heading
                story.append(Paragraph(line, self.heading_style))
            elif line.startswith('- ') or line.startswith('• '):  # List item
                story.append(Paragraph(line, self.list_style))
            elif line.isdigit() and len(line) <= 3:  # Section number
                continue  # Skip standalone numbers
            else:  # Regular body text
                story.append(Paragraph(line, self.body_style))
        
        # Build PDF
        doc.build(story)
        print(f"PDF created: {output_file}")
        return output_file
    
    def convert_all_documents(self, input_dir="documents"):
        """Convert all text files in the input directory to PDF"""
        if not os.path.exists(input_dir):
            print(f"Input directory not found: {input_dir}")
            return
        
        converted_files = []
        
        for filename in os.listdir(input_dir):
            if filename.endswith('.txt'):
                input_path = os.path.join(input_dir, filename)
                try:
                    output_path = self.convert_text_to_pdf(input_path)
                    converted_files.append(output_path)
                except Exception as e:
                    print(f"Error converting {filename}: {e}")
        
        print(f"\nConverted {len(converted_files)} files to PDF")
        return converted_files

def main():
    parser = argparse.ArgumentParser(description='Convert text files to PDF format')
    parser.add_argument('--input', '-i', help='Input text file or directory')
    parser.add_argument('--output', '-o', help='Output PDF file (for single file conversion)')
    parser.add_argument('--output-dir', '-d', default='documents/pdf', help='Output directory for PDFs')
    
    args = parser.parse_args()
    
    converter = PDFConverter(args.output_dir)
    
    if args.input:
        if os.path.isfile(args.input):
            # Convert single file
            converter.convert_text_to_pdf(args.input, args.output)
        elif os.path.isdir(args.input):
            # Convert all files in directory
            converter.convert_all_documents(args.input)
        else:
            print(f"Input path not found: {args.input}")
    else:
        # Convert all documents in default directory
        converter.convert_all_documents()

if __name__ == "__main__":
    main() 