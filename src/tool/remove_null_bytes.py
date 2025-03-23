#!/usr/bin/env python3
"""
Simple script to remove null bytes from a file.
"""

import sys

def remove_null_bytes(input_file, output_file=None):
    """Remove null bytes from a file."""
    if output_file is None:
        output_file = input_file
    
    print(f"Processing file: {input_file}")
    
    # Read the file and replace null bytes
    with open(input_file, 'rb') as file:
        content = file.read().replace(b'\x00', b'')
    
    # Write the cleaned content back
    with open(output_file, 'wb') as file:
        file.write(content)
    
    print(f"Null bytes removed. Output saved to: {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python remove_null_bytes.py <input_file> [output_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    remove_null_bytes(input_file, output_file) 