#!/usr/bin/env python3
"""
Script to analyze and display the structure of a JSONL file.
JSONL files contain one JSON object per line.
"""

import json
import argparse
import sys
from collections import defaultdict
from typing import Dict, Any, List, Set


def analyze_json_value(value: Any, path: str = "", structure: Dict = None, max_depth: int = 5) -> Dict:
    """
    Recursively analyze a JSON value to extract structure information.
    
    Args:
        value: The JSON value to analyze
        path: Current path in the JSON structure
        structure: Dictionary to store structure information
        max_depth: Maximum depth to analyze
    
    Returns:
        Dictionary containing structure information
    """
    if structure is None:
        structure = {
            "paths": set(),
            "types": {},
            "sample_values": {},
            "depth": 0
        }
    
    if max_depth <= 0:
        return structure
    
    structure["depth"] = max(structure["depth"], len(path.split(".")) if path else 0)
    
    if isinstance(value, dict):
        for key, val in value.items():
            current_path = f"{path}.{key}" if path else key
            structure["paths"].add(current_path)
            value_type = type(val).__name__
            
            if current_path not in structure["types"]:
                structure["types"][current_path] = set()
            structure["types"][current_path].add(value_type)
            
            if current_path not in structure["sample_values"]:
                structure["sample_values"][current_path] = []
            if len(structure["sample_values"][current_path]) < 3:
                # Store sample values (truncated if too long)
                sample = str(val)
                if len(sample) > 100:
                    sample = sample[:100] + "..."
                structure["sample_values"][current_path].append(sample)
            
            analyze_json_value(val, current_path, structure, max_depth - 1)
    
    elif isinstance(value, list):
        if value:  # Only analyze non-empty lists
            # Analyze the first few items to understand the structure
            for i, item in enumerate(value[:3]):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                structure["paths"].add(current_path)
                value_type = type(item).__name__
                
                list_path = f"{path}[]" if path else "[]"
                if list_path not in structure["types"]:
                    structure["types"][list_path] = set()
                structure["types"][list_path].add(value_type)
                
                analyze_json_value(item, current_path, structure, max_depth - 1)
    
    return structure


def analyze_jsonl_file(file_path: str, max_lines: int = 100, max_depth: int = 5) -> Dict[str, Any]:
    """
    Analyze a JSONL file and return structure information.
    
    Args:
        file_path: Path to the JSONL file
        max_lines: Maximum number of lines to analyze
        max_depth: Maximum depth to analyze for each JSON object
    
    Returns:
        Dictionary containing structure analysis
    """
    combined_structure = {
        "file_info": {
            "path": file_path,
            "total_lines": 0,
            "analyzed_lines": 0,
            "valid_lines": 0,
            "invalid_lines": 0
        },
        "structure": {
            "paths": set(),
            "types": {},
            "sample_values": {},
            "depth": 0
        },
        "line_stats": []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                combined_structure["file_info"]["total_lines"] += 1
                
                if line_num > max_lines:
                    break
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    json_obj = json.loads(line)
                    combined_structure["file_info"]["valid_lines"] += 1
                    combined_structure["file_info"]["analyzed_lines"] += 1
                    
                    # Analyze this JSON object
                    line_structure = analyze_json_value(json_obj, max_depth=max_depth)
                    
                    # Merge with combined structure
                    combined_structure["structure"]["paths"].update(line_structure["paths"])
                    
                    for path, types in line_structure["types"].items():
                        if path not in combined_structure["structure"]["types"]:
                            combined_structure["structure"]["types"][path] = set()
                        combined_structure["structure"]["types"][path].update(types)
                    
                    for path, values in line_structure["sample_values"].items():
                        if path not in combined_structure["structure"]["sample_values"]:
                            combined_structure["structure"]["sample_values"][path] = []
                        # Add new unique sample values
                        for val in values:
                            if val not in combined_structure["structure"]["sample_values"][path]:
                                combined_structure["structure"]["sample_values"][path].append(val)
                                if len(combined_structure["structure"]["sample_values"][path]) >= 3:
                                    break
                    
                    combined_structure["structure"]["depth"] = max(
                        combined_structure["structure"]["depth"], 
                        line_structure["depth"]
                    )
                    
                    # Store line statistics
                    line_stats = {
                        "line_number": line_num,
                        "object_type": type(json_obj).__name__,
                        "keys": list(json_obj.keys()) if isinstance(json_obj, dict) else None,
                        "array_length": len(json_obj) if isinstance(json_obj, list) else None
                    }
                    combined_structure["line_stats"].append(line_stats)
                    
                except json.JSONDecodeError as e:
                    combined_structure["file_info"]["invalid_lines"] += 1
                    print(f"Warning: Invalid JSON on line {line_num}: {e}", file=sys.stderr)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)
    
    return combined_structure


def print_structure(analysis: Dict[str, Any]) -> None:
    """
    Print the structure analysis in a readable format.
    
    Args:
        analysis: The analysis result from analyze_jsonl_file
    """
    file_info = analysis["file_info"]
    structure = analysis["structure"]
    
    print("=" * 60)
    print("JSONL FILE STRUCTURE ANALYSIS")
    print("=" * 60)
    
    print(f"\nFile: {file_info['path']}")
    print(f"Total lines: {file_info['total_lines']}")
    print(f"Analyzed lines: {file_info['analyzed_lines']}")
    print(f"Valid JSON lines: {file_info['valid_lines']}")
    print(f"Invalid JSON lines: {file_info['invalid_lines']}")
    print(f"Maximum depth: {structure['depth']}")
    
    print("\n" + "=" * 60)
    print("STRUCTURE PATHS")
    print("=" * 60)
    
    # Sort paths for consistent output
    sorted_paths = sorted(structure["paths"])
    
    for path in sorted_paths:
        types = structure["types"].get(path, set())
        type_str = ", ".join(sorted(types))
        samples = structure["sample_values"].get(path, [])
        
        print(f"\nPath: {path}")
        print(f"  Type(s): {type_str}")
        
        if samples:
            print(f"  Sample values:")
            for i, sample in enumerate(samples, 1):
                print(f"    {i}. {sample}")
    
    print("\n" + "=" * 60)
    print("LINE STATISTICS")
    print("=" * 60)
    
    for stat in analysis["line_stats"][:10]:  # Show first 10 lines
        print(f"\nLine {stat['line_number']}:")
        print(f"  Object type: {stat['object_type']}")
        
        if stat['keys'] is not None:
            print(f"  Keys: {', '.join(stat['keys'])}")
        
        if stat['array_length'] is not None:
            print(f"  Array length: {stat['array_length']}")
    
    if len(analysis["line_stats"]) > 10:
        print(f"\n... and {len(analysis['line_stats']) - 10} more lines")


def main():
    """Main function to handle command line arguments and run the analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze and display the structure of a JSONL file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python jsonl_structure_analyzer.py data.jsonl
  python jsonl_structure_analyzer.py data.jsonl --max-lines 50
  python jsonl_structure_analyzer.py data.jsonl --max-depth 3
  python jsonl_structure_analyzer.py data.jsonl --output structure.json
        """
    )
    
    parser.add_argument(
        "file_path",
        help="Path to the JSONL file to analyze"
    )
    
    parser.add_argument(
        "--max-lines",
        type=int,
        default=100,
        help="Maximum number of lines to analyze (default: 100)"
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=5,
        help="Maximum depth to analyze for each JSON object (default: 5)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        help="Output file to save the analysis (JSON format)"
    )
    
    args = parser.parse_args()
    
    # Analyze the file
    analysis = analyze_jsonl_file(args.file_path, args.max_lines, args.max_depth)
    
    # Convert sets to lists for JSON serialization
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        else:
            return obj
    
    analysis_serializable = convert_sets(analysis)
    
    # Save to output file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
        print(f"Analysis saved to: {args.output}")
    
    # Print the structure
    print_structure(analysis)


if __name__ == "__main__":
    main()