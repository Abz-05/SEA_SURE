#!/usr/bin/env python3
"""
Fix all use_container_width deprecations in app_integreted.py
Replaces width="stretch" with width="stretch"
"""

import re

def fix_file(filepath):
    """Fix use_container_width in a file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Count original occurrences
    original_count = content.count('use_container_width')
    
    # Replace width="stretch" with width="stretch"
    content = re.sub(
        r'use_container_width\s*=\s*True',
        'width="stretch"',
        content
    )
    
    # Replace width="content" with width="content"  
    content = re.sub(
        r'use_container_width\s*=\s*False',
        'width="content"',
        content
    )
    
    # Count remaining
    remaining_count = content.count('use_container_width')
    
    # Write back
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    fixed_count = original_count - remaining_count
    print(f"✅ Fixed {fixed_count} occurrences in {filepath}")
    if remaining_count > 0:
        print(f"⚠️  Warning: {remaining_count} occurrences remain (may need manual review)")
    
    return fixed_count

if __name__ == "__main__":
    files_to_fix = [
        "c:\\sea\\app_integreted.py",
        "c:\\sea\\app.py"
    ]
    
    total_fixed = 0
    for filepath in files_to_fix:
        try:
            fixed = fix_file(filepath)
            total_fixed += fixed
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error processing {filepath}: {e}")
    
    print(f"\n🎉 Total fixed: {total_fixed} occurrences")
