"""
Animation Organizer Utility
Helps organize .vrma files into emotion-based folders.

Run this script in the same directory as your 'animations' folder.
"""

import os
import shutil
from pathlib import Path

# Keywords that suggest which category an animation belongs to
CATEGORY_KEYWORDS = {
    'idle': ['idle', 'stand', 'breath', 'wait', 'rest', 'relax', 'default'],
    'happy': ['happy', 'joy', 'smile', 'laugh', 'cheer', 'celebrate', 'excited', 'yay'],
    'sad': ['sad', 'cry', 'sorrow', 'depress', 'down', 'slouch', 'sigh'],
    'angry': ['angry', 'anger', 'mad', 'rage', 'stomp', 'fist', 'frustrated'],
    'surprised': ['surprise', 'shock', 'gasp', 'wow', 'jump', 'startle'],
    'talking': ['talk', 'gesture', 'point', 'explain', 'shrug', 'hand'],
    'greeting': ['wave', 'bow', 'hello', 'hi', 'bye', 'greet', 'welcome'],
    'dance': ['dance', 'groove', 'move', 'sway', 'bounce'],
}

def suggest_category(filename: str) -> str:
    """Suggest a category based on filename keywords"""
    name_lower = filename.lower()
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in name_lower:
                return category
    
    return 'general'

def organize_animations(animations_dir: Path, dry_run: bool = True):
    """Organize animations into category folders"""
    
    if not animations_dir.exists():
        print(f"❌ Directory not found: {animations_dir}")
        return
    
    # Find all .vrma files in root of animations folder
    vrma_files = list(animations_dir.glob("*.vrma"))
    
    if not vrma_files:
        print("No .vrma files found in root of animations folder.")
        print("(Files already in subfolders are left as-is)")
        return
    
    print(f"Found {len(vrma_files)} .vrma file(s) to organize:\n")
    
    # Group by suggested category
    moves = {}
    for file_path in vrma_files:
        category = suggest_category(file_path.stem)
        if category not in moves:
            moves[category] = []
        moves[category].append(file_path)
    
    # Show planned moves
    for category, files in sorted(moves.items()):
        print(f"📁 {category}/")
        for f in files:
            print(f"   ← {f.name}")
    
    print()
    
    if dry_run:
        print("This is a DRY RUN. No files were moved.")
        print("Run with --move to actually move files.")
        return
    
    # Actually move files
    for category, files in moves.items():
        category_dir = animations_dir / category
        category_dir.mkdir(exist_ok=True)
        
        for file_path in files:
            dest = category_dir / file_path.name
            shutil.move(str(file_path), str(dest))
            print(f"✓ Moved {file_path.name} → {category}/")
    
    print("\n✅ Organization complete!")

def list_current_structure(animations_dir: Path):
    """Show current animation folder structure"""
    
    if not animations_dir.exists():
        print(f"❌ Directory not found: {animations_dir}")
        return
    
    print(f"\n📂 Current structure of {animations_dir}:\n")
    
    # Count files
    total = 0
    
    # Files in root
    root_files = list(animations_dir.glob("*.vrma"))
    if root_files:
        print(f"(root) - {len(root_files)} file(s)")
        for f in root_files[:5]:
            print(f"  - {f.name}")
        if len(root_files) > 5:
            print(f"  ... and {len(root_files) - 5} more")
        total += len(root_files)
    
    # Files in subfolders
    for subdir in sorted(animations_dir.iterdir()):
        if subdir.is_dir():
            files = list(subdir.glob("*.vrma"))
            if files:
                print(f"{subdir.name}/ - {len(files)} file(s)")
                for f in files[:3]:
                    print(f"  - {f.name}")
                if len(files) > 3:
                    print(f"  ... and {len(files) - 3} more")
                total += len(files)
    
    print(f"\nTotal: {total} animation(s)")

def interactive_organize(animations_dir: Path):
    """Interactive mode for organizing animations"""
    
    vrma_files = list(animations_dir.glob("*.vrma"))
    
    if not vrma_files:
        print("No loose .vrma files to organize.")
        return
    
    print(f"\nInteractive organization of {len(vrma_files)} file(s)")
    print("For each file, enter a category name or press Enter to accept suggestion.")
    print("Type 'skip' to leave file in place, 'quit' to stop.\n")
    
    for file_path in vrma_files:
        suggested = suggest_category(file_path.stem)
        
        response = input(f"{file_path.name} → [{suggested}]: ").strip().lower()
        
        if response == 'quit':
            break
        elif response == 'skip':
            continue
        elif response == '':
            category = suggested
        else:
            category = response
        
        # Move file
        category_dir = animations_dir / category
        category_dir.mkdir(exist_ok=True)
        dest = category_dir / file_path.name
        shutil.move(str(file_path), str(dest))
        print(f"  ✓ Moved to {category}/\n")

def main():
    import sys
    
    animations_dir = Path("./animations")
    
    print("=" * 50)
    print("🎬 Animation Organizer Utility")
    print("=" * 50)
    
    # Check command line args
    if len(sys.argv) > 1:
        if sys.argv[1] == '--move':
            organize_animations(animations_dir, dry_run=False)
        elif sys.argv[1] == '--list':
            list_current_structure(animations_dir)
        elif sys.argv[1] == '--interactive':
            interactive_organize(animations_dir)
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("\nUsage:")
            print("  python animation_organizer.py           # Dry run (preview)")
            print("  python animation_organizer.py --move    # Actually move files")
            print("  python animation_organizer.py --list    # Show current structure")
            print("  python animation_organizer.py --interactive  # Organize one by one")
    else:
        # Show menu
        print("\nOptions:")
        print("  1. Preview organization (dry run)")
        print("  2. Organize files (move)")
        print("  3. Show current structure")
        print("  4. Interactive mode")
        print("  0. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            organize_animations(animations_dir, dry_run=True)
        elif choice == '2':
            confirm = input("This will move files. Continue? (y/n): ")
            if confirm.lower() == 'y':
                organize_animations(animations_dir, dry_run=False)
        elif choice == '3':
            list_current_structure(animations_dir)
        elif choice == '4':
            interactive_organize(animations_dir)
        elif choice == '0':
            print("Bye!")
        else:
            print("Invalid choice")

if __name__ == "__main__":
    main()
