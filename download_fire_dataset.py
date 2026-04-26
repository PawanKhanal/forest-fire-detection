"""Download dataset using credentials from .env file."""

import os
import sys
import zipfile
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DatasetDownloader:
    """Handle Kaggle dataset download using .env credentials."""
    
    def __init__(self, dataset_name: str = None):
        """
        Initialize dataset downloader.
        
        Args:
            dataset_name: Kaggle dataset identifier (uses .env if None)
        """
        self.dataset_name = dataset_name or os.getenv('DATASET_NAME', 'elmadafri/the-wildfire-dataset')
        self.raw_dir = Path(os.getenv('DATA_DIR', 'data/raw'))
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._setup_environment()
    
    def _setup_environment(self) -> None:
        """Set up Kaggle environment variables."""
        username = os.getenv('KAGGLE_USERNAME')
        key = os.getenv('KAGGLE_KEY')
        
        if username and key:
            os.environ['KAGGLE_USERNAME'] = username
            os.environ['KAGGLE_KEY'] = key
        else:
            print("⚠️  Warning: Kaggle credentials not found in .env file")
    
    def download_with_kagglehub(self) -> bool:
        """
        Download using kagglehub (supports API tokens better).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import kagglehub
            
            print(f"📥 Downloading {self.dataset_name} using kagglehub...")
            path = kagglehub.dataset_download(self.dataset_name)
            
            # Copy to our data directory
            import shutil
            shutil.copytree(path, self.raw_dir, dirs_exist_ok=True)
            
            print("✅ Download complete!")
            return True
            
        except ImportError:
            print("⚠️  kagglehub not installed. Install with: pip install kagglehub")
            return False
        except Exception as e:
            print(f"❌ kagglehub download failed: {e}")
            return False
    
    def download_with_kaggle_cli(self) -> bool:
        """
        Download using Kaggle CLI.
        
        Returns:
            True if successful, False otherwise
        """
        print(f"📥 Downloading {self.dataset_name} using Kaggle CLI...")
        
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", self.dataset_name, "-p", str(self.raw_dir)],
                check=True,
                capture_output=True,
                text=True,
                env=os.environ.copy()
            )
            print("✅ Download complete!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Kaggle CLI download failed: {e.stderr}")
            return False
        except FileNotFoundError:
            print("❌ Kaggle CLI not found.")
            return False
    
    def download(self) -> bool:
        """
        Download dataset using best available method.
        
        Returns:
            True if successful, False otherwise
        """
        # Try kagglehub first (better for API tokens)
        if self.download_with_kagglehub():
            return True
        
        # Fallback to Kaggle CLI
        if self.download_with_kaggle_cli():
            return True
        
        return False
    
    def extract(self) -> bool:
        """
        Extract downloaded zip files if any exist.
        
        Returns:
            True if extraction successful or no zip files found
        """
        zip_files = list(self.raw_dir.glob("*.zip"))
        
        if not zip_files:
            print("ℹ️  No zip files to extract (data already extracted)")
            return True
        
        for zip_path in zip_files:
            print(f"📦 Extracting {zip_path.name}...")
            
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
                
                # Remove zip file after extraction
                zip_path.unlink()
                print(f"🧹 Cleaned up {zip_path.name}")
                
            except zipfile.BadZipFile:
                print(f"❌ Corrupted zip file: {zip_path.name}")
                return False
        
        return True
    
    def verify_structure(self) -> None:
        """Display dataset structure."""
        print("\n📁 Dataset structure:")
        print("=" * 50)
        
        total_images = 0
        for root, dirs, files in os.walk(self.raw_dir):
            level = root.replace(str(self.raw_dir), '').count(os.sep)
            indent = ' ' * 2 * level
            folder_name = os.path.basename(root)
            
            if level <= 2:  # Show only top levels
                print(f"{indent}{folder_name}/")
                
                subindent = ' ' * 2 * (level + 1)
                image_files = [f for f in files if f.endswith(('.jpg', '.png', '.jpeg'))]
                
                if image_files:
                    total_images += len(image_files)
                    print(f"{subindent}[{len(image_files)} images]")
        
        print(f"\n📊 Total images found: {total_images}")
    
    def run(self) -> None:
        """Execute complete download pipeline."""
        print("🌲 Forest Fire Dataset Downloader")
        print("=" * 50)
        
        # Check credentials
        username = os.getenv('KAGGLE_USERNAME')
        if not username:
            print("❌ Kaggle credentials not found in .env file!")
            print("\n📋 Please create a .env file with:")
            print("KAGGLE_USERNAME=your_username")
            print("KAGGLE_KEY=your_api_token")
            sys.exit(1)
        
        print(f"👤 Authenticated as: {username}")
        print()
        
        if self.download():
            if self.extract():
                self.verify_structure()
                print("\n✅ Dataset ready for training!")
            else:
                print("\n⚠️  Extraction issues. Check data/raw/ directory.")
        else:
            print("\n❌ All download methods failed.")
            print("\n📋 Manual download:")
            print(f"1. Visit: https://www.kaggle.com/datasets/{self.dataset_name}")
            print("2. Click 'Download'")
            print(f"3. Extract to: {self.raw_dir}")

def main():
    """Main execution."""
    downloader = DatasetDownloader()
    downloader.run()

if __name__ == "__main__":
    main()