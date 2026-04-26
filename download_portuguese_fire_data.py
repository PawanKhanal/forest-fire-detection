"""Download Portuguese Forest Fire Dataset from UCI Repository."""

import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import urllib.request
import zipfile

# Load environment variables
load_dotenv()

class PortugueseFireDataDownloader:
    """
    Download Portuguese Forest Fire Dataset from UCI ML Repository.
    This is REAL sensor data with temperature, humidity, wind, rain
    and actual fire occurrence labels.
    """
    
    def __init__(self):
        """Initialize the downloader."""
        self.dataset_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
        self.data_dir = Path(os.getenv('SENSOR_DATA_DIR', 'data/sensor'))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.data_dir / "forestfires.csv"
        
    def download(self) -> bool:
        """
        Download the dataset from UCI Repository.
        
        Returns:
            True if successful, False otherwise
        """
        print("🌲 Downloading Portuguese Forest Fire Dataset")
        print("=" * 50)
        print(f"Source: UCI Machine Learning Repository")
        print(f"Target: {self.dataset_path}")
        print()
        
        try:
            print("📥 Downloading forestfires.csv...")
            urllib.request.urlretrieve(self.dataset_url, self.dataset_path)
            print("✅ Download complete!")
            return True
            
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return False
    
    def verify_dataset(self) -> None:
        """Verify and display dataset statistics."""
        print("\n📊 Dataset Verification")
        print("=" * 50)
        
        try:
            df = pd.read_csv(self.dataset_path)
            
            print(f"✅ Dataset loaded successfully!")
            print(f"\n📈 Dataset Overview:")
            print(f"   Total records: {len(df)}")
            print(f"   Features: {list(df.columns)}")
            
            # Fire statistics
            fire_days = (df['area'] > 0).sum()
            print(f"\n🔥 Fire Statistics:")
            print(f"   Days with fire: {fire_days} ({fire_days/len(df)*100:.1f}%)")
            print(f"   Total burned area: {df['area'].sum():.2f} hectares")
            print(f"   Average fire size: {df[df['area']>0]['area'].mean():.2f} hectares")
            
            # Weather statistics
            print(f"\n🌡️  Weather Statistics:")
            print(f"   Temperature range: {df['temp'].min():.1f}°C - {df['temp'].max():.1f}°C")
            print(f"   Humidity range: {df['RH'].min():.1f}% - {df['RH'].max():.1f}%")
            print(f"   Wind speed range: {df['wind'].min():.1f} - {df['wind'].max():.1f} km/h")
            
            # Compare fire vs non-fire days
            fire_df = df[df['area'] > 0]
            no_fire_df = df[df['area'] == 0]
            
            print(f"\n📊 Fire Days vs Non-Fire Days:")
            print(f"   Fire days - Avg Temp: {fire_df['temp'].mean():.1f}°C")
            print(f"   Non-fire days - Avg Temp: {no_fire_df['temp'].mean():.1f}°C")
            print(f"   Fire days - Avg Humidity: {fire_df['RH'].mean():.1f}%")
            print(f"   Non-fire days - Avg Humidity: {no_fire_df['RH'].mean():.1f}%")
            
            # Save enhanced version with binary label
            df['fire_occurred'] = (df['area'] > 0).astype(int)
            enhanced_path = self.data_dir / "forestfires_enhanced.csv"
            df.to_csv(enhanced_path, index=False)
            print(f"\n💾 Enhanced dataset saved to: {enhanced_path}")
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
    
    def create_sample_for_arduino(self) -> None:
        """
        Create a simplified version with only temperature and humidity.
        This matches what your Arduino DHT11 can measure.
        """
        try:
            df = pd.read_csv(self.dataset_path)
            
            # Create binary label
            df['fire_occurred'] = (df['area'] > 0).astype(int)
            
            # Select only temp and humidity (what Arduino measures)
            arduino_df = df[['temp', 'RH', 'fire_occurred']]
            arduino_df.columns = ['temperature', 'humidity', 'fire_risk']
            
            arduino_path = self.data_dir / "arduino_training_data.csv"
            arduino_df.to_csv(arduino_path, index=False)
            
            print(f"\n🔧 Arduino-compatible dataset saved to: {arduino_path}")
            print(f"   Features: temperature, humidity")
            print(f"   Target: fire_risk (0/1)")
            
        except Exception as e:
            print(f"❌ Failed to create Arduino dataset: {e}")
    
    def display_sample(self) -> None:
        """Display sample records from the dataset."""
        print("\n📋 Sample Records:")
        print("=" * 70)
        
        try:
            df = pd.read_csv(self.dataset_path)
            df['fire_occurred'] = (df['area'] > 0).astype(int)
            
            # Show 3 fire days and 3 non-fire days
            fire_samples = df[df['fire_occurred'] == 1].head(3)
            no_fire_samples = df[df['fire_occurred'] == 0].head(3)
            
            print("\n🔥 FIRE DAYS:")
            for idx, row in fire_samples.iterrows():
                print(f"  Day {idx}: {row['temp']:.1f}°C, {row['RH']:.1f}% humidity, "
                      f"{row['wind']:.1f} km/h wind, {row['area']:.2f} ha burned")
            
            print("\n✅ NON-FIRE DAYS:")
            for idx, row in no_fire_samples.iterrows():
                print(f"  Day {idx}: {row['temp']:.1f}°C, {row['RH']:.1f}% humidity, "
                      f"{row['wind']:.1f} km/h wind")
                      
        except Exception as e:
            print(f"❌ Failed to display samples: {e}")
    
    def run(self) -> None:
        """Execute complete download and verification pipeline."""
        print("🔥 Portuguese Forest Fire Dataset Downloader")
        print("=" * 50)
        print("Dataset: Cortez & Morais, 2007")
        print("Source: UCI Machine Learning Repository")
        print()
        
        if self.download():
            self.verify_dataset()
            self.create_sample_for_arduino()
            self.display_sample()
            print("\n✅ Dataset ready for ML training!")
            print("\n📋 Next steps:")
            print("   1. Run: python train_sensor_model.py")
            print("   2. Model will be saved to models/saved/sensor_model.pkl")
        else:
            print("\n❌ Download failed.")
            print("\n📋 Manual download:")
            print("   1. Visit: https://archive.ics.uci.edu/ml/datasets/Forest+Fires")
            print("   2. Download: forestfires.csv")
            print(f"   3. Save to: {self.data_dir}/")

def main():
    """Main execution."""
    downloader = PortugueseFireDataDownloader()
    downloader.run()

if __name__ == "__main__":
    main()