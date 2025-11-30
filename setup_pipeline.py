"""
Setup and Configuration Utility for Fish ML Pipeline
Handles environment setup, model training orchestration, and system checks
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
import shutil
from datetime import datetime

class FishMLPipelineSetup:
    """Setup utility for the complete fish ML pipeline"""
    
    def __init__(self):
        self.base_dir = Path.cwd()
        self.models_dir = self.base_dir / 'models'
        self.data_dir = self.base_dir / 'data'
        self.logs_dir = self.base_dir / 'logs'
        
        # Required files for the pipeline
        self.required_files = {
            'train_image_model.py': 'CNN training script',
            'train_tabular_model.py': 'XGBoost training script', 
            'combined_inference.py': 'Combined prediction pipeline',
            'inference_streamlit.py': 'Streamlit web app',
            'tamil_nadu_fish_dataset_50k.csv': 'Synthetic training dataset'
        }
        
        # Model artifacts that should be created
        self.model_artifacts = {
            'best_fish_model.pt': 'Trained CNN model',
            'fish_freshness_model.pkl': 'Trained XGBoost model',
            'species_encoder.pkl': 'Species label encoder',
            'freshness_encoder.pkl': 'Freshness label encoder', 
            'tabular_label_encoders.pkl': 'Tabular model encoders',
            'training_results.json': 'CNN training results',
            'tabular_training_results.json': 'XGBoost training results'
        }
        
        # Dataset paths configuration
        self.default_dataset_paths = {
            'fish_market': r'E:\sea\datasets\fish_market',
            'freshness': r'E:\sea\datasets\fresh and non-fresh fish',
            'fish_dataset': r'E:\sea\datasets\Fish dataset',
            'rohu': r'E:\sea\datasets\Rohu'
        }
    
    def create_directories(self):
        """Create necessary directories"""
        print("📁 Creating directory structure...")
        
        directories = [self.models_dir, self.data_dir, self.logs_dir]
        
        for dir_path in directories:
            dir_path.mkdir(exist_ok=True)
            print(f"  ✅ {dir_path}")
    
    def check_requirements(self):
        """Check if all required packages are installed"""
        print("🔍 Checking Python package requirements...")
        
        required_packages = [
            'torch', 'torchvision', 'xgboost', 'scikit-learn',
            'pandas', 'numpy', 'matplotlib', 'seaborn', 
            'streamlit', 'plotly', 'pillow', 'faker'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✅ {package}")
            except ImportError:
                print(f"  ❌ {package}")
                missing_packages.append(package)
        
        if missing_packages:
            print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
            print("Install with: pip install " + " ".join(missing_packages))
            return False
        
        print("✅ All required packages are installed!")
        return True
    
    def check_files(self):
        """Check if all required files exist"""
        print("📄 Checking required files...")
        
        missing_files = []
        
        for file_name, description in self.required_files.items():
            file_path = self.base_dir / file_name
            if file_path.exists():
                print(f"  ✅ {file_name} ({description})")
            else:
                print(f"  ❌ {file_name} ({description})")
                missing_files.append(file_name)
        
        if missing_files:
            print(f"\n⚠️  Missing files: {', '.join(missing_files)}")
            return False
        
        print("✅ All required files found!")
        return True
    
    def check_dataset_paths(self, custom_paths=None):
        """Check if dataset paths exist"""
        print("💾 Checking dataset paths...")
        
        dataset_paths = custom_paths if custom_paths else self.default_dataset_paths
        valid_paths = {}
        
        for name, path in dataset_paths.items():
            if os.path.exists(path):
                print(f"  ✅ {name}: {path}")
                valid_paths[name] = path
            else:
                print(f"  ❌ {name}: {path} (not found)")
        
        if len(valid_paths) == 0:
            print("⚠️  No valid dataset paths found!")
            print("Please update paths in setup_pipeline.py or provide custom paths")
            return None
        
        print(f"✅ Found {len(valid_paths)}/{len(dataset_paths)} dataset paths")
        return valid_paths
    
    def create_config_file(self, dataset_paths):
        """Create configuration file"""
        config = {
            'dataset_paths': dataset_paths,
            'model_config': {
                'cnn_model': 'resnet50',
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 30,
                'train_split': 0.8
            },
            'tabular_config': {
                'model_type': 'xgboost',
                'test_size': 0.2,
                'optimize_hyperparams': True,
                'n_estimators': 200,
                'max_depth': 6
            },
            'inference_config': {
                'confidence_threshold': 0.7,
                'enable_batch_processing': True,
                'max_batch_size': 50
            },
            'setup_date': datetime.now().isoformat()
        }
        
        config_path = self.base_dir / 'pipeline_config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_path}")
        return config
    
    def train_models(self, dataset_paths, skip_if_exists=True):
        """Run the complete training pipeline"""
        print("🚀 Starting model training pipeline...")
        
        # Check if models already exist
        if skip_if_exists:
            existing_models = []
            for model_file in ['best_fish_model.pt', 'fish_freshness_model.pkl']:
                if (self.base_dir / model_file).exists():
                    existing_models.append(model_file)
            
            if existing_models:
                print(f"⚠️  Found existing models: {existing_models}")
                response = input("Do you want to retrain? (y/N): ").lower().strip()
                if response != 'y':
                    print("Skipping model training...")
                    return True
        
        # Step 1: Train CNN model
        print("\n📸 Step 1: Training CNN model...")
        try:
            # Update dataset paths in training script if needed
            self.update_dataset_paths_in_script('train_image_model.py', dataset_paths)
            
            result = subprocess.run([sys.executable, 'train_image_model.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CNN model training completed!")
            else:
                print(f"❌ CNN training failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running CNN training: {e}")
            return False
        
        # Step 2: Train tabular model
        print("\n📊 Step 2: Training XGBoost model...")
        try:
            result = subprocess.run([sys.executable, 'train_tabular_model.py'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ XGBoost model training completed!")
            else:
                print(f"❌ XGBoost training failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running XGBoost training: {e}")
            return False
        
        print("🎉 All models trained successfully!")
        return True
    
    def update_dataset_paths_in_script(self, script_name, dataset_paths):
        """Update dataset paths in training scripts"""
        script_path = self.base_dir / script_name
        
        if not script_path.exists():
            return
        
        # Read script content
        with open(script_path, 'r') as f:
            content = f.read()
        
        # Create backup
        backup_path = script_path.with_suffix('.py.bak')
        shutil.copy2(script_path, backup_path)
        
        # Update dataset paths
        for name, path in dataset_paths.items():
            old_pattern = f"'{name}': r'E:\\sea\\datasets\\{name.replace('_', ' ')}'"
            new_pattern = f"'{name}': r'{path}'"
            content = content.replace(old_pattern, new_pattern)
        
        # Write updated content
        with open(script_path, 'w') as f:
            f.write(content)
        
        print(f"  📝 Updated dataset paths in {script_name}")
    
    def check_model_artifacts(self):
        """Check if all model artifacts were created"""
        print("🔍 Checking model artifacts...")
        
        missing_artifacts = []
        
        for artifact, description in self.model_artifacts.items():
            artifact_path = self.base_dir / artifact
            if artifact_path.exists():
                print(f"  ✅ {artifact} ({description})")
            else:
                print(f"  ❌ {artifact} ({description})")
                missing_artifacts.append(artifact)
        
        if missing_artifacts:
            print(f"\n⚠️  Missing artifacts: {', '.join(missing_artifacts)}")
            return False
        
        print("✅ All model artifacts found!")
        return True
    
    def test_inference_pipeline(self):
        """Test the complete inference pipeline"""
        print("🧪 Testing inference pipeline...")
        
        try:
            # Import and test combined predictor
            sys.path.append(str(self.base_dir))
            from combined_inference import CombinedFishPredictor
            
            predictor = CombinedFishPredictor()
            print("✅ Combined predictor loaded successfully!")
            
            # Test with dummy parameters
            print("  Testing freshness prediction...")
            test_prediction = predictor.predict_freshness_days(
                species='Indian Mackerel',
                weight=200,
                storage_temp=5,
                hours_since_catch=12
            )
            print(f"  ✅ Test prediction: {test_prediction:.2f} days")
            
            return True
            
        except Exception as e:
            print(f"❌ Inference pipeline test failed: {e}")
            return False
    
    def run_streamlit_demo(self):
        """Launch Streamlit demo app"""
        print("🌐 Launching Streamlit demo app...")
        
        try:
            subprocess.Popen([sys.executable, '-m', 'streamlit', 'run', 'inference_streamlit.py'])
            print("✅ Streamlit app started!")
            print("🌐 Open http://localhost:8501 in your browser")
            return True
        except Exception as e:
            print(f"❌ Failed to start Streamlit: {e}")
            return False
    
    def full_setup(self, dataset_paths=None, train_models=True):
        """Run complete setup process"""
        print("🐟 FISH ML PIPELINE SETUP")
        print("=" * 60)
        
        # Step 1: Create directories
        self.create_directories()
        
        # Step 2: Check requirements
        if not self.check_requirements():
            return False
        
        # Step 3: Check files
        if not self.check_files():
            return False
        
        # Step 4: Check datasets
        valid_dataset_paths = self.check_dataset_paths(dataset_paths)
        if not valid_dataset_paths:
            return False
        
        # Step 5: Create config
        config = self.create_config_file(valid_dataset_paths)
        
        # Step 6: Train models (if requested)
        if train_models:
            if not self.train_models(valid_dataset_paths):
                return False
        
        # Step 7: Check model artifacts
        if not self.check_model_artifacts():
            print("⚠️  Some model artifacts missing. Run training pipeline first.")
        
        # Step 8: Test inference
        if not self.test_inference_pipeline():
            print("⚠️  Inference pipeline test failed. Check model artifacts.")
        
        print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("✅ Environment ready")
        print("✅ Models trained") 
        print("✅ Inference pipeline working")
        print("\n🚀 Next steps:")
        print("  1. Run: python setup_pipeline.py --demo")
        print("  2. Or: streamlit run inference_streamlit.py")
        print("  3. Upload fish images and test predictions!")
        
        return True


def main():
    """Main setup function with CLI"""
    parser = argparse.ArgumentParser(description='Fish ML Pipeline Setup')
    parser.add_argument('--check-only', action='store_true', 
                       help='Only check requirements and files')
    parser.add_argument('--train', action='store_true',
                       help='Run model training')
    parser.add_argument('--demo', action='store_true',
                       help='Launch Streamlit demo')
    parser.add_argument('--dataset-paths', type=str,
                       help='JSON file with custom dataset paths')
    
    args = parser.parse_args()
    
    setup = FishMLPipelineSetup()
    
    # Load custom dataset paths if provided
    custom_paths = None
    if args.dataset_paths:
        try:
            with open(args.dataset_paths, 'r') as f:
                custom_paths = json.load(f)
        except Exception as e:
            print(f"❌ Error loading dataset paths: {e}")
            return
    
    if args.check_only:
        # Just run checks
        setup.check_requirements()
        setup.check_files()
        setup.check_dataset_paths(custom_paths)
        setup.check_model_artifacts()
        
    elif args.train:
        # Just run training
        dataset_paths = setup.check_dataset_paths(custom_paths)
        if dataset_paths:
            setup.train_models(dataset_paths, skip_if_exists=False)
    
    elif args.demo:
        # Launch demo
        setup.run_streamlit_demo()
    
    else:
        # Full setup
        setup.full_setup(custom_paths, train_models=True)

if __name__ == "__main__":
    main()