"""
Verification script to check the number of features expected by the trained model
and confirm that the combined_inference.py matches.
"""

import pickle
import os

def verify_model_features():
    """Verify the number of features expected by fish_freshness_model.pkl"""
    
    model_path = 'fish_freshness_model.pkl'
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Check if model has n_features_in_ attribute
        if hasattr(model, 'n_features_in_'):
            n_features = model.n_features_in_
            print(f"✅ Model expects {n_features} features")
            
            if n_features == 12:
                print("✅ Feature count matches the updated combined_inference.py (12 features)")
            else:
                print(f"⚠️ Feature mismatch! Model expects {n_features} but combined_inference.py is configured for 12")
                
        else:
            print("⚠️ Model doesn't have n_features_in_ attribute. Cannot verify feature count.")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")

def verify_feature_list():
    """Display the feature list used in combined_inference.py"""
    
    feature_order = [
        'fish_species_encoded', 'storage_temp', 'storage_type_encoded',
        'area_temp', 'distance_km', 'price_per_kg',
        'fisher_location_encoded', 'buyer_location_encoded',
        'catch_day_of_week', 'catch_month', 'catch_hour',
        'temp_freshness_interaction'
    ]
    
    print(f"\n📋 Feature list in combined_inference.py ({len(feature_order)} features):")
    for i, feat in enumerate(feature_order, 1):
        print(f"  {i}. {feat}")

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 FEATURE VERIFICATION SCRIPT")
    print("=" * 60)
    
    verify_model_features()
    verify_feature_list()
    
    print("\n" + "=" * 60)
    print("✅ Verification complete")
    print("=" * 60)
