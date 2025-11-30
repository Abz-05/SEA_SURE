import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import uuid
from faker import Faker
import math
from typing import Dict, List, Tuple
# Initialize Faker for Indian data
fake = Faker('en_IN')
np.random.seed(42)
random.seed(42)

class TamilNaduFishDatasetGenerator:
    def __init__(self):
        # Fish species that EXACTLY match your image folder names (excluding FishImgDataset)
        self.aligned_fish_species = {
    'Indian Mackerel': {
        'scientific_name': 'Rastrelliger kanagurta',
        'tamil_name': 'கன்னங்கத்தா / அயிலா',
        'english_name': 'Indian Mackerel',
        'freshness_0_5_c': 3.5,      # Days at 0-5°C
        'freshness_5_15_c': 2.5,     # Days at 5-15°C
        'freshness_15_25_c': 1.5,    # Days at 15-25°C
        'min_length_cm': 15,
        'max_length_cm': 35,
        'min_weight_g': 80,
        'max_weight_g': 350,
        'min_price_kg': 280,
        'max_price_kg': 320,
        'image_folders': ['Indian Mackerel(Oozi)-Goatfish(Oola)'],
        'freshness_indicators': ['clear_eyes', 'bright_red_gills', 'firm_flesh', 'metallic_shine']
    },
    'Anchovy': {
        'scientific_name': 'Stolephorus commersonnii',
        'tamil_name': 'நெத்திலி',
        'english_name': 'Indian Anchovy',
        'freshness_0_5_c': 2.5,
        'freshness_5_15_c': 1.5,
        'freshness_15_25_c': 0.75,
        'min_length_cm': 6,
        'max_length_cm': 15,
        'min_weight_g': 2,
        'max_weight_g': 30,
        'min_price_kg': 280,
        'max_price_kg': 320,
        'image_folders': ['Indian Anchovy(Nethili)'],
        'freshness_indicators': ['silver_shine', 'clear_eyes', 'firm_texture']
    },
    'Oil Sardine': {
        'scientific_name': 'Sardinella longiceps',
        'tamil_name': 'மத்தி / சாளை',
        'english_name': 'Oil Sardine',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 18,
        'max_length_cm': 35,
        'min_weight_g': 100,
        'max_weight_g': 400,
        'min_price_kg': 200,
        'max_price_kg': 230,
        'image_folders': ['Indian Oil Sardine(Maththi)-Sardine(Saala)'],
        'freshness_indicators': ['silver_scales', 'red_gills', 'firm_body']
    },
    'Pomfret': {
        'scientific_name': 'Pampus argenteus',
        'tamil_name': 'வவ்வால் மீன்',
        'english_name': 'Silver Pomfret',
        'freshness_0_5_c': 4.5,
        'freshness_5_15_c': 3.5,
        'freshness_15_25_c': 2.5,
        'min_length_cm': 20,
        'max_length_cm': 40,
        'min_weight_g': 200,
        'max_weight_g': 1500,
        'min_price_kg': 500,
        'max_price_kg': 750,
        'image_folders': ['Pomfret'],
        'freshness_indicators': ['silver_skin', 'clear_eyes', 'firm_flesh']
    },
    'Black Pomfret': {
        'scientific_name': 'Parastromateus niger',
        'tamil_name': 'கறுப்பு வவ்வால்',
        'english_name': 'Black Pomfret',
        'freshness_0_5_c': 4.0,
        'freshness_5_15_c': 3.0,
        'freshness_15_25_c': 2.0,
        'min_length_cm': 25,
        'max_length_cm': 45,
        'min_weight_g': 300,
        'max_weight_g': 2000,
        'min_price_kg': 600,
        'max_price_kg': 900,
        'image_folders': ['Black Pomfret'],
        'freshness_indicators': ['dark_skin', 'clear_eyes', 'firm_flesh']
    },
    'Seer Fish': {
        'scientific_name': 'Scomberomorus commerson',
        'tamil_name': 'வஞ்சரம் / நாகர',
        'english_name': 'King Mackerel',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 40,
        'max_length_cm': 200,
        'min_weight_g': 2000,
        'max_weight_g': 45000,
        'min_price_kg': 1800,
        'max_price_kg': 2100,
        'image_folders': ['Seer Fish -King Mackerel(Nagara)', 'Red Seer Fish(Sevapu Nagara)'],
        'freshness_indicators': ['metallic_back', 'red_gills', 'firm_flesh']
    },
    'Ribbon Fish': {
        'scientific_name': 'Trichiurus lepturus',
        'tamil_name': 'கனவாய் / வாளை',
        'english_name': 'Ribbon Fish',
        'freshness_0_5_c': 2.5,
        'freshness_5_15_c': 1.5,
        'freshness_15_25_c': 0.75,
        'min_length_cm': 50,
        'max_length_cm': 150,
        'min_weight_g': 100,
        'max_weight_g': 500,
        'min_price_kg': 200,
        'max_price_kg': 350,
        'image_folders': ['Ribbon Fish(Kanavai)'],
        'freshness_indicators': ['silver_skin', 'firm_flesh', 'clear_eyes']
    },
    'Barramundi': {
        'scientific_name': 'Lates calcarifer',
        'tamil_name': 'கொடுவா / சீபாஸ்',
        'english_name': 'Sea Bass',
        'freshness_0_5_c': 4.5,
        'freshness_5_15_c': 3.5,
        'freshness_15_25_c': 2.5,
        'min_length_cm': 30,
        'max_length_cm': 90,
        'min_weight_g': 1000,
        'max_weight_g': 10000,
        'min_price_kg': 600,
        'max_price_kg': 1000,
        'image_folders': ['Barramundi -Sea Bass(Vaaval)'],
        'freshness_indicators': ['silver_white_body', 'sweet_smell', 'firm_texture']
    },
    'Catfish': {
        'scientific_name': 'Tachysurus thalassinus',
        'tamil_name': 'கெலுத்தி / கேளங்கன்',
        'english_name': 'Marine Catfish',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 30,
        'max_length_cm': 60,
        'min_weight_g': 300,
        'max_weight_g': 1500,
        'min_price_kg': 150,
        'max_price_kg': 280,
        'image_folders': ['Catfish-(Keluthi)-(Kelangan)'],
        'freshness_indicators': ['clear_eyes', 'firm_flesh', 'no_slime']
    },
    'Silver Catfish': {
        'scientific_name': 'Pangasius hypophthalmus',
        'tamil_name': 'வெள்ளி கெலுத்தி',
        'english_name': 'Silver Catfish',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 25,
        'max_length_cm': 50,
        'min_weight_g': 300,
        'max_weight_g': 2000,
        'min_price_kg': 200,
        'max_price_kg': 350,
        'image_folders': ['Silver Catfish'],
        'freshness_indicators': ['silver_skin', 'clear_eyes', 'no_slime']
    },
    'Trevally': {
        'scientific_name': 'Caranx ignobilis',
        'tamil_name': 'பாரா மீன் / வெளா மீன்',
        'english_name': 'Giant Trevally',
        'freshness_0_5_c': 4.5,
        'freshness_5_15_c': 3.5,
        'freshness_15_25_c': 2.5,
        'min_length_cm': 30,
        'max_length_cm': 70,
        'min_weight_g': 500,
        'max_weight_g': 3000,
        'min_price_kg': 400,
        'max_price_kg': 650,
        'image_folders': ['Trevally-Rockfish(Paara Meen)', 'Trevally (Velaa meen)'],
        'freshness_indicators': ['silver_body', 'firm_flesh', 'clear_eyes']
    },
    'Snakehead': {
        'scientific_name': 'Channa striata',
        'tamil_name': 'வரால்',
        'english_name': 'Murrel Fish',
        'freshness_0_5_c': 4.0,
        'freshness_5_15_c': 3.0,
        'freshness_15_25_c': 2.0,
        'min_length_cm': 25,
        'max_length_cm': 50,
        'min_weight_g': 200,
        'max_weight_g': 1000,
        'min_price_kg': 250,
        'max_price_kg': 450,
        'image_folders': ['Snakehead Fish(Murrel Meen)-Longnose Garfish(Mooku Oola)'],
        'freshness_indicators': ['firm_flesh', 'clear_eyes', 'natural_color']
    },
    'White Grouper': {
        'scientific_name': 'Epinephelus aeneus',
        'tamil_name': 'வெள்ளை கேளங்கன்',
        'english_name': 'White Grouper',
        'freshness_0_5_c': 4.0,
        'freshness_5_15_c': 3.0,
        'freshness_15_25_c': 2.0,
        'min_length_cm': 25,
        'max_length_cm': 60,
        'min_weight_g': 500,
        'max_weight_g': 2000,
        'min_price_kg': 350,
        'max_price_kg': 550,
        'image_folders': ['White Grouper(Vellai Kelangan)'],
        'freshness_indicators': ['white_flesh', 'clear_eyes', 'firm_texture']
    },
    'Prawn': {
        'scientific_name': 'Penaeus indicus',
        'tamil_name': 'இறால்',
        'english_name': 'Indian White Prawn',
        'freshness_0_5_c': 2.5,
        'freshness_5_15_c': 1.5,
        'freshness_15_25_c': 0.5,
        'min_length_cm': 8,
        'max_length_cm': 18,
        'min_weight_g': 5,
        'max_weight_g': 50,
        'min_price_kg': 450,
        'max_price_kg': 750,
        'image_folders': ['Prawn'],
        'freshness_indicators': ['translucent_body', 'firm_texture', 'no_blackening']
    },
    'Jelabi Kenda': {
        'scientific_name': 'Sillago sihama',
        'tamil_name': 'ஜெலாபி கென்டா',
        'english_name': 'Silver Sillago',
        'freshness_0_5_c': 3.0,
        'freshness_5_15_c': 2.0,
        'freshness_15_25_c': 1.0,
        'min_length_cm': 15,
        'max_length_cm': 25,
        'min_weight_g': 50,
        'max_weight_g': 200,
        'min_price_kg': 300,
        'max_price_kg': 450,
        'image_folders': ['Jelabi Kenda fish'],
        'freshness_indicators': ['silver_body', 'clear_eyes', 'firm_flesh']
    },
    'Black Snapper': {
        'scientific_name': 'Lutjanus fulviflamma',
        'tamil_name': 'கறுப்பு சிவப்பு மீன்',
        'english_name': 'Black Snapper',
        'freshness_0_5_c': 4.5,
        'freshness_5_15_c': 3.5,
        'freshness_15_25_c': 2.5,
        'min_length_cm': 20,
        'max_length_cm': 40,
        'min_weight_g': 400,
        'max_weight_g': 1800,
        'min_price_kg': 450,
        'max_price_kg': 700,
        'image_folders': ['Black Snapper'],
        'freshness_indicators': ['red_hue', 'clear_eyes', 'firm_flesh']
    },
    'Indian Carp': {
        'scientific_name': 'Labeo rohita',
        'tamil_name': 'கெண்டை மீன்',
        'english_name': 'Rohu Carp',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 30,
        'max_length_cm': 60,
        'min_weight_g': 800,
        'max_weight_g': 4000,
        'min_price_kg': 180,
        'max_price_kg': 280,
        'image_folders': ['Indian Carp'],
        'freshness_indicators': ['bright_scales', 'red_gills', 'no_muddy_smell']
    },
    'Pink Perch': {
        'scientific_name': 'Nemipterus japonicus',
        'tamil_name': 'பிங்க் பெர்ச்',
        'english_name': 'Japanese Threadfin Bream',
        'freshness_0_5_c': 3.5,
        'freshness_5_15_c': 2.5,
        'freshness_15_25_c': 1.5,
        'min_length_cm': 18,
        'max_length_cm': 35,
        'min_weight_g': 150,
        'max_weight_g': 800,
        'min_price_kg': 300,
        'max_price_kg': 450,
        'image_folders': ['Pink Perch'],
        'freshness_indicators': ['pinkish_hue', 'clear_eyes', 'firm_flesh']
    }
}
        
        # Tamil Nadu coastal fishing locations with real GPS coordinates
        self.tamil_nadu_locations = {
    # Northern Coast
    'Pulicat': {'lat': 13.4167, 'lon': 80.3167, 'district': 'Tiruvallur'},
    'Ennore': {'lat': 13.2167, 'lon': 80.3167, 'district': 'Tiruvallur'},
    'Chennai': {'lat': 13.0827, 'lon': 80.2707, 'district': 'Chennai'},
    'Mamallapuram': {'lat': 12.6208, 'lon': 80.1982, 'district': 'Chengalpattu'},

    # Coromandel Coast
    'Pondicherry': {'lat': 11.9416, 'lon': 79.8083, 'district': 'Puducherry'},
    'Cuddalore': {'lat': 11.7447, 'lon': 79.7689, 'district': 'Cuddalore'},
    'Karaikal': {'lat': 10.9254, 'lon': 79.8380, 'district': 'Karaikal'},
    'Nagapattinam': {'lat': 10.7667, 'lon': 79.8420, 'district': 'Nagapattinam'},
    'Vedaranyam': {'lat': 10.3667, 'lon': 79.85, 'district': 'Nagapattinam'},

    # Palk Strait & Gulf of Mannar
    'Adirampattinam': {'lat': 10.3340, 'lon': 79.3835, 'district': 'Thanjavur'},
    'Pamban': {'lat': 9.2812, 'lon': 79.2095, 'district': 'Ramanathapuram'},
    'Rameswaram': {'lat': 9.2876, 'lon': 79.3129, 'district': 'Ramanathapuram'},
    'Tuticorin': {'lat': 8.8055, 'lon': 78.1554, 'district': 'Thoothukudi'},

    # Southern Coast
    'Kanyakumari': {'lat': 8.0883, 'lon': 77.5385, 'district': 'Kanyakumari'},
    'Colachel': {'lat': 8.1795, 'lon': 77.2520, 'district': 'Kanyakumari'}
}
        
        # Storage conditions that affect freshness
        self.storage_conditions = [
            {'type': 'Ice-cold storage', 'temp_range': (0, 5), 'multiplier': 1.0},
            {'type': 'Refrigerated transport', 'temp_range': (5, 15), 'multiplier': 0.8},
            {'type': 'Ambient coastal temperature', 'temp_range': (20, 30), 'multiplier': 0.4}
        ]

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS points using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = R * c
        
        return distance

    def get_seasonal_temperature(self, catch_date, base_temp=28):
        """Get realistic Tamil Nadu coastal temperature based on season"""
        month = catch_date.month
        
        # Tamil Nadu seasonal patterns
        if month in [12, 1, 2]:  # Winter
            temp_adjustment = np.random.uniform(-4, -1)
        elif month in [3, 4, 5]:  # Summer  
            temp_adjustment = np.random.uniform(3, 8)
        elif month in [6, 7, 8, 9]:  # Monsoon
            temp_adjustment = np.random.uniform(-2, 2)
        else:  # Post-monsoon
            temp_adjustment = np.random.uniform(0, 3)
            
        # Daily variation
        daily_var = np.random.uniform(-2, 3)
        final_temp = base_temp + temp_adjustment + daily_var
        
        return round(max(22, min(38, final_temp)), 1)

    def calculate_freshness(self, species_info, storage_temp, hours_elapsed):
        """Calculate remaining freshness based on species, temperature, and time"""
        # Get base freshness for storage temperature
        if storage_temp <= 5:
            base_freshness_days = species_info['freshness_0_5_c']
        elif storage_temp <= 15:
            base_freshness_days = species_info['freshness_5_15_c']
        else:
            base_freshness_days = species_info['freshness_15_25_c']
        
        # Calculate degradation based on time elapsed
        days_elapsed = hours_elapsed / 24.0
        remaining_freshness = max(0.1, base_freshness_days - days_elapsed)
        
        # Add some realistic variation
        variation = np.random.uniform(-0.2, 0.2)
        remaining_freshness += variation
        
        return max(0.1, remaining_freshness)

    def get_freshness_category(self, freshness_days):
        """Categorize freshness level"""
        if freshness_days >= 3.0:
            return "Very Fresh"
        elif freshness_days >= 2.0:
            return "Fresh"
        elif freshness_days >= 1.0:
            return "Moderately Fresh"
        elif freshness_days >= 0.5:
            return "Needs Quick Sale"
        else:
            return "Immediate Sale Required"

    def generate_single_record(self, record_id: int) -> Dict:
        """Generate a single fish record with realistic data"""
        # Select random species
        species_name = random.choice(list(self.aligned_fish_species.keys()))
        species_info = self.aligned_fish_species[species_name]
        
        # Generate catch date (within last 7 days for freshness relevance)
        catch_date = fake.date_time_between(start_date='-7d', end_date='now')
        
        # Fisher location (coastal Tamil Nadu)
        fisher_location_name = random.choice(list(self.tamil_nadu_locations.keys()))
        fisher_coords = self.tamil_nadu_locations[fisher_location_name]
        
        # Add GPS variation for exact coordinates
        fisher_lat = fisher_coords['lat'] + np.random.uniform(-0.02, 0.02)
        fisher_lon = fisher_coords['lon'] + np.random.uniform(-0.02, 0.02)
        
        # Buyer location (can be same or different city)
        buyer_location_name = random.choice(list(self.tamil_nadu_locations.keys()))
        buyer_coords = self.tamil_nadu_locations[buyer_location_name]
        buyer_lat = buyer_coords['lat'] + np.random.uniform(-0.03, 0.03)
        buyer_lon = buyer_coords['lon'] + np.random.uniform(-0.03, 0.03)
        
        # Area temperature at catch time
        area_temp = self.get_seasonal_temperature(catch_date)
        
        # Storage conditions
        storage_condition = random.choice(self.storage_conditions)
        storage_temp = np.random.uniform(*storage_condition['temp_range'])
        
        # Time since catch (0.5 to 48 hours)
        hours_since_catch = np.random.uniform(0.5, 48.0)
        
        # Fish physical characteristics
        fish_weight = np.random.uniform(species_info['min_weight_g'], species_info['max_weight_g'])
        fish_length = np.random.uniform(species_info['min_length_cm'], species_info['max_length_cm'])
        
        # Calculate freshness
        freshness_days = self.calculate_freshness(species_info, storage_temp, hours_since_catch)
        freshness_category = self.get_freshness_category(freshness_days)
        
        # Calculate market price with freshness and demand factors
        base_price = np.random.uniform(species_info['min_price_kg'], species_info['max_price_kg'])
        
        # Price adjustments
        freshness_factor = max(0.6, freshness_days / species_info['freshness_0_5_c'])
        market_demand = np.random.uniform(0.85, 1.15)  # Market fluctuation
        seasonal_factor = np.random.uniform(0.9, 1.1)   # Seasonal availability
        
        final_price = base_price * freshness_factor * market_demand * seasonal_factor
        
        # Calculate distance between fisher and buyer
        distance_km = self.calculate_distance(fisher_lat, fisher_lon, buyer_lat, buyer_lon)
        
        return {
            'fish_species': species_name,
            'fish_scientific_name': species_info['scientific_name'],
            'fish_local_name': species_info['tamil_name'],
            'fish_english_name': species_info['english_name'],
            'freshness_days': round(freshness_days, 2),
            'freshness_category': freshness_category,
            'latitude': round(fisher_lat, 6),  # Fisher catch location
            'longitude': round(fisher_lon, 6),
            'storage_temp': round(storage_temp, 1),
            'storage_type': storage_condition['type'],
            'area_temp': area_temp,
            'price_per_kg': round(final_price, 2),
            'date_of_catch': catch_date.strftime('%Y-%m-%d %H:%M:%S'),
            'fish_weight': round(fish_weight, 2),
            'fish_length': round(fish_length, 2),
            'buyer_location_lat': round(buyer_lat, 6),
            'buyer_location_lon': round(buyer_lon, 6),
            'fisher_location': fisher_location_name,
            'buyer_location': buyer_location_name,
            'distance_km': round(distance_km, 2),
            'hours_since_catch': round(hours_since_catch, 2),
            'catch_id': f"FISH{record_id:05d}",
            'image_folder_reference': random.choice(species_info['image_folders']),
            'freshness_indicators': ','.join(random.sample(species_info['freshness_indicators'], 
                                                         min(3, len(species_info['freshness_indicators']))))
        }

    def generate_dataset(self, num_records=50000):
        """Generate the complete aligned dataset"""
        print(f"🐟 Generating {num_records:,} Tamil Nadu fish records...")
        print(f"📁 Aligned with {len(self.aligned_fish_species)} species from your image folders")
        
        records = []
        
        for i in range(num_records):
            if (i + 1) % 5000 == 0:
                print(f"Generated {i+1:,}/{num_records:,} records ({((i+1)/num_records)*100:.1f}%)")
            
            record = self.generate_single_record(i)
            records.append(record)
        
        df = pd.DataFrame(records)
        return df

    def analyze_dataset(self, df):
        """Comprehensive dataset analysis"""
        print("\n" + "="*80)
        print("TAMIL NADU FISH DATASET - ANALYSIS REPORT")
        print("="*80)
        
        print(f"📊 Total Records: {len(df):,}")
        print(f"📅 Date Range: {df['date_of_catch'].min()} to {df['date_of_catch'].max()}")
        
        print(f"\n🐟 SPECIES DISTRIBUTION:")
        species_counts = df['fish_species'].value_counts()
        for species, count in species_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {species}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\n🌡️ FRESHNESS CATEGORY DISTRIBUTION:")
        freshness_counts = df['freshness_category'].value_counts()
        for category, count in freshness_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {category}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\n🧊 STORAGE TYPE DISTRIBUTION:")
        storage_counts = df['storage_type'].value_counts()
        for storage_type, count in storage_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {storage_type}: {count:,} records ({percentage:.1f}%)")
        
        print(f"\n💰 PRICE STATISTICS:")
        print(f"  Average Price/kg: ₹{df['price_per_kg'].mean():.2f}")
        print(f"  Price Range: ₹{df['price_per_kg'].min():.2f} - ₹{df['price_per_kg'].max():.2f}")
        print(f"  Median Price/kg: ₹{df['price_per_kg'].median():.2f}")
        
        print(f"\n⏰ FRESHNESS STATISTICS:")
        print(f"  Average Freshness: {df['freshness_days'].mean():.2f} days")
        print(f"  Freshness Range: {df['freshness_days'].min():.2f} - {df['freshness_days'].max():.2f} days")
        print(f"  Median Freshness: {df['freshness_days'].median():.2f} days")
        
        print(f"\n📍 LOCATION STATISTICS:")
        print(f"  Average Distance (Fisher→Buyer): {df['distance_km'].mean():.2f} km")
        print(f"  Maximum Distance: {df['distance_km'].max():.2f} km")
        
        print(f"\n🌡️ TEMPERATURE STATISTICS:")
        print(f"  Average Storage Temperature: {df['storage_temp'].mean():.1f}°C")
        print(f"  Average Area Temperature: {df['area_temp'].mean():.1f}°C")
        
        print(f"\n📂 IMAGE FOLDER ALIGNMENT:")
        folder_counts = df['image_folder_reference'].value_counts()
        print(f"  Total unique image folders referenced: {len(folder_counts)}")
        for folder, count in folder_counts.head(10).items():
            print(f"  {folder}: {count:,} records")

    def save_dataset_with_splits(self, df, base_filename='tamil_nadu_fish_dataset_50k'):
        """Save dataset with train/validation/test splits"""
        
        # Save main dataset
        main_file = f'{base_filename}.csv'
        df.to_csv(main_file, index=False)
        print(f"\n✅ Main dataset saved: {main_file}")
        
        # Create stratified splits to ensure all species are represented
        train_dfs = []
        val_dfs = []
        test_dfs = []
        
        for species in df['fish_species'].unique():
            species_df = df[df['fish_species'] == species].sample(frac=1, random_state=42).reset_index(drop=True)
            
            n = len(species_df)
            train_end = int(0.7 * n)
            val_end = int(0.85 * n)
            
            train_dfs.append(species_df[:train_end])
            val_dfs.append(species_df[train_end:val_end])
            test_dfs.append(species_df[val_end:])
        
        # Combine splits
        train_df = pd.concat(train_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = pd.concat(val_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        test_df = pd.concat(test_dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save splits
        train_file = f'{base_filename}_train.csv'
        val_file = f'{base_filename}_validation.csv'
        test_file = f'{base_filename}_test.csv'
        
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        test_df.to_csv(test_file, index=False)
        
        print(f"✅ Training set: {train_file} ({len(train_df):,} records)")
        print(f"✅ Validation set: {val_file} ({len(val_df):,} records)")
        print(f"✅ Test set: {test_file} ({len(test_df):,} records)")
        
        # Species distribution check
        print(f"\n📊 SPECIES DISTRIBUTION ACROSS SPLITS:")
        print("Species | Train | Val | Test")
        print("-" * 40)
        for species in sorted(df['fish_species'].unique()):
            train_count = len(train_df[train_df['fish_species'] == species])
            val_count = len(val_df[val_df['fish_species'] == species])
            test_count = len(test_df[test_df['fish_species'] == species])
            print(f"{species[:15]:15} | {train_count:5} | {val_count:3} | {test_count:4}")
        
        return df

    def get_image_folder_mapping(self):
        """Return mapping for ML model alignment"""
        mapping = {}
        for species, info in self.aligned_fish_species.items():
            mapping[species] = {
                'scientific_name': info['scientific_name'],
                'tamil_name': info['tamil_name'],
                'image_folders': info['image_folders'],
                'freshness_indicators': info['freshness_indicators']
            }
        return mapping


def main():
    print("🐟 TAMIL NADU FISH DATASET GENERATOR")
    print("="*80)
    print("Creating aligned dataset for ML training with your image folders")
    
    # Initialize generator
    generator = TamilNaduFishDatasetGenerator()
    
    # Show species alignment
    print(f"\n📂 SPECIES-TO-IMAGE-FOLDER ALIGNMENT:")
    mapping = generator.get_image_folder_mapping()
    for species, info in mapping.items():
        print(f"  {species} → {', '.join(info['image_folders'])}")
    
    print(f"\n🎯 DATASET FEATURES:")
    print("  ✅ Species classification labels aligned with image folders")
    print("  ✅ Realistic freshness decay patterns by temperature")
    print("  ✅ Tamil Nadu market prices and locations")
    print("  ✅ GPS coordinates for fisher and buyer locations")
    print("  ✅ Freshness indicators for image model training")
    
    # Generate dataset
    print(f"\n🔄 Generating dataset...")
    dataset = generator.generate_dataset(num_records=50000)
    
    # Analyze dataset
    generator.analyze_dataset(dataset)
    
    # Save dataset with splits
    final_dataset = generator.save_dataset_with_splits(dataset)
    
    print(f"\n🎉 DATASET GENERATION COMPLETED!")
    print("="*80)
    print("✅ Dataset ready for ML training pipeline")
    print("✅ Image folders perfectly aligned with species labels")
    print("✅ Freshness prediction data with realistic decay patterns")
    print("✅ Tamil Nadu market data with GPS coordinates")
    
    # Integration guide
    print(f"\n🔗 ML INTEGRATION WORKFLOW:")
    print("1️⃣  Image Model: Train on your species folders for classification")
    print("2️⃣  Tabular Model: Train on this dataset for freshness prediction")
    print("3️⃣  Combined Pipeline:")
    print("   📸 Upload photo → Classify species → Match dataset row")
    print("   👁️  Analyze image freshness indicators → Adjust prediction")
    print("   💰 Output: Species + Freshness + Price + Location data")
    
    # Sample data preview
    print(f"\n📋 SAMPLE RECORDS:")
    print("="*80)
    sample_cols = ['fish_species', 'fish_local_name', 'freshness_days', 
                   'freshness_category', 'price_per_kg', 'storage_type', 'image_folder_reference']
    sample_data = final_dataset[sample_cols].head(8)
    print(sample_data.to_string(index=False, max_colwidth=20))
    
    print(f"\n📈 NEXT STEPS:")
    print("1. Use this dataset for tabular ML model training")
    print("2. Train image classification model on your species folders:")
    print("   - fish_market, Fish dataset, fresh and non-fresh fish, Rohu")
    print("3. Integrate both models in your Streamlit app")
    print("4. Test with real fish photos for species + freshness prediction")


if __name__ == "__main__":
    main()
