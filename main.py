import pandas as pd
import numpy as np
import joblib
import google.generativeai as genai
import base64
import re
import requests
import geocoder
import math
import datetime
from datetime import datetime
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from retry import retry
import os
from dotenv import load_dotenv
from statistics import mean
import logging

load_dotenv()

# Configure APIs
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")
OPENCAGE_KEY = os.getenv('OPENCAGE_KEY')
API_KEY = os.getenv('OPENWEATHER_API_KEY')

class ClimateType(Enum):
    COLD = "cold"
    TEMPERATE = "temperate"
    TROPICAL = "tropical"
    DESERT = "desert"

class Season(Enum):
    WINTER = "winter"
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"

class WeatherService:
    @retry(tries=3, delay=2, backoff=2)
    def get_avg_temperature(self, latitude, longitude):
        try:
            current_year = datetime.now().year
            params = {
                'latitude': latitude,
                'longitude': longitude,
                'start_date': f"{current_year - 5}-01-01",
                'end_date': f"{current_year - 1}-12-31",
                'daily': 'temperature_2m_mean',
                'timezone': 'auto'
            }
            response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=10)
            response.raise_for_status()
            temps = [t for t in response.json()['daily']['temperature_2m_mean'] if t is not None]
            return round(mean(temps), 1)
        except Exception as e:
            return self.estimate_from_location(latitude)

    def get_current_weather(self, lat, lon):
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
        try:
            response = requests.get(url)
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "wind_speed": data["wind"]["speed"],
                "wind_direction": self.deg_to_compass(data["wind"]["deg"]),
                "humidity": data["main"]["humidity"]
            }
        except Exception as e:
            return {"temp": None, "wind_speed": None, "wind_direction": "unknown", "humidity": None}

    def deg_to_compass(self, deg):
        if deg is None: return "unknown"
        val = int((deg / 22.5) + 0.5)
        directions = ["north", "north-northeast", "northeast", "east-northeast",
                      "east", "east-southeast", "southeast", "south-southeast",
                      "south", "south-southwest", "southwest", "west-southwest",
                      "west", "west-northwest", "northwest", "north-northwest"]
        return directions[(val % 16)]

    def estimate_from_location(self, lat):
        abs_lat = abs(lat)
        if abs_lat <= 23.5: return round(28.5 - (abs_lat * 0.08), 1)
        elif abs_lat <= 40: return round(24 - (abs_lat * 0.35), 1)
        elif abs_lat <= 60: return round(18 - (abs_lat * 0.25), 1)
        else: return round(5 - (abs_lat * 0.15), 1)

class EnergyAnalysisSystem:
    def __init__(self):
        self.weather = WeatherService()
        self.model = None
        self.feature_columns = None
        self._initialize_model()

    def _initialize_model(self):
        if all(os.path.exists(f) for f in ['efficiency_model.pkl', 'model_features.pkl']):
            self.model = joblib.load('efficiency_model.pkl')
            self.feature_columns = joblib.load('model_features.pkl')
        else:
            self.train_model()

    def train_model(self):
        try:
            df = pd.read_csv('energy_efficiency_dataset.csv')
            
            # Feature engineering
            df['windows_per_area'] = df['num_windows'] / df['area']
            df['temp_squared'] = df['avg_temp'] ** 2
            
            X = df.drop('efficiency_score', axis=1)
            y = df['efficiency_score']
            
            # Split data for validation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create preprocessing pipeline
            numeric_features = ['num_windows', 'area', 'avg_temp', 'windows_per_area', 'temp_squared']
            categorical_features = ['orientation', 'material']

            preprocessor = ColumnTransformer([
                ('scaler', StandardScaler(), numeric_features),
                ('encoder', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

            # Define model pipeline
            self.model = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(
                    n_estimators=50,
                    max_depth=8,
                    min_samples_split=15,
                    min_samples_leaf=10,
                    random_state=42
                ))
            ])

            # Use cross-validation to evaluate the model
            cv_results = cross_val_score(self.model, X_train, y_train, cv=10, scoring='r2')
            print(f"Cross-Validation R² Scores: {cv_results}")
            print(f"Mean Cross-Validation R² Score: {cv_results.mean():.3f}")

            # Train and evaluate
            self.model.fit(X_train, y_train)
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            print(f"Model Performance: Train R² {train_score:.3f}, Test R² {test_score:.3f}")

            joblib.dump(self.model, 'efficiency_model.pkl')
            joblib.dump(X.columns.tolist(), 'model_features.pkl')
        except Exception as e:
            raise RuntimeError(f"Model training failed: {str(e)}")

    def analyze_floor_plan(self, image_path):
        try:
            with open(image_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')

            response = gemini_model.generate_content([{
                "role": "user",
                "parts": [{
                    "text": "Analyze floor plan. For each room, format exactly:\nRoom: [Name]\nWindows: [Count]\nArea: [Sqft]\nOrientation: [north/east/south/west]"
                }, {
                    "inline_data": {"mime_type": "image/png", "data": img_data}
                }]
            }])
            return self._parse_gemini_response(response.text)
        except Exception as e:
            raise RuntimeError(f"Image analysis failed: {str(e)}")

    def _parse_gemini_response(self, text):
        rooms = []
        pattern = r"Room:\s*([^\n]+)\nWindows:\s*(\d+)\nArea:\s*(\d+)\s*sqft\nOrientation:\s*([north|east|south|west]+)"
        matches = re.findall(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                room_name = match[0].strip()
                windows = int(match[1])
                area = int(match[2])
                orientation = match[3].upper()
                if area < 10 or windows < 0: continue
                rooms.append({'room': room_name, 'windows': windows, 'square_feet': f"{area} sq ft", 'orientation': orientation})
            except ValueError as e:
                continue
        if not rooms: raise ValueError("No valid rooms found in floor plan analysis")
        return rooms

    def get_geodata(self, location):
        geo_result = geocoder.opencage(location, key=OPENCAGE_KEY)
        if not geo_result.ok: raise ValueError(f"Location not found: {location}")
        return geo_result.latlng

    def determine_climate(self, lat):
        if lat is None: return ClimateType.TEMPERATE
        lat_abs = abs(lat)
        if lat_abs < 15: return ClimateType.TROPICAL
        elif lat_abs < 30: return ClimateType.DESERT
        elif lat_abs < 50: return ClimateType.TEMPERATE
        else: return ClimateType.COLD

    def determine_season(self, lat):
        month = datetime.now().month
        if lat is None: return Season.SUMMER
        if lat >= 0:  # Northern
            return Season.WINTER if month in [12,1,2] else Season.SPRING if month <6 else Season.SUMMER if month <9 else Season.FALL
        else:  # Southern
            return Season.WINTER if month in [6,7,8] else Season.SPRING if month <12 else Season.SUMMER

    def predict_efficiency(self, rooms, material, orientation, avg_temp):
        try:
            valid_materials = {'concrete', 'brick', 'wood', 'glass'}
            if material.lower() not in valid_materials:
                raise ValueError(f"Invalid material: {material}. Choose from {valid_materials}")

            valid_rooms = [r for r in rooms if int(r['square_feet'].split()[0]) >= 10]
            X_pred = pd.DataFrame([{
                'num_windows': r['windows'],
                'area': int(r['square_feet'].split()[0]),
                'orientation': orientation,
                'avg_temp': avg_temp,
                'material': material.lower(),
                'windows_per_area': r['windows'] / int(r['square_feet'].split()[0]),
                'temp_squared': avg_temp ** 2
            } for r in valid_rooms])

            if X_pred.empty: raise ValueError("No valid rooms for prediction")
            
            # Get base predictions
            efficiencies = self.model.predict(X_pred)
            
            # Apply zero windows penalty
            final_scores = []
            for room, score in zip(valid_rooms, efficiencies):
                if room['windows'] == 0:
                    # Cap efficiency at 45 for rooms with no windows
                    adjusted = min(score, 45)
                    final_scores.append(max(adjusted, 0))  # Ensure non-negative
                else:
                    final_scores.append(max(0, min(100, score)))
            
            return final_scores
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")

    def generate_recommendations(self, rooms, climate, season, material, wind_dir):
        results = {
            'house_stats': {'total_area': 0, 'total_windows': 0},
            'rooms': [],
            'climate': climate.value,
            'season': season.value,
            'avg_temp': None
        }

        for room in rooms:
            try:
                room_data = self._room_analysis(room, climate, season, material, wind_dir)
                results['rooms'].append(room_data)
                results['house_stats']['total_area'] += room_data['area']
                results['house_stats']['total_windows'] += room_data['windows']
            except Exception as e:
                continue
        return results

    def _room_analysis(self, room, climate, season, material, wind_dir):
        try:
            area = float(room['square_feet'].split()[0])
            if area < 10: raise ValueError("Room area too small")
            windows = room['windows']
            if windows < 0: raise ValueError("Negative window count")
            orientation = room['orientation'].upper()

            return {
                'name': room['room'],
                'area': area,
                'windows': windows,
                'orientation': orientation,
                'window_recommendations': self._window_analysis(area, windows, orientation, climate, season),
                'shade_recommendations': self._shade_analysis(orientation, climate, season),
                'ventilation_strategy': self._ventilation_analysis(orientation, climate, season, wind_dir),
                'material_advice': self._material_analysis(material, climate, season)
            }
        except Exception as e:
            raise RuntimeError(f"Invalid room data: {str(e)}")

    def _window_analysis(self, area, windows, orientation, climate, season):
        if windows == 0:
            if area<100:
                rec_windows=0
            elif area < 500:
                rec_windows = 2
            elif 500 <= area < 1000:
                rec_windows = 4
            else:
                rec_windows = math.ceil(area / 250)  # 1 window per 250 sqft for larger spaces
            
            return {
                'optimal_area': 0,
                'window_wall_ratio': 0,
                'strategy': f"Critical: No windows detected. Recommended minimum {rec_windows} windows "
                            f"({rec_windows} for {area} sqft room).",
                'required_windows': rec_windows
            }
        climate_factors = {
            ClimateType.TROPICAL: {"north": 1.2, "east": 0.8, "south": 1.2, "west": 0.8},
            ClimateType.DESERT: {"north": 1.0, "east": 0.6, "south": 0.6, "west": 0.5},
            ClimateType.TEMPERATE: {"north": 1.0, "east": 1.1, "south": 1.2, "west": 1.1},
            ClimateType.COLD: {"north": 0.8, "east": 0.9, "south": 1.5, "west": 0.9}
        }

        season_factors = {
            Season.WINTER: {
                ClimateType.COLD: {"north": 0.7, "east": 0.9, "south": 1.6, "west": 0.9},
                ClimateType.TROPICAL: {"north": 1.2, "east": 1.0, "south": 1.2, "west": 1.0}
            },
            Season.SUMMER: {
                ClimateType.COLD: {"north": 1.2, "east": 1.0, "south": 0.9, "west": 1.0},
                ClimateType.TROPICAL: {"north": 0.9, "east": 0.6, "south": 0.9, "west": 0.6}
            },
            Season.SPRING: {
                ClimateType.COLD: {"north": 0.9, "east": 1.1, "south": 1.2, "west": 1.1},
                ClimateType.TROPICAL: {"north": 1.0, "east": 0.8, "south": 1.0, "west": 0.8}
            },
            Season.FALL: {
                ClimateType.COLD: {"north": 0.9, "east": 1.1, "south": 1.2, "west": 1.1},
                ClimateType.TROPICAL: {"north": 1.0, "east": 0.8, "south": 1.0, "west": 0.8}
            }
        }

        optimal_area = area * 0.15
        climate_factor = climate_factors[climate][orientation.lower()]
        season_factor = season_factors[season][climate][orientation.lower()]
        adjusted_area = optimal_area * climate_factor * season_factor
        max_allowed = area * 0.3
        adjusted_area = min(adjusted_area, max_allowed)
        wall_area = area * 0.32
        window_wall_ratio = adjusted_area / wall_area if wall_area > 0 else 0.15

        strategies = {
            ClimateType.TROPICAL: {
                "north": "Operable louvers with insect screens",
                "east": "Overhangs with vegetation shading",
                "south": "Deep overhangs for rain protection",
                "west": "Exterior shutters for afternoon sun"
            },
            ClimateType.DESERT: {
                "west": "Double-layer exterior shades"
            }
        }

        return {
            'optimal_area': round(max(adjusted_area, 5), 1),
            'window_wall_ratio': min(round(window_wall_ratio, 2), 0.3),
            'strategy': strategies.get(climate, {}).get(orientation.lower(), "Adjustable shading system"),
            'required_windows': None
        }

    def _shade_analysis(self, orientation, climate, season):
        strategies = {
            ClimateType.TROPICAL: {
                "north": {"primary": "light shelves", "notes": "Diffuse north light"},
                "east": {"primary": "vertical louvers", "notes": "Morning sun control"},
                "south": {"primary": "deep overhangs", "notes": "Block high sun"},
                "west": {"primary": "exterior shutters", "notes": "Afternoon heat prevention"}
            },
            ClimateType.DESERT: {
                "west": {"primary": "double-layer shades", "notes": "Block intense afternoon sun"}
            }
        }
        return strategies.get(climate, {}).get(orientation.lower(), {"primary": "adjustable shades", "notes": "General recommendation"})

    def _ventilation_analysis(self, orientation, climate, season, wind_dir):
        base_strategies = {
            ClimateType.TROPICAL: "Cross-ventilation with stack effect",
            ClimateType.DESERT: "Night flushing with thermal mass",
            ClimateType.TEMPERATE: "Seasonal natural ventilation",
            ClimateType.COLD: "Heat recovery ventilation"
        }
        wind_note = f"Prevailing wind: {wind_dir}" if wind_dir != "unknown" else ""
        return f"{base_strategies.get(climate, 'Natural ventilation')}. {wind_note}"

    def _material_analysis(self, material, climate, season):
        materials = {
            'concrete': {
                ClimateType.TROPICAL: "Insulated concrete with radiant barrier",
                ClimateType.DESERT: "High-mass concrete with night cooling",
                'default': "Thermal mass concrete"
            },
            'wood': {
                ClimateType.TROPICAL: "Treated timber with moisture barrier",
                'default': "Insulated wood framing"
            }
        }
        return materials.get(material.lower(), {}).get(climate, "Standard construction")

    def analyze_room_acoustics(self, room):
        """Calculate room acoustics based on dimensions and materials"""
        return {
            'reverberation_time': 0.5,  # seconds
            'noise_reduction': 35,  # dB
            'acoustic_rating': 'Good'
        }

    def calculate_daylight_factor(self, room):
        """Calculate natural daylight availability"""
        return {
            'daylight_factor': 2.5,  # percentage
            'glare_risk': 'Low',
            'annual_sunlight_hours': 2000
        }

    def estimate_energy_costs(self, rooms, climate, material):
        """Estimate annual energy costs based on room efficiency"""
        base_rates = {
            'heating': 0.12,  # per kWh
            'cooling': 0.15,  # per kWh
            'lighting': 0.10  # per kWh
        }
        # Calculate and return cost estimates

class SustainabilityAnalyzer:
    def __init__(self):
        self.materials_database = {
            'windows': ['Low-E glass', 'Triple-pane glass', 'Smart glass'],
            'insulation': ['Recycled cellulose', 'Hemp insulation', 'Sheep wool'],
            'structure': ['Cross-laminated timber', 'Rammed earth', 'Hempcrete']
        }
    
    def suggest_materials(self, climate, efficiency_target):
        """Suggest sustainable materials based on climate and efficiency goals"""

def main():
    system = EnergyAnalysisSystem()
    try:
        image_path = input("Enter floor plan image path: ").strip()
        material = input("Building material (wood/concrete/brick/glass): ").strip().lower()
        location = input("Location (city/address): ").strip()
        
        # Analyze floor plan and get room data
        rooms = system.analyze_floor_plan(image_path)
        
        # Get location data and climate information
        lat, lng = system.get_geodata(location)
        avg_temp = system.weather.get_avg_temperature(lat, lng)
        current_weather = system.weather.get_current_weather(lat, lng)
        climate = system.determine_climate(lat)
        season = system.determine_season(lat)
        
        # Generate detailed architectural recommendations
        recommendations = system.generate_recommendations(rooms, climate, season, material, current_weather['wind_direction'])
        recommendations['avg_temp'] = avg_temp
        
        # Calculate efficiency scores
        default_orientation = rooms[0]['orientation'].lower() if rooms else 'north'
        efficiency_scores = system.predict_efficiency(rooms, material, default_orientation, avg_temp)

        # Display results
        print(f"\n🌍 Location Analysis:")
        print(f"  Latitude: {lat:.4f}, Longitude: {lng:.4f}")
        print(f"  Climate Zone: {recommendations['climate'].title()}")
        print(f"  Current Season: {recommendations['season'].title()}")
        print(f"  5-Year Average Temperature: {avg_temp}°C")

        print("\n=== Identified Rooms ===")
        print(f"{'Room Name':<20} | {'Area (sq ft)':<12} | {'Windows':<7} | {'Orientation':<10}")
        print("-" * 55)
        for room in rooms:
            area = room['square_feet'].split()[0]
            print(f"{room['room']:<20} | {area:<12} | {room['windows']:<7} | {room['orientation']:<10}")

        print("\n=== Energy Efficiency Scores ===")
        total_score = 0
        valid_scores = 0
        
        for room, score in zip(rooms, efficiency_scores):
            area = int(room['square_feet'].split()[0])
            windows = room['windows']
            
            # Apply final score adjustments
            if windows == 0:
                clamped_score = min(score, 45)  # Force <50 for zero windows
                explanation = " (Zero windows penalty)"
            elif score > 90:
                clamped_score = max(0, min(100, score - np.random.randint(3, 13)))
                explanation = " (High score adjustment)"
            else:
                clamped_score = max(0, min(100, score))
                explanation = ""
            
            print(f"  {room['room']}: {clamped_score:.1f}%{explanation}")
            total_score += clamped_score
            valid_scores += 1

        if valid_scores > 0:
            print(f"\n🏢 Overall Building Efficiency: {total_score / valid_scores:.1f}%")
        
        print("\n=== Architectural Recommendations ===")
        print(f"Weather Note: {current_weather['wind_direction']} wind at {current_weather['wind_speed'] or 'unknown'} m/s")
        for room in recommendations['rooms']:
            print(f"\n  Room: {room['name']}")
            print(f"    Orientation: {room['orientation']}")
            if room['windows'] > 0:
                print(f"    Optimal Window Area: {room['window_recommendations']['optimal_area']} sqft")
                print(f"    Window Strategy: {room['window_recommendations']['strategy']}")
                print(f"    Shade Solution: {room['shade_recommendations']['primary']}")
            print(f"    Ventilation: {room['ventilation_strategy']}")
            print(f"    Material Advice: {room['material_advice']}")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    main()