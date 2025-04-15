# app.py
import time
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import io
import base64
import time
from main import EnergyAnalysisSystem

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit
app.secret_key = os.urandom(24)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

system = EnergyAnalysisSystem()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'floor_plan' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['floor_plan']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Add artificial delay to show loading screen (remove in production)
            time.sleep(2)  # Simulates processing time
            
            # Convert image to base64
            with open(filepath, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Process form data
            material = request.form['material'].lower()
            location = request.form['location']
            
            # Add artificial delay (remove in production)
            time.sleep(2)
            
            # Analyze floor plan
            rooms = system.analyze_floor_plan(filepath)
            
            # Get geodata and climate info
            lat, lng = system.get_geodata(location)
            avg_temp = system.weather.get_avg_temperature(lat, lng)
            current_weather = system.weather.get_current_weather(lat, lng)
            climate = system.determine_climate(lat)
            season = system.determine_season(lat)
            
            # Add artificial delay (remove in production)
            time.sleep(1)
            
            # Generate recommendations with material advice
            recommendations = system.generate_recommendations(
                rooms, climate, season, material, 
                current_weather['wind_direction']
            )
            recommendations['avg_temp'] = avg_temp
            recommendations['material'] = material  # Add material info
            
            # Calculate efficiency scores
            default_orientation = rooms[0]['orientation'].lower() if rooms else 'north'
            efficiency_scores = system.predict_efficiency(
                rooms, material, default_orientation, avg_temp
            )

            # Prepare results data with material advice
            results = {
                'image': encoded_image,
                'location': {
                    'lat': lat,
                    'lng': lng,
                    'climate': climate.value,
                    'season': season.value,
                    'avg_temp': avg_temp,
                    'material': material  # Add material info
                },
                'rooms': [],
                'efficiency': {
                    'scores': [],
                    'average': 0
                },
                'recommendations': recommendations
            }

            # Update room data to include material advice
            total_score = 0
            valid_scores = 0
            for room, score in zip(rooms, efficiency_scores):
                clamped_score = max(0, min(100, score - np.random.randint(3, 13)))
                material_advice = system._material_analysis(material, climate, season)
                results['rooms'].append({
                    'name': room['room'],
                    'area': room['square_feet'],
                    'windows': room['windows'],
                    'orientation': room['orientation'],
                    'score': clamped_score,
                    'material_advice': material_advice  # Add material advice
                })
                total_score += clamped_score
                valid_scores += 1

            if valid_scores > 0:
                results['efficiency']['average'] = total_score / valid_scores

            return render_template('results.html', results=results)
    
    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)