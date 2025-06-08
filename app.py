from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import torch
from torchvision import models, transforms
from torchvision.models import DenseNet121_Weights
from PIL import Image
import os
import pandas as pd
import base64
from io import BytesIO
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
from bson import ObjectId
from flask_cors import CORS


app = Flask(__name__)
app.secret_key = '123'  # Change this to a secure secret key
CORS(app)

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['soil_classifier']
users_collection = db['users']

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, user_data):
        self.id = str(user_data['_id'])
        self.email = user_data['email']
        self.password_hash = user_data['password']

    @staticmethod
    def get(user_id):
        try:
            user_data = users_collection.find_one({'_id': ObjectId(user_id)})
            return User(user_data) if user_data else None
        except:
            return None

@login_manager.user_loader
def load_user(user_id):
    return User.get(user_id)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        user_data = users_collection.find_one({'email': email})
        
        if user_data and check_password_hash(user_data['password'], password):
            user = User(user_data)
            login_user(user)
            print("Login successful, redirecting to home page")  # Debugging
            return redirect(url_for('home'))
        
        flash('Invalid email or password', 'error')
        print("Login failed")  # Debugging
        return redirect(url_for('login'))
    
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        if users_collection.find_one({'email': email}):
            flash('Email already registered')
            return redirect(url_for('register'))
        
        hashed_password = generate_password_hash(password)
        users_collection.insert_one({
            'email': email,
            'password': hashed_password
        })
        
        flash('Registration successful! Please login.')
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route("/")
@login_required
def home():
    print(f"Current user: {current_user}")  # Debugging
    return render_template("index.html")


@app.route('/analyze')
def analyze():
    return render_template('analyze.html')

@app.route('/blogs')
def blogs():
    return render_template('blogs.html')

CROP_DATA = [
    # Maharashtra
    {"crop": "Rice", "soil": "Clay", "season": "Kharif", "region": "Maharashtra", "months": ["June", "July", "August"]},
    {"crop": "Wheat", "soil": "Loam", "season": "Rabi", "region": "Maharashtra", "months": ["November", "December", "January"]},
    {"crop": "Maize", "soil": "Sandy", "season": "Zaid", "region": "Maharashtra", "months": ["March", "April", "May"]},
    {"crop": "Sorghum (Jowar)", "soil": "Black", "season": "Rabi", "region": "Maharashtra", "months": ["October", "November"]},
    {"crop": "Tur (Arhar)", "soil": "Loamy", "season": "Kharif", "region": "Maharashtra", "months": ["June", "July", "August"]},

    # Gujarat
    {"crop": "Bajra", "soil": "Black", "season": "Kharif", "region": "Gujarat", "months": ["July", "August", "September"]},
    {"crop": "Groundnut", "soil": "Sandy Loam", "season": "Kharif", "region": "Gujarat", "months": ["June", "July"]},
    {"crop": "Cotton", "soil": "Alluvial", "season": "Kharif", "region": "Gujarat", "months": ["June", "July", "August"]},
    {"crop": "Wheat", "soil": "Loamy", "season": "Rabi", "region": "Gujarat", "months": ["November", "December", "January"]},

    # Punjab
    {"crop": "Wheat", "soil": "Loamy", "season": "Rabi", "region": "Punjab", "months": ["November", "December", "January"]},
    {"crop": "Rice", "soil": "Clay", "season": "Kharif", "region": "Punjab", "months": ["June", "July", "August"]},
    {"crop": "Sugarcane", "soil": "Loamy", "season": "Annual", "region": "Punjab", "months": ["February", "March"]},
    {"crop": "Maize", "soil": "Alluvial", "season": "Zaid", "region": "Punjab", "months": ["March", "April"]},

    # Tamil Nadu
    {"crop": "Ragi", "soil": "Red Loam", "season": "Kharif", "region": "Tamil Nadu", "months": ["June", "July", "August"]},
    {"crop": "Sugarcane", "soil": "Loamy", "season": "Annual", "region": "Tamil Nadu", "months": ["January", "February"]},
    {"crop": "Paddy", "soil": "Clay Loam", "season": "Rabi", "region": "Tamil Nadu", "months": ["December", "January", "February"]},
    {"crop": "Chillies", "soil": "Sandy Loam", "season": "Zaid", "region": "Tamil Nadu", "months": ["April", "May", "June"]},

    # West Bengal
    {"crop": "Jute", "soil": "Alluvial", "season": "Kharif", "region": "West Bengal", "months": ["March", "April"]},
    {"crop": "Potato", "soil": "Loamy", "season": "Rabi", "region": "West Bengal", "months": ["November", "December", "January"]},
    {"crop": "Rice", "soil": "Clay", "season": "Kharif", "region": "West Bengal", "months": ["June", "July", "August"]},

    # Uttar Pradesh
    {"crop": "Wheat", "soil": "Alluvial", "season": "Rabi", "region": "Uttar Pradesh", "months": ["November", "December", "January"]},
    {"crop": "Pulses", "soil": "Sandy Loam", "season": "Zaid", "region": "Uttar Pradesh", "months": ["April", "May"]},
    {"crop": "Sugarcane", "soil": "Loamy", "season": "Annual", "region": "Uttar Pradesh", "months": ["February", "March"]},

    # Karnataka
    {"crop": "Finger Millet", "soil": "Red Soil", "season": "Kharif", "region": "Karnataka", "months": ["June", "July", "August"]},
    {"crop": "Turmeric", "soil": "Loamy", "season": "Zaid", "region": "Karnataka", "months": ["April", "May"]},
    {"crop": "Sunflower", "soil": "Red Loam", "season": "Rabi", "region": "Karnataka", "months": ["October", "November", "December"]},

    # Rajasthan
    {"crop": "Pearl Millet", "soil": "Sandy", "season": "Kharif", "region": "Rajasthan", "months": ["June", "July"]},
    {"crop": "Barley", "soil": "Sandy Loam", "season": "Rabi", "region": "Rajasthan", "months": ["November", "December"]},
    {"crop": "Gram", "soil": "Loamy", "season": "Rabi", "region": "Rajasthan", "months": ["November", "December", "January"]},

    # Bihar
    {"crop": "Maize", "soil": "Sandy Loam", "season": "Zaid", "region": "Bihar", "months": ["March", "April", "May"]},
    {"crop": "Paddy", "soil": "Alluvial", "season": "Kharif", "region": "Bihar", "months": ["June", "July"]},
    {"crop": "Lentils", "soil": "Loamy", "season": "Rabi", "region": "Bihar", "months": ["November", "December"]},
]


@app.route('/calender')
def index():
    return render_template('calender.html')

@app.route('/crop-planner', methods=['POST'])
def crop_planner():
    data = request.json
    soil = data.get('soilType', '').lower()
    season = data.get('season', '')
    location = data.get('location', '').lower()

    matches = []
    for crop in CROP_DATA:
        if soil in crop["soil"].lower() and season == crop["season"] and location in crop["region"].lower():
            matches.append({"crop": crop["crop"], "months": crop["months"]})

    if not matches:
        matches.append({"crop": "No ideal match found. Try general crops like pulses or vegetables.", "months": []})

    return jsonify({"suggestions": matches})

@app.route('/set_language/<lang>')
def set_language(lang):
    resp = redirect(url_for('home'))
    resp.set_cookie('lang', lang)
    return resp


# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_SIZE = 224
NUM_CLASSES = 8

# Initialize model with updated weights parameter
model = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(model.classifier.in_features, 512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.4),
    torch.nn.Linear(512, NUM_CLASSES)
)

# Load model weights with weights_only=True for security
model.load_state_dict(
    torch.load(
        './Models/final_fine_tuned_model.pth',
        map_location=device,
        weights_only=True
    )
)
model = model.to(device)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_names = ['Alluvial Soil', 'Black Soil', 'Cinder Soil', 'Clay Soil', 
               'Laterite Soil', 'Peat Soil', 'Red Soil', 'Yellow Soil']

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = outputs.max(1)
    
    return predicted_class.item()

def validate_soil_types(images):
    """
    Validates that all uploaded images are of the same soil type.
    Returns (is_valid, soil_type, error_message)
    """
    if not images:
        return False, None, "No images uploaded"
    
    predictions = []
    for image in images:
        predicted_class = predict_image(image)
        predictions.append(predicted_class)
    
    # Check if all predictions are the same
    if len(set(predictions)) != 1:
        return False, None, "Different soil types detected. Please upload images of the same soil type."
    
    return True, class_names[predictions[0]], None

def load_mapping_table():
    return pd.read_csv('./soil_crop_mapping.csv')


def get_soil_explanation(soil_type):
    explanations = {
        "Alluvial Soil": (
            "Alluvial soils are formed by the deposition of river sediments and are among the most fertile soils found across the Indo-Gangetic plains and deltas of major rivers.\n"
            "They are rich in humus and essential nutrients like potash and phosphoric acid, and they exhibit excellent water retention and drainage balance.\n"
            "These properties make alluvial soils ideal for crops requiring deep root penetration and consistent moisture, such as rice and wheat. Maize, sugarcane, and pulses also perform well here.\n"
            "Due to their fertility, these soils support multiple cropping cycles and are critical to India's food grain output.\n"
        ),
        "Black Soil": (
            "Black soil, also known as Regur soil, originates from basaltic lava and is predominantly found in the Deccan plateau, covering states like Maharashtra, Madhya Pradesh, and Gujarat.\n"
            "This soil is clayey, deep, and moisture-retentive, making it ideal for dryland farming. It is rich in calcium carbonate, magnesium, and potash but deficient in nitrogen and phosphorus.\n"
            "Cotton is the most suitable crop, earning the soil its nickname 'black cotton soil.' Other suitable crops include sorghum, soybean, groundnut, and pulses.\n"
            "Its ability to swell and shrink with moisture helps retain nutrients, although it can be challenging to manage during the rainy season due to cracking.\n"
        ),
        "Cinder Soil": (
            "Cinder soil is formed from volcanic ash and is typically found near dormant or active volcanic regions. It has a loose, porous structure that promotes excellent aeration and drainage.\n"
            "Though low in organic content and essential nutrients, it warms quickly and supports crops that are resilient to nutrient stress and water variability.\n"
            "Drought-tolerant crops such as maize and sorghum thrive in cinder soils. Additionally, pineapple cultivation benefits due to the well-aerated conditions that promote healthy root development.\n"
            "To increase productivity, this soil requires regular addition of compost and organic fertilizers.\n"
        ),
        "Clay Soil": (
            "Clay soils have fine particles and a compact structure, which allows them to retain water and nutrients effectively but makes them prone to waterlogging and poor aeration.\n"
            "Common in river basins and low-lying areas, they are ideal for crops that can tolerate excess water, such as rice and sugarcane.\n"
            "These soils are also rich in potassium and phosphorus, which benefit tuber crops and vegetables under controlled irrigation.\n"
            "To enhance productivity, techniques like raised bed planting, organic mulching, and proper drainage systems are often employed.\n"
        ),
        "Laterite Soil": (
            "Laterite soils are formed in regions with high rainfall and temperature, leading to intense leaching and a high concentration of iron and aluminum oxides.\n"
            "These soils are acidic, low in fertility, and common in states like Kerala, Karnataka, and parts of Odisha.\n"
            "They support perennial crops like tea, coffee, cashew nuts, and rubber, which are adapted to acidic and well-drained conditions.\n"
            "With regular liming and organic enrichment, the productivity of laterite soils can be significantly improved.\n"
        ),
        "Peat Soil": (
            "Peat soils are found in marshy and waterlogged regions and are characterized by their dark color and high organic matter content, often consisting of partially decomposed plant residues.\n"
            "They are highly acidic and retain water extremely well, creating ideal conditions for root crops like potatoes and carrots.\n"
            "These soils are also used for growing berries and other moisture-loving crops in temperate climates.\n"
            "To maximize their potential, careful drainage management and periodic nutrient replenishment are essential due to their poor mineral nutrient availability.\n"
        ),
        "Red Soil": (
            "Red soils are formed by the weathering of igneous and metamorphic rocks and are rich in iron oxides, giving them their characteristic red color.\n"
            "They are generally poor in nitrogen, phosphorus, and organic matter, yet well-drained and aerated.\n"
            "These conditions make them suitable for hardy, drought-resistant crops such as millets, pulses, groundnuts, and cotton.\n"
            "Red soils are prevalent in Tamil Nadu, parts of Karnataka, Andhra Pradesh, and Chhattisgarh, where traditional dryland farming practices are commonly used.\n"
            "Incorporation of green manure and legumes helps in improving fertility and restoring nutrient balance.\n"
        ),
        "Yellow Soil": (
            "Yellow soils are similar to red soils but have a lighter hue due to a lower concentration of iron oxide. They typically form in regions with moderate rainfall and good drainage.\n"
            "Though they lack sufficient nitrogen and phosphorus, they are moderately fertile and suitable for low-input farming systems.\n"
            "Groundnuts, pulses, cassava, and maize are well-suited to these soils because of their adaptability and low nutrient demand.\n"
            "Farmers often use crop rotation and organic composting to maintain soil health and boost yields sustainably.\n"
        )
    }
    return explanations.get(soil_type, "No explanation available.")


@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'})
    
    files = request.files.getlist('files[]')
    if not files or files[0].filename == '':
        return jsonify({'error': 'No files selected'})
    
    try:
        # Validate all files are images
        images = []
        for file in files:
            if not file.content_type.startswith('image/'):
                return jsonify({'error': 'Invalid file type. Please upload only images.'})
            image = Image.open(file).convert('RGB')
            images.append(image)
        
        # Validate soil types
        is_valid, soil_type, error_message = validate_soil_types(images)
        if not is_valid:
            return jsonify({'error': error_message})
        
        # Get crop recommendations
        mapping_df = load_mapping_table()
        soil_data = mapping_df[mapping_df['Soil Type'] == soil_type]
        
        if not soil_data.empty:
            nutrients = soil_data.iloc[0]
            
            # Convert all images to base64
            image_strings = []
            for img in images:
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                image_strings.append(f'data:image/jpeg;base64,{img_str}')
            
            response = {
                'success': True,
                'soil_type': soil_type,
                'nutrients': {
                    'Nitrogen': nutrients['Nitrogen (N)'],
                    'Phosphorus': nutrients['Phosphorus (P)'],
                    'Potassium': nutrients['Potassium (K)'],
                    'Calcium': nutrients['Calcium (Ca)'],
                    'Magnesium': nutrients['Magnesium (Mg)'],
                    'Iron': nutrients['Iron (Fe)'],
                    'Organic Matter': nutrients['Organic Matter']
                },
                'recommended_crops': nutrients['Recommended Crops'],
                'explanation': get_soil_explanation(soil_type),
                'images': image_strings
            }
        else:
            response = {
                'success': False,
                'error': 'Soil type not found in database'
            }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
