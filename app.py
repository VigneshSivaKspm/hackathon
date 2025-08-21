import os
import json
import firebase_admin
from flask import Flask
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, jsonify, flash, send_from_directory
from flask_session import Session
from firebase_admin import auth, credentials, initialize_app, db as firebase_db
from dotenv import load_dotenv
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib
import smtplib
from email.mime.text import MIMEText
import traceback
from werkzeug.exceptions import NotFound
from babel.numbers import format_number as babel_format_number


# Load environment variables
load_dotenv()

app = Flask(__name__, static_folder='static')
app.secret_key = os.getenv('SECRET_KEY')

# Enhanced session configuration
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR='./flask_session',
    SESSION_COOKIE_NAME='disaster_mgmt_session',
    PERMANENT_SESSION_LIFETIME=timedelta(hours=12),
    SESSION_COOKIE_SECURE=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    SESSION_REFRESH_EACH_REQUEST=True
)

Session(app)

# Firebase configuration and initialization
FIREBASE_FRONTEND_CONFIG = {
    'apiKey': os.getenv('FIREBASE_API_KEY'),
    'authDomain': os.getenv('FIREBASE_AUTH_DOMAIN'),
    'projectId': os.getenv('FIREBASE_PROJECT_ID'),
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET'),
    'messagingSenderId': os.getenv('FIREBASE_MESSAGING_SENDER_ID'),
    'appId': os.getenv('FIREBASE_APP_ID'),
    'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
}

try:
    service_account_info = json.loads(os.getenv('FIREBASE_SERVICE_ACCOUNT'))
    cred = credentials.Certificate(service_account_info)
    firebase_app = initialize_app(cred, {
        'databaseURL': os.getenv('FIREBASE_DATABASE_URL')
    })
    print("🔥 Firebase initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize Firebase: {str(e)}")
    raise RuntimeError(f"Firebase initialization failed: {str(e)}")

# Constants and AI Model Manager
MODEL_DIR = 'ai_models'
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATHS = {
    'medical': os.path.join(MODEL_DIR, 'medical_predictor.pkl'),
    'food': os.path.join(MODEL_DIR, 'food_predictor.pkl'),
    'shelter': os.path.join(MODEL_DIR, 'shelter_predictor.pkl')
}
VALID_DISASTER_TYPES = {'earthquake', 'flood', 'hurricane', 'fire', 'tsunami', 'tornado', 'drought'}
VALID_POPULATION_DENSITIES = {'low', 'medium', 'high', 'urban', 'rural'}

class AIModelManager:
    def __init__(self):
        self.models = {'medical': None, 'food': None, 'shelter': None}
        self.preprocessor = None
        self._initialize_models()
    
    def _initialize_models(self):
        try:
            self.models['medical'] = joblib.load(MODEL_PATHS['medical'])
            self.models['food'] = joblib.load(MODEL_PATHS['food'])
            self.models['shelter'] = joblib.load(MODEL_PATHS['shelter'])
            print("🤖 AI models loaded successfully")
        except Exception as e:
            print(f"🔄 Creating and training new AI models: {e}")
            self._train_new_models()
    
    def _train_new_models(self):
        training_data = {
            'disaster_type': ['earthquake', 'earthquake', 'flood', 'flood', 'hurricane',
                              'hurricane', 'fire', 'fire', 'tsunami', 'tsunami', 'tornado',
                              'drought', 'earthquake', 'flood'],
            'severity': [3, 7, 4, 8, 5, 9, 3, 6, 6, 9, 5, 4, 6, 5],
            'population_density': ['medium', 'high', 'medium', 'urban', 'high',
                                   'urban', 'low', 'medium', 'medium', 'high',
                                   'rural', 'low', 'urban', 'rural'],
            'medical_kits': [300, 800, 450, 950, 600, 1100, 250, 700, 750, 1200, 400, 300, 850, 500],
            'food_packets': [700, 1800, 900, 2200, 1200, 2500, 600, 1500, 1600, 2800, 1000, 800, 2000, 1100],
            'shelter_capacity': [200, 500, 300, 600, 400, 700, 150, 450, 500, 800, 300, 200, 550, 350]
        }
        df = pd.DataFrame(training_data)
        categorical_features = ['disaster_type', 'population_density']
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        X = df[['disaster_type', 'severity', 'population_density']]
        y_medical = df['medical_kits']
        y_food = df['food_packets']
        y_shelter = df['shelter_capacity']
        model_pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_depth=12,
                min_samples_leaf=3,
                n_jobs=-1
            ))
        ])
        self.models['medical'] = model_pipeline.fit(X, y_medical)
        joblib.dump(self.models['medical'], MODEL_PATHS['medical'])
        self.models['food'] = model_pipeline.fit(X, y_food)
        joblib.dump(self.models['food'], MODEL_PATHS['food'])
        self.models['shelter'] = model_pipeline.fit(X, y_shelter)
        joblib.dump(self.models['shelter'], MODEL_PATHS['shelter'])
        print("✅ New AI models created and trained with enhanced features.")
    
    def validate_input(self, disaster_type, severity, population_density):
        if disaster_type.lower() not in VALID_DISASTER_TYPES:
            raise ValueError(f"Invalid disaster type. Must be one of: {VALID_DISASTER_TYPES}")
        if population_density.lower() not in VALID_POPULATION_DENSITIES:
            raise ValueError(f"Invalid population density. Must be one of: {VALID_POPULATION_DENSITIES}")
        try:
            severity = float(severity)
            if not (1 <= severity <= 10):
                raise ValueError("Severity must be between 1 and 10")
        except ValueError:
            raise ValueError("Severity must be a numeric value")
        return disaster_type.lower(), severity, population_density.lower()
    
    def predict(self, disaster_type, severity, population_density):
        try:
            disaster_type, severity, population_density = self.validate_input(
                disaster_type, severity, population_density
            )
            input_data = pd.DataFrame({
                'disaster_type': [disaster_type],
                'severity': [severity],
                'population_density': [population_density]
            })
            results = {
                'medical_kits': max(0, int(self.models['medical'].predict(input_data)[0])),
                'food_packets': max(0, int(self.models['food'].predict(input_data)[0])),
                'shelter_capacity': max(0, int(self.models['shelter'].predict(input_data)[0]))
            }
            if severity > 7:
                results = {k: int(v * 1.2) for k, v in results.items()}
            elif severity > 5:
                results = {k: int(v * 1.1) for k, v in results.items()}
            return results
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            base_value = severity * 100
            return {
                'medical_kits': base_value,
                'food_packets': base_value * 3,
                'shelter_capacity': base_value * 2
            }

ai_model_manager = AIModelManager()

def format_number(value):
    if isinstance(value, (int, float)):
        return "{:,}".format(value)  # Basic comma formatting
    return value

# Template filters and context processors
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    """Jinja2 filter to format datetime objects."""
    if value is None:
        return 'N/A'
    
    try:
        # Handle Firebase server timestamps
        if isinstance(value, dict) and '.sv' in value:
            return datetime.now().strftime(format)
        
        # Handle numeric timestamps (assumed milliseconds)
        if isinstance(value, (int, float)):
            timestamp_seconds = value / 1000
            return datetime.fromtimestamp(timestamp_seconds).strftime(format)
        
        # Handle string timestamps
        if isinstance(value, str):
            try:
                # Try to parse as milliseconds first
                timestamp_seconds = float(value) / 1000
                return datetime.fromtimestamp(timestamp_seconds).strftime(format)
            except ValueError:
                # If not numeric, try to parse as ISO format
                try:
                    return datetime.fromisoformat(value).strftime(format)
                except ValueError:
                    return value  # Return original if can't parse
        
        return 'Invalid Date'
    except Exception as e:
        print(f"Date formatting error: {str(e)}")
        return 'Date Error'

app.jinja_env.filters['datetimeformat'] = datetimeformat



@app.context_processor
def inject_now():
    """Make current datetime available to all templates."""
    return {'now': datetime.utcnow}

def to_locale_string(value, locale='en_IN'):
    """Jinja2 filter to format numbers using babel for locale."""
    if value is None:
        return '-'  # Or some other default representation for None
    try:
        return babel_format_number(value, locale)
    except (ValueError, TypeError):
        return str(value)

app.jinja_env.filters['toLocaleString'] = to_locale_string
# Authentication middleware with role-based access control
@app.before_request
def check_authentication():
    open_routes = ['login', 'signup', 'signup_volunteer', 'static_files', 'firebase_config', 'home']
    if request.endpoint in open_routes:
        return

    if 'user' not in session:
        print("🔒 No user in session - redirecting to login")
        return redirect(url_for('login'))

    try:
        user = auth.get_user(session['user']['uid'])
        session['user']['role'] = user.custom_claims.get('role', 'user')
    except:
        print("❌ Invalid session - clearing and redirecting")
        session.clear()
        return redirect(url_for('login'))

    role = session['user']['role']
    if request.endpoint in ['volunteers_list'] and role != 'admin':
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    elif request.endpoint in ['manage_resources', 'add_resource', 'edit_resource', 'update_resource'] and role != 'resource_manager':
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    elif request.endpoint in ['manage_alerts'] and role != 'admin':
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))

# Routes
@app.route('/')
def home():
    if 'user' in session:
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/firebase-config')
def firebase_config():
    return jsonify(FIREBASE_FRONTEND_CONFIG)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if 'user' in session:
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error="Email and password are required")

        try:
            user = auth.get_user_by_email(email)
            session['user'] = {
                'uid': user.uid,
                'email': user.email,
                'name': user.display_name or email.split('@')[0],
                'role': user.custom_claims.get('role', 'user')
            }
            session.permanent = True
            print(f"✅ User {email} logged in as {session['user']['role']}")
            return redirect(url_for('dashboard'))

        except auth.UserNotFoundError:
            return render_template('login.html', error="User not found")
        except Exception as e:
            print(f"❌ Login error: {str(e)}")
            return render_template('login.html', error="Login failed. Please try again.")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        role = request.form.get('role', 'affected')  # Default role for regular signup

        if not all([email, password, name]):
            return render_template('signup.html', error="All fields are required")

        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=name
            )
            auth.set_custom_user_claims(user.uid, {'role': role})
            firebase_db.reference(f'users/{user.uid}').set({
                'email': email,
                'name': name,
                'role': role,
                'created_at': {'.sv': 'timestamp'}
            })
            session['user'] = {
                'uid': user.uid,
                'email': email,
                'name': name,
                'role': role
            }
            session.permanent = True
            print(f"🎉 New user created as {role}: {email}")
            return redirect(url_for('dashboard'))

        except auth.EmailAlreadyExistsError:
            return render_template('signup.html', error="Email already registered")
        except Exception as e:
            print(f"❌ Signup error: {str(e)}")
            return render_template('signup.html', error="Registration failed. Please try again.")

    return render_template('signup.html')

@app.route('/signup/volunteer', methods=['GET', 'POST'])
def signup_volunteer():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        name = request.form.get('name', '').strip()
        skills = request.form.get('skills', '').strip()
        availability = request.form.get('availability', 'flexible').strip()
        contact = request.form.get('contact', '').strip()
        if not all([email, password, name, contact]):
            flash("Email, password, name, and contact are required", "error")
            return render_template('signup_volunteer.html')

        try:
            user = auth.create_user(
                email=email,
                password=password,
                display_name=name
            )
            auth.set_custom_user_claims(user.uid, {'role': 'volunteer'})
            firebase_db.reference(f'users/{user.uid}').set({
                'email': email,
                'name': name,
                'role': 'volunteer',
                'created_at': {'.sv': 'timestamp'}
            })
            firebase_db.reference(f'volunteers/{user.uid}').set({
                'skills': skills,
                'availability': availability,
                'contact': contact,
                'status': 'available',
                'last_updated': {'.sv': 'timestamp'}
            })
            session['user'] = {'uid': user.uid, 'email': email, 'name': name, 'role': 'volunteer'}
            session.permanent = True
            flash("Volunteer registration successful!", "success")
            return redirect(url_for('dashboard'))
        except auth.EmailAlreadyExistsError:
            flash("Email already registered", "error")
        except Exception as e:
            print(f"Volunteer signup error: {e}")
            flash(f"Registration failed: {str(e)}", "error")

    return render_template('signup_volunteer.html')
@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))

    role = session['user']['role']

    if role == 'admin':
        reports = firebase_db.reference('reports').get() or {}
        reports_list_admin = []
        for report_id, report_data in reports.items():
            if isinstance(report_data, dict):
                try:
                    report_data['severity'] = float(report_data.get('severity', 1))
                except (TypeError, ValueError):
                    report_data['severity'] = 1
                reports_list_admin.append((report_id, report_data))
            else:
                print(f"Warning: Unexpected data format for report ID {report_id}: {report_data}")

        dashboard_predictions = {}
        for report_id, report in dict(reports_list_admin).items():
            prediction = ai_model_manager.predict(
                report.get('help_type', 'unknown'),
                report.get('severity', 1),
                report.get('population_density', 'medium')
            )
            dashboard_predictions[report_id] = prediction

        volunteers = firebase_db.reference('volunteers').get() or {}
        alerts = firebase_db.reference('alerts').order_by_child('timestamp').limit_to_last(5).get() or {}
        resources = firebase_db.reference('resources').get() or {}  # Ensure resources is fetched

        return render_template(
            'admin_dashboard.html',
            reports=reports_list_admin,
            predictions=dashboard_predictions,
            volunteers=volunteers,
            resources=resources,
            alerts=dict(alerts),
            user=session['user']
        )

    elif role == 'resource_manager':
        resources = firebase_db.reference('resources').get() or {}
        reports = firebase_db.reference('reports').order_by_child('status').equal_to('pending').get() or {}
        
        # Convert to list of tuples and sort by timestamp
        sorted_reports = sorted(
            reports.items(),
            key=lambda x: x[1].get('timestamp', 0),
            reverse=True
        )
        top_reports = dict(sorted_reports[:5])  # Get top 5 most recent

        # Similarly for predictions if needed
        predictions = {}  # Add your predictions logic here
        
        return render_template(
            'resource_manager_dashboard.html',
            resources=resources,
            pending_reports=top_reports,
            predictions=predictions,
            user=session['user']
        )

    elif role == 'volunteer':
        alerts_data = firebase_db.reference('alerts').get() or {}
        alerts_list = list(alerts_data.items())
        sorted_alerts = sorted(alerts_list, key=lambda item: item[1].get('timestamp', 0), reverse=True)
        reports = firebase_db.reference('reports').order_by_child('status').equal_to('pending').get() or {}
        return render_template('volunteer_dashboard.html',
                               alerts=sorted_alerts[:10],
                               emergency_reports=reports,
                               user=session['user'])

    elif role == 'affected':
        reports_data = firebase_db.reference('reports').order_by_child('user_id').equal_to(session['user']['uid']).get() or {}
        reports_list = list(reports_data.items())
        # Sort the list by the 'timestamp' attribute (newest first)
        reports_list.sort(key=lambda item: item[1].get('timestamp') or 0, reverse=True)
        return render_template(
            'affected_dashboard.html',
            reports=reports_list,
            user=session['user']
        )

    else:
        return render_template('dashboard.html', user=session['user'])
@app.route('/volunteers')
def volunteers_list():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    try:
        volunteers = firebase_db.reference('volunteers').get() or {}
        users = firebase_db.reference('users').get() or {}
        volunteer_details = {}
        for vid, vdata in volunteers.items():
            if vid in users:
                volunteer_details[vid] = {**vdata, 'name': users[vid].get('name'), 'email': users[vid].get('email')}
        return render_template('volunteers_list.html', 
                             volunteers=volunteer_details, 
                             user=session['user'])
    except Exception as e:
        print(f"Error fetching volunteers: {e}")
        flash("Failed to load volunteer list", "error")
        return render_template('volunteers_list.html', 
                             volunteers={}, 
                             user=session['user'])
    
@app.route('/resources', methods=['GET', 'POST'])
def manage_resources():
    if 'user' not in session or session['user']['role'] not in ['admin', 'resource_manager']:
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        # Handle resource addition
        try:
            quantity = int(request.form.get('quantity', 0))
            resource_data = {
                'name': request.form.get('name', '').strip(),
                'type': request.form.get('type', 'general').strip(),
                'quantity': max(0, quantity),
                'location': request.form.get('location', '').strip(),
                'status': 'available',
                'expiry': request.form.get('expiry', None) or None,
                'last_updated': {'.sv': 'timestamp'}
            }
            if not all([resource_data['name'], resource_data['type']]):
                flash("Name and type are required", "error")
            else:
                firebase_db.reference('resources').push(resource_data)
                flash("Resource added successfully!", "success")
                return redirect(url_for('manage_resources'))
        except ValueError:
            flash("Invalid quantity value", "error")
        except Exception as e:
            print(f"Resource add failed: {str(e)}")
            flash("Failed to add resource", "error")
        return redirect(url_for('manage_resources'))

    # GET request handling
    try:
        resources = firebase_db.reference('resources').get() or {}
        return render_template('resources.html', 
                            resources=resources, 
                            user=session['user'])
    except Exception as e:
        print(f"Error fetching resources: {e}")
        flash("Failed to load resources", "error")
        return render_template('resources.html', 
                            resources={}, 
                            user=session['user'])

@app.route('/resources/edit/<resource_id>', methods=['GET'])
def edit_resource(resource_id):
    if 'user' not in session or session['user']['role'] not in ['admin', 'resource_manager']:
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    try:
        resource = firebase_db.reference(f'resources/{resource_id}').get()
        if resource:
            resource['id'] = resource_id
            return render_template('edit_resource.html', 
                                resource=resource, 
                                user=session['user'])
        else:
            flash("Resource not found", "warning")
            return redirect(url_for('manage_resources'))
    except Exception as e:
        print(f"Error fetching resource for edit: {e}")
        flash("Failed to load resource for editing", "error")
        return redirect(url_for('manage_resources'))

@app.route('/resources/update/<resource_id>', methods=['POST'])
def update_resource(resource_id):
    if 'user' not in session or session['user']['role'] not in ['admin', 'resource_manager']:
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    try:
        updated_data = {
            'name': request.form.get('name', '').strip(),
            'type': request.form.get('type', 'general').strip(),
            'quantity': int(request.form.get('quantity', 0)),
            'location': request.form.get('location', '').strip(),
            'expiry': request.form.get('expiry', None) or None,
            'last_updated': {'.sv': 'timestamp'}
        }
        if not all([updated_data['name'], updated_data['type']]):
            flash("Name and type are required", "error")
        elif updated_data['quantity'] < 0:
            flash("Quantity cannot be negative", "error")
        else:
            firebase_db.reference(f'resources/{resource_id}').update(updated_data)
            flash("Resource updated successfully!", "success")
    except ValueError:
        flash("Invalid quantity value", "error")
    except Exception as e:
        print(f"Resource update failed: {str(e)}")
        flash("Failed to update resource", "error")
    return redirect(url_for('manage_resources'))

@app.route('/alerts', methods=['GET'])
def manage_alerts():
    if 'user' not in session or session['user']['role'] not in ['admin', 'volunteer']:
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    try:
        alerts = firebase_db.reference('alerts').order_by_child('timestamp').limit_to_last(50).get() or {}
        return render_template('alerts.html', 
                           alerts=alerts, 
                           user=session['user'])
    except Exception as e:
        print(f"Alerts page error: {str(e)}")
        flash("Failed to load alerts", "error")
        return render_template('alerts.html', 
                           alerts={}, 
                           user=session['user'])

@app.route('/alerts/send', methods=['POST'])
def send_alert():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Unauthorized access", "error")
        return redirect(url_for('dashboard'))
    try:
        alert_data = {
            'message': request.form.get('message', '').strip(),
            'priority': request.form.get('priority', 'medium').strip(),
            'target': request.form.get('target', 'all').strip(),
            'sent_by': session['user']['uid'],
            'sent_by_name': session['user']['name'],
            'timestamp': {'.sv': 'timestamp'}
        }
        if not alert_data['message']:
            flash("Message cannot be empty", "error")
        else:
            firebase_db.reference('alerts').push(alert_data)
            print(f"Sending alert notifications: {alert_data}")
            flash("Alert sent successfully!", "success")
    except Exception as e:
        print(f"Alert send failed: {str(e)}")
        flash("Failed to send alert", "error")
    return redirect(url_for('manage_alerts'))

@app.route('/report', methods=['GET', 'POST'])
def report():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            severity = int(request.form.get('severity', 1))
            severity = max(1, min(severity, 10))
        except:
            severity = 1

        report_data = {
            'location': request.form.get('location', '').strip(),
            'help_type': request.form.get('help_type', 'general').strip(),
            'description': request.form.get('description', '').strip(),
            'severity': severity,
            'status': 'pending',
            'user_id': session['user']['uid'],
            'user_name': session['user']['name'],
            'timestamp': {'.sv': 'timestamp'},
            'population_density': request.form.get('population_density', 'medium').strip()
        }

        if not all([report_data['location'], report_data['help_type']]):
            flash("Location and help type are required", "error")
            return render_template('report.html')

        try:
            new_report_ref = firebase_db.reference('reports').push(report_data)
            print(f"Alert triggered for report: {new_report_ref.key} with data: {report_data}")
            flash("Report submitted successfully!", "success")
            return redirect(url_for('dashboard'))
        except Exception as e:
            print(f"Failed to save report: {str(e)}")
            flash("Failed to save report. Please try again.", "error")

    return render_template('report.html', user=session.get('user'))

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))

    try:
        user_data = firebase_db.reference(f'users/{session["user"]["uid"]}').get() or {}
        user_reports = {}
        all_reports = firebase_db.reference('reports').get() or {}
        for report_id, report in all_reports.items():
            if report.get('user_id') == session['user']['uid']:
                user_reports[report_id] = report

        return render_template('profile.html',
                           user_data=user_data,
                           user_reports=user_reports,
                           user=session['user'])
    except Exception as e:
        print(f"Profile error: {str(e)}")
        flash("Failed to load profile data", "error")
        return render_template('profile.html',
                           user_data={},
                           user_reports={},
                           user=session['user'])
 # --- AI Prediction Dashboard ---
@app.route('/ai_predictions')
def ai_predictions():
    if 'user' not in session:
        return redirect(url_for('login'))

    predictions_list = []
    latest_prediction_data = None

    try:
        # Get predictions from Firebase
        predictions_ref = firebase_db.reference('predictions')
        predictions_data = predictions_ref.order_by_child('timestamp').limit_to_last(100).get() or {}

        for pred_id, pred_data in predictions_data.items():
            if isinstance(pred_data, dict):
                processed_data = {'id': pred_id}

                # Handle timestamp
                timestamp = pred_data.get('timestamp')
                if isinstance(timestamp, dict) and '.sv' in timestamp:
                    processed_data['timestamp'] = datetime.now().timestamp() * 1000
                elif isinstance(timestamp, (int, float)):
                    processed_data['timestamp'] = timestamp
                else:
                    processed_data['timestamp'] = None

                # Process numeric fields, ensuring they are numbers or None
                numeric_fields = ['medical_kits', 'food_packets', 'shelter_capacity', 'volunteers_needed']
                for field in numeric_fields:
                    value = pred_data.get(field)
                    if value is not None:
                        try:
                            processed_data[field] = int(value)
                        except (ValueError, TypeError):
                            processed_data[field] = None
                    else:
                        processed_data[field] = None

                # Process other string fields with a default
                processed_data['disaster_type'] = pred_data.get('disaster_type', '').strip()
                severity_value = pred_data.get('severity')
                if isinstance(severity_value, str):
                    processed_data['severity'] = severity_value.strip()
                else:
                    processed_data['severity'] = str(severity_value) if severity_value is not None else ''
                processed_data['population_density'] = pred_data.get('population_density', '').strip()
                processed_data['location'] = pred_data.get('location', '').strip()
                processed_data['prediction_source'] = pred_data.get('prediction_source', '').strip()

                predictions_list.append(processed_data)

        # Sort by timestamp (newest first), handling potential None timestamps
        predictions_list.sort(key=lambda x: x.get('timestamp') or 0, reverse=True)

        latest_prediction_data = predictions_list[0] if predictions_list else None

        return render_template('ai_predictions.html',
                               predictions=predictions_list,
                               latest_prediction=latest_prediction_data,
                               user=session['user'])

    except Exception as e:
        print(f"AI Predictions error: {str(e)}")
        flash("Failed to load historical predictions", "error")
        return render_template('ai_predictions.html',
                               predictions=[],
                               latest_prediction=None,
                               user=session.get('user'))
                               
# --- AI Prediction Logic ---
@app.route('/ai/predict', methods=['POST'])
def predict_resources():
    if request.method == 'POST':
        try:
            data = request.get_json()
            location = data.get('location')
            disaster_type = data.get('disaster_type')
            severity = int(data.get('severity', 1))
            population_density = data.get('population_density', 'medium')
            prediction_source = data.get('prediction_source', 'manual_input')

            # --- Basic Rule-Based Prediction Logic ---
            medical_kits = 10 * severity
            food_packets = 50 * severity
            shelter_capacity = 20 * severity
            volunteers_needed = 5 * severity

            if population_density == 'high' or population_density == 'urban':
                medical_kits *= 2
                food_packets *= 3
                shelter_capacity *= 2.5
                volunteers_needed *= 1.5
            elif population_density == 'low' or population_density == 'rural':
                medical_kits *= 0.75
                food_packets *= 0.8
                shelter_capacity *= 0.6
                volunteers_needed *= 0.9

            predictions = {
                'medical_kits': int(medical_kits),
                'food_packets': int(food_packets),
                'shelter_capacity': int(shelter_capacity),
                'volunteers_needed': int(volunteers_needed)
            }

            # Store the prediction in Firebase
            predictions_ref = firebase_db.reference('predictions')
            predictions_ref.push({
                'timestamp': datetime.now().timestamp() * 1000,
                'location': location,
                'disaster_type': disaster_type,
                'severity': severity,
                'population_density': population_density,
                'prediction_source': prediction_source,
                'medical_kits': predictions['medical_kits'],
                'food_packets': predictions['food_packets'],
                'shelter_capacity': predictions['shelter_capacity'],
                'volunteers_needed': predictions['volunteers_needed']
            })

            return jsonify(predictions)

        except Exception as e:
            print(f"Prediction API error: {str(e)}")
            return jsonify({'error': 'Failed to generate prediction'}), 500

    return jsonify({'error': 'Invalid request method'}), 400

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("You have been logged out.", "info")
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)