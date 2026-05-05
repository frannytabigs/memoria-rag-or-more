from flask import Flask

def create_app():
    # Point Flask to your templates and static folders in the root directory
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    
    with app.app_context():
        # Import routes so the app knows your endpoints exist
        from . import routes
        
    return app