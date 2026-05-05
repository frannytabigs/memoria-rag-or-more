from app import create_app
from app.database import init_db

app = create_app()

if __name__ == "__main__":
    # Create the SQLite file and table before the server starts
    init_db() 
    app.run(debug=True)