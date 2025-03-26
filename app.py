from src.utils.system_status import get_system_status

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', system_status=get_system_status()) 