# api/index.py
from app import app  # Import your Flask instance
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.wrappers import Response

# Vercel expects a function called 'handler'
def handler(environ, start_response):
    return app.wsgi_app(environ, start_response)
