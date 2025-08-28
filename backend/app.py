from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import os
import logging
from datetime import datetime
import traceback

# Import your existing ExcelQuerySystem class
from test import ExcelQuerySystem  # Replace with your actual filename

# Set up Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store the query system instance
query_system = None

def initialize_query_system():
    """Initialize the Excel Query System"""
    global query_system
    try:
        query_system = ExcelQuerySystem()
        logger.info("Excel Query System initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Excel Query System: {str(e)}")
        return False

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system_ready": query_system is not None
    }), 200

@app.route('/query', methods=['POST'])
def query_excel_data():
    """Main query endpoint"""
    try:
        # Check if system is initialized
        if query_system is None:
            return jsonify({
                "error": "System not initialized",
                "message": "Excel Query System is not ready"
            }), 500
       
        # Get request data
        data = request.get_json()
       
        if not data or 'query' not in data:
            return jsonify({
                "error": "Invalid request",
                "message": "Query parameter is required"
            }), 400
       
        user_query = data['query'].strip()
       
        if not user_query:
            return jsonify({
                "error": "Empty query",
                "message": "Query cannot be empty"
            }), 400
       
        # Optional parameters
        top_k = data.get('top_k', 15)
       
        logger.info(f"Processing query: {user_query}")
       
        # Process the query
        result = query_system.query(user_query, top_k)
       
        # Build response
        response = {
            "query": result['query'],
            "answer": result['answer'],
            "sources": result['sources'][:5],  # Limit to top 5 sources for API response
            "total_results_found": result['total_results_found'],
            "chart_generated": result.get('chart_generated', False),
            "chart_url": None,
            "chart_type": result.get('chart_type', None),
            "numerical_summary": {
                "total_numbers_found": len(result['numerical_summary'].get('numerical_context', [])),
                "aggregations": result['numerical_summary'].get('aggregations', {})
            },
            "timestamp": datetime.now().isoformat()
        }
       
        # Add chart URL if chart was generated
        if result.get('chart_generated') and result.get('chart_path'):
            chart_filename = os.path.basename(result['chart_path'])
            response['chart_url'] = url_for('serve_chart', filename=chart_filename, _external=True)
       
        return jsonify(response), 200
       
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        logger.error(traceback.format_exc())
       
        return jsonify({
            "error": "Internal server error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/charts/<filename>', methods=['GET'])
def serve_chart(filename):
    """Serve generated chart images"""
    try:
        charts_dir = "generated_charts"
        chart_path = os.path.join(charts_dir, filename)
       
        if not os.path.exists(chart_path):
            return jsonify({
                "error": "Chart not found",
                "message": f"Chart file {filename} does not exist"
            }), 404
       
        return send_file(chart_path, mimetype='image/png')
       
    except Exception as e:
        logger.error(f"Error serving chart: {str(e)}")
        return jsonify({
            "error": "Error serving chart",
            "message": str(e)
        }), 500

@app.route('/charts', methods=['GET'])
def list_charts():
    """List all available charts"""
    try:
        charts_dir = "generated_charts"
       
        if not os.path.exists(charts_dir):
            return jsonify({
                "charts": [],
                "count": 0
            }), 200
       
        chart_files = [f for f in os.listdir(charts_dir) if f.endswith('.png')]
       
        charts = []
        for filename in sorted(chart_files, reverse=True):  # Most recent first
            file_path = os.path.join(charts_dir, filename)
            file_stat = os.stat(file_path)
           
            charts.append({
                "filename": filename,
                "url": url_for('serve_chart', filename=filename, _external=True),
                "created": datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                "size": file_stat.st_size
            })
       
        return jsonify({
            "charts": charts,
            "count": len(charts)
        }), 200
       
    except Exception as e:
        logger.error(f"Error listing charts: {str(e)}")
        return jsonify({
            "error": "Error listing charts",
            "message": str(e)
        }), 500

@app.route('/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    try:
        if query_system is None:
            return jsonify({
                "error": "System not initialized"
            }), 500
       
        info = {
            "system_status": "ready",
            "total_chunks": len(query_system.metadata),
            "vector_db_path": query_system.vector_db_path,
            "charts_directory": query_system.charts_dir,
            "available_charts": len([f for f in os.listdir(query_system.charts_dir)
                                   if f.endswith('.png')]) if os.path.exists(query_system.charts_dir) else 0,
            "supported_chart_types": ["finance", "attrition"],
            "example_queries": [
                "Show me financial trend",
                "What is attrition trend?",
                "What is the total budget for this year?",
                "Display finance numbers",
                "Show attrition data"
            ]
        }
       
        return jsonify(info), 200
       
    except Exception as e:
        logger.error(f"Error getting system info: {str(e)}")
        return jsonify({
            "error": "Error getting system info",
            "message": str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "message": "The requested endpoint does not exist"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "An unexpected error occurred"
    }), 500

if __name__ == '__main__':
    # Initialize the query system on startup
    print("🚀 Starting Excel Query API...")
   
    if initialize_query_system():
        print("✅ Excel Query System initialized successfully")
        print("\n📋 Available Endpoints:")
        print("POST /query - Query Excel data")
        print("GET /charts/<filename> - Serve chart images")
        print("GET /charts - List all charts")
        print("GET /system/info - Get system information")
        print("GET /health - Health check")
        print(f"\n🌐 Server starting at http://localhost:5000")
        print("\n📖 Example API call:")
        print("curl -X POST http://localhost:5000/query \\")
        print("  -H 'Content-Type: application/json' \\")
        print("  -d '{\"query\": \"Show me financial trend\"}'")
       
        # Run the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True  # Set to False in production
        )
    else:
        print("❌ Failed to initialize Excel Query System")
        print("Make sure:")
        print("1. Vector database files exist")
        print("2. OpenAI credentials are configured")
        print("3. All dependencies are installed")