#!/usr/bin/env python3
"""
Marketing Mix Model Dashboard Runner

This script starts the FastAPI server with the dashboard UI.
"""

import uvicorn
import sys
from pathlib import Path

def main():
    """Run the Marketing Mix Model API server with dashboard."""
    
    # Add the src directory to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    print("🚀 Starting Marketing Mix Model Dashboard...")
    print("📊 Dashboard will be available at: http://localhost:8000")
    print("📚 API documentation at: http://localhost:8000/docs")
    print("🔍 API health check at: http://localhost:8000/health")
    print("\nPress Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        uvicorn.run(
            "src.api.app:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting dashboard: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 