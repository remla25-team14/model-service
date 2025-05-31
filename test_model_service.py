#!/usr/bin/env python3
"""Test script for model-service with dynamic versioning"""

def test_imports():
    """Test that all required libraries can be imported"""
    try:
        from libversion import VersionUtil
        print("✅ lib-version import: OK")
        version = VersionUtil.get_version()
        print(f"✅ Version: {version}")
    except ImportError as e:
        print(f"❌ lib-version import failed: {e}")
        return False
    
    try:
        from libml.data_preprocessing import preprocess_reviews
        print("✅ lib-ml import: OK")
        
        # Test preprocessing
        test_data = {"Review": ["Great food!"]}
        result = preprocess_reviews(test_data)
        print(f"✅ Preprocessing test: {result}")
    except ImportError as e:
        print(f"❌ lib-ml import failed: {e}")
        return False
    
    return True

def test_app_creation():
    """Test that the Flask app can be created"""
    try:
        # Import app components
        import app
        print("✅ App import: OK")
        print(f"✅ Service version: {app.SERVICE_VERSION}")
        return True
    except Exception as e:
        print(f"❌ App creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing model-service with dynamic versioning...")
    print("=" * 50)
    
    try:
        if test_imports() and test_app_creation():
            print("\n" + "=" * 50)
            print("🎉 All tests passed! Model-service is ready.")
            print("✅ Dynamic versioning: Working")
            print("✅ lib-ml integration: Working")
            print("✅ App initialization: Working")
        else:
            print("\n❌ Some tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\n💡 Make sure you have installed the dependencies:")
        print("pip install -r requirements.txt")
        exit(1) 