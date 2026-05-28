import sys
import traceback

try:
    print("Attempting to import Generator...")
    from src.stage1_gan.generator import Generator
    print("Success! Generator imported")
    print(f"Generator: {Generator}")
except Exception as e:
    print("Error:", e)
    traceback.print_exc()

# Try importing the module
try:
    print("\nAttempting to import generator module...")
    import src.stage1_gan.generator as gen_module
    print(f"Module contents: {dir(gen_module)}")
except Exception as e:
    print("Module import error:", e)
    traceback.print_exc()
