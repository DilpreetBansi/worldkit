"""Serve a WorldKit model via FastAPI.

Requires: pip install worldkit[serve]
"""

print("WorldKit API Server Example")
print("=" * 50)
print()
print("To serve a model:")
print()
print("  1. Install serve extras: pip install worldkit[serve]")
print("  2. Save a model: model.save('my_model.wk')")
print("  3. Start the server:")
print("     worldkit serve --model my_model.wk --port 8000")
print()
print("API Endpoints:")
print("  GET  /health       - Server status")
print("  POST /encode       - Encode observation to latent")
print("  POST /predict      - Predict future states")
print("  POST /plan         - Plan action sequence to goal")
print("  POST /plausibility - Score physical plausibility")
