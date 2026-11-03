"""Writes requirements-serve.txt pinned to the versions installed NOW (the ones that trained the model)."""
import importlib.metadata as m
pk = ["numpy", "pandas", "scikit-learn", "scipy", "lightgbm", "joblib", "fastapi", "uvicorn", "pydantic"]
open("requirements-serve.txt", "w").write("\n".join(f"{p}=={m.version(p)}" for p in pk) + "\n")
print(open("requirements-serve.txt").read())
