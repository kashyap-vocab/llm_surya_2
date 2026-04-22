"""
Thin launcher for the compiled wrapper module.
wrapper.py is compiled to .so by Cython — __name__ inside it is never
'__main__', so uvicorn.run() in wrapper.py won't fire. We call it here.
"""
import os
import uvicorn
from wrapper import app  # imports from wrapper.cpython-3xx-*.so

if __name__ == "__main__":
    port = int(os.environ.get("WRAPPER_PORT", "9000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
