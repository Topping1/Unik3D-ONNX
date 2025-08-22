# Unik3D-ONNX
Single image to 3D points and 3D mesh - [UniK3D](https://github.com/lpiccinelli-eth/UniK3D) converted to ONNX with new interface.

# TL;DR
Modification of the UniK3D project:
- Safetensors models (small, base, large) converted to ONNX
- Upgraded Gradio interface: filter points by distance, simple density-based point filtering, ball-pivoting 3D mesh reconstruction.

# Installing
- CPU install ```pip install -r requirements.txt```
- GPU install ```pip install -r requirements-gpu.txt```
- Download and place the [ONNX models](https://github.com/Topping1/Unik3D-ONNX/releases/tag/0.1-alpha) in the same folder as the ```UniK3D-onnx.py``` and execute ```python UniK3D-onnx.py```
- FYI the libraries are gradio, onnxruntime (or onnxruntime-gpu), trimesh, pillow, numpy, open3d (optional: the gradio interface will detect if this library is installed. If not installed, only point cloud reconstruction will be available. If installed, both 3D point reconstruction and 3D mesh reconstruction will be available)

# Enjoy
Enjoy!
