import os
import onnx
# from onnx import optimizer
from onnxsim import simplify

# Folder containing ONNX models
folder_path = "onnx_fp32/"

# Optimization passes to apply
optimization_passes = [
    'eliminate_deadend', 
    'eliminate_identity',
    'eliminate_nop_transpose', 
    'fuse_consecutive_transposes',
    'fuse_add_bias_into_conv',
    'fuse_consecutive_squeezes',
    'eliminate_nop_pad'
]

# Iterate through all files in the folder
for file_name in os.listdir(folder_path):
    # Check if the file is an ONNX model
    if file_name.endswith(".onnx"):
        model_path = os.path.join(folder_path, file_name)
        print(f"Processing model: {file_name}")

        # Load the ONNX model
        model = onnx.load(model_path)

        # Flag to track if we modified the model
        modified = False

        # Iterate through the graph nodes
        for node in model.graph.node:
            # Check for Pad nodes and replace "reflect" mode with "constant"
            if node.op_type == "Pad":
                for attr in node.attribute:
                    if attr.name == "mode" and attr.s == b"reflect":
                        print(f"  Found mirror padding in node {node.name}. Replacing with zero padding.")
                        attr.s = b"constant"  # Change to constant mode (zero padding)
                        modified = True

            # Check for Resize nodes and fix unsupported modes
            if node.op_type == "Resize":
                for attr in node.attribute:
                    # Fix coordinate_transformation_mode (only "align_corners" and "half_pixel" are supported)
                    if attr.name == "coordinate_transformation_mode" and attr.s == b"asymmetric":
                        print(f"  Found unsupported 'asymmetric' in node {node.name}. Replacing with 'align_corners'.")
                        attr.s = b"align_corners"  # Change to 'align_corners' or 'half_pixel' as required
                        modified = True

                    # Fix nearest_mode (only "round_prefer_ceil" is supported, not "floor")
                    if attr.name == "nearest_mode" and attr.s == b"floor":
                        print(f"  Found unsupported 'floor' mode in node {node.name}. Replacing with 'round_prefer_ceil'.")
                        attr.s = b"round_prefer_ceil"  # Change to 'round_prefer_ceil'
                        modified = True

        # If the model was modified, perform shape inference
        if modified:
            # Perform shape inference
            print("  Running shape inference...")
            model = onnx.shape_inference.infer_shapes(model)

            # Simplify the model
            print("  Running ONNX simplification...")
            model_simp, check = simplify(model)
            if not check:
                raise ValueError(f"  Simplification check failed for {file_name}")
            model = model_simp

            # Apply optimization passes
            print("  Running ONNX optimization...")
            # model = optimizer.optimize(model, optimization_passes)

            # Save the new model with _nomp suffix
            new_model_path = os.path.join(folder_path, file_name.replace(".onnx", "_nomp.onnx"))
            onnx.save(model, new_model_path)
            print(f"  Saved modified model as: {new_model_path}")
        else:
            print(f"  No modifications required for {file_name}.")

print("Processing completed.")
