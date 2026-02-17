import h5py

def list_weights(h5_path):
    with h5py.File(h5_path, 'r') as f:
        def print_node(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"Dataset: {name}, Shape: {obj.shape}")
            elif isinstance(obj, h5py.Group):
                print(f"Group: {name}")
        
        # f.visititems(print_node)
        # Just check top level weight groups
        if 'model_weights' in f:
             weights = f['model_weights']
             print(f"Weights groups: {list(weights.keys())}")
             if 'xception' in weights:
                 print(f"Xception weights: {list(weights['xception'].keys())[:5]}...")

if __name__ == "__main__":
    list_weights("xception_v4_1_10_0.998.h5")
