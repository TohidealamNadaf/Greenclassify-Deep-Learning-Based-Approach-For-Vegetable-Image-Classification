import h5py

def list_nested_weights(h5_path):
    with h5py.File(h5_path, 'r') as f:
        if 'model_weights' in f:
             weights = f['model_weights']
             if 'xception' in weights:
                 xception_group = weights['xception']
                 print(f"Xception internal groups/layers: {list(xception_group.keys())[:20]}...")
                 # Check one layer
                 first_layer = list(xception_group.keys())[0]
                 print(f"Weights for {first_layer}: {list(xception_group[first_layer].keys())}")

if __name__ == "__main__":
    list_nested_weights("xception_v4_1_10_0.998.h5")
