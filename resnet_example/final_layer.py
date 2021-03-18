for layer in base_model.layers:
    layer.trainable = False