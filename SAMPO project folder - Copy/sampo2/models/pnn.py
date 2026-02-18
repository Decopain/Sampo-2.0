import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

class ParallelNeuralNetwork:
    def __init__(self, input_shapes, output_dim=41, name="pnn_model"):
        """
        initializes the PNN model.
        
        Args:
            input_shapes (dict): Dictionary mapping branch names to input shapes (tuple).
                                 e.g. {'daily': (58,), 'h1': (58,), 'm15': (58,)}
            output_dim (int): Dimension of the final prediction output.
            name (str): Model name.
        """
        self.input_shapes = input_shapes
        self.output_dim = output_dim
        self.name = name
        self.model = self._build_model()

    def _build_branch(self, input_shape, name_prefix):
        """Builds a single branch of the PNN."""
        inp = Input(shape=input_shape, name=f"{name_prefix}_input")
        x = Dense(64, activation='relu')(inp)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        return inp, x

    def _build_model(self):
        inputs = []
        branch_outputs = []

        # Build branches dynamically based on config
        # Expect keys like 'daily', 'h1', 'm15'
        for branch_name, shape in self.input_shapes.items():
            inp, out = self._build_branch(shape, branch_name)
            inputs.append(inp)
            branch_outputs.append(out)

        # Merge
        if len(branch_outputs) > 1:
            x = Concatenate()(branch_outputs)
        else:
            x = branch_outputs[0]

        # Common Head
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Latent Representation (Enriched Features)
        # We name this layer so we can extract it later
        enriched = Dense(100, activation='linear', name='enriched_features')(x) 
        
        # Final Output (for auxiliary training)
        output = Dense(self.output_dim, activation='linear', name='output')(enriched)

        model = Model(inputs=inputs, outputs=output, name=self.name)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def summary(self):
        return self.model.summary()

    def fit(self, X_train, y_train, validation_data=None, epochs=10, batch_size=32):
        """
        X_train should be a list of numpy arrays corresponding to the branches.
        """
        return self.model.fit(X_train, y_train, 
                              validation_data=validation_data, 
                              epochs=epochs, 
                              batch_size=batch_size,
                              verbose=1)

    def predict(self, X):
        return self.model.predict(X)

    def get_feature_extractor(self):
        """Returns a model that outputs the 'enriched_features' layer."""
        return Model(inputs=self.model.inputs, outputs=self.model.get_layer('enriched_features').output)

    def save(self, filepath):
        # Save in native Keras format
        if not str(filepath).endswith('.keras') and not str(filepath).endswith('.h5'):
             filepath = str(filepath) + ".keras"
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load(filepath):
        from tensorflow.keras.models import load_model
        return load_model(filepath)
