from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adam

# Settings
IMG_SIZE = (48, 48)
NUM_CLASSES = 7

# Build model
base_model = MobileNetV2(input_shape=IMG_SIZE + (3,), include_top=False, weights=None)
x = GlobalAveragePooling2D()(base_model.output)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Compile and save
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
model.save("expression_model.h5")

print("✅ Dummy model created and saved as expression_model.h5")
