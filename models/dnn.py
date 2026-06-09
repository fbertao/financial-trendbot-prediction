import tensorflow as tf


def create_model(input_dim):

    model = tf.keras.models.Sequential(
        [tf.keras.layers.Input((input_dim,)
        ),

        tf.keras.layers.Dense(32,
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(16,
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(8,
            activation='relu'
        ),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(2,
            activation='softmax'
        )

    ])

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4
    )

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )

    return model