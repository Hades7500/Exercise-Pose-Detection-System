import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH = "landmarks.csv"
MODEL_DIR = "models"

SEQUENCE_LENGTH = 20   # 20 frames ~= 2 seconds at ~10 FPS
STRIDE = 5              # Move the temporal window by 5 frames

RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15

BATCH_SIZE = 32
EPOCHS = 40


# ============================================================
# FEATURES
# ============================================================

ANGLE_FEATURES = ['left_elbow_angle', 'right_elbow_angle', 'left_shoulder_angle', 'right_shoulder_angle', 'left_knee_angle', 'right_knee_angle', 'left_hip_angle', 'right_hip_angle', 'elbow_symmetry', 'knee_symmetry', 'shoulder_symmetry', 'hip_symmetry', 'torso_lean']

LANDMARK_FEATURES = ['left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z', 'left_shoulder_vis', 'right_shoulder_x', 'right_shoulder_y', 'right_shoulder_z', 'right_shoulder_vis', 'left_elbow_x', 'left_elbow_y', 'left_elbow_z', 'left_elbow_vis', 'right_elbow_x', 'right_elbow_y', 'right_elbow_z', 'right_elbow_vis', 'left_wrist_x', 'left_wrist_y', 'left_wrist_z', 'left_wrist_vis', 'right_wrist_x', 'right_wrist_y', 'right_wrist_z', 'right_wrist_vis', 'left_hip_x', 'left_hip_y', 'left_hip_z', 'left_hip_vis', 'right_hip_x', 'right_hip_y', 'right_hip_z', 'right_hip_vis', 'left_knee_x', 'left_knee_y', 'left_knee_z', 'left_knee_vis', 'right_knee_x', 'right_knee_y', 'right_knee_z', 'right_knee_vis', 'left_ankle_x', 'left_ankle_y', 'left_ankle_z', 'left_ankle_vis', 'right_ankle_x', 'right_ankle_y', 'right_ankle_z', 'right_ankle_vis']

FEATURES = ANGLE_FEATURES + LANDMARK_FEATURES


# ============================================================
# SPLIT BY VIDEO
# ============================================================

def split_videos(df):
    """
    Split videos rather than individual frames/sequences.

    This prevents nearly identical overlapping sequences from
    the same video appearing in both train and test sets.
    """

    videos = (
        df[["video_file", "exercise"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    print(f"Total videos: {len(videos)}")
    print("\nVideos by exercise:")
    print(videos["exercise"].value_counts())

    train_videos, temp_videos = train_test_split(
        videos,
        test_size=TEST_SIZE + VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=videos["exercise"],
    )

    # Split the remaining 30% equally into validation/test.
    relative_test_size = TEST_SIZE / (TEST_SIZE + VAL_SIZE)

    val_videos, test_videos = train_test_split(
        temp_videos,
        test_size=relative_test_size,
        random_state=RANDOM_STATE,
        stratify=temp_videos["exercise"],
    )

    print("\nVideo split:")
    print(f"  Train:      {len(train_videos)}")
    print(f"  Validation: {len(val_videos)}")
    print(f"  Test:       {len(test_videos)}")

    print("\nTrain classes:")
    print(train_videos["exercise"].value_counts())

    print("\nValidation classes:")
    print(val_videos["exercise"].value_counts())

    print("\nTest classes:")
    print(test_videos["exercise"].value_counts())

    return train_videos, val_videos, test_videos


# ============================================================
# CREATE TEMPORAL SEQUENCES
# ============================================================

def make_sequences(df, video_split):
    """
    Convert individual frame rows into temporal sequences.

    Each sequence contains:
        SEQUENCE_LENGTH frames
        x len(FEATURES) features

    Example:
        (20, 61)
    """

    allowed_videos = set(video_split["video_file"])

    subset = df[df["video_file"].isin(allowed_videos)].copy()

    X = []
    y = []

    for video_file, group in subset.groupby("video_file"):
        group = group.sort_values("frame_idx")

        exercise = group["exercise"].iloc[0]

        # Safety check: all rows in a video should have the same exercise.
        if group["exercise"].nunique() != 1:
            print(f"WARNING: multiple exercise labels in {video_file}")
            continue

        values = (
            group[FEATURES]
            .fillna(0)
            .to_numpy(dtype=np.float32)
        )

        if len(values) < SEQUENCE_LENGTH:
            # Video is too short to form one sequence.
            continue

        for start in range(
            0,
            len(values) - SEQUENCE_LENGTH + 1,
            STRIDE,
        ):
            end = start + SEQUENCE_LENGTH

            sequence = values[start:end]

            if sequence.shape != (SEQUENCE_LENGTH, len(FEATURES)):
                continue

            X.append(sequence)
            y.append(exercise)

    if not X:
        raise RuntimeError(
            "No sequences were generated. "
            "Check CSV_PATH and SEQUENCE_LENGTH."
        )

    return np.asarray(X, dtype=np.float32), np.asarray(y)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_sequences(X_train, X_val, X_test):
    """
    Fit the scaler ONLY on training data.

    CNN input:
        samples x frames x features

    StandardScaler works on:
        samples*frames x features
    """

    n_train, sequence_length, n_features = X_train.shape

    scaler = StandardScaler()

    train_flat = X_train.reshape(-1, n_features)

    scaler.fit(train_flat)

    def transform(X):
        return scaler.transform(
            X.reshape(-1, n_features)
        ).reshape(X.shape).astype(np.float32)

    return (
        transform(X_train),
        transform(X_val),
        transform(X_test),
        scaler,
    )


# ============================================================
# CNN
# ============================================================

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(
            shape=(SEQUENCE_LENGTH, len(FEATURES))
        ),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu",
            padding="same",
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.MaxPooling1D(
            pool_size=2
        ),

        tf.keras.layers.Conv1D(
            filters=128,
            kernel_size=3,
            activation="relu",
            padding="same",
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(
            64,
            activation="relu",
        ),

        tf.keras.layers.Dropout(0.30),

        tf.keras.layers.Dense(
            2,
            activation="softmax",
        ),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("TEMPORAL CNN TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load CSV
    # --------------------------------------------------------

    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{CSV_PATH}'. "
            "Put landmarks.csv in the same directory as this script."
        )

    df = pd.read_csv(CSV_PATH)

    print(f"\nLoaded {len(df)} frame rows.")
    print(f"Features per frame: {len(FEATURES)}")

    required_columns = (
        FEATURES
        + ["exercise", "video_file", "frame_idx"]
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in CSV:\n"
            + "\n".join(missing)
        )

    print("\nExercises:")
    print(df["exercise"].value_counts())

    # --------------------------------------------------------
    # Split by VIDEO
    # --------------------------------------------------------

    train_videos, val_videos, test_videos = split_videos(df)

    # --------------------------------------------------------
    # Create sequences
    # --------------------------------------------------------

    print("\nCreating temporal sequences...")

    X_train, y_train = make_sequences(df, train_videos)
    X_val, y_val = make_sequences(df, val_videos)
    X_test, y_test = make_sequences(df, test_videos)

    print(f"\nTrain sequences:      {X_train.shape}")
    print(f"Validation sequences: {X_val.shape}")
    print(f"Test sequences:       {X_test.shape}")

    print("\nSequence labels:")
    print("Train:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("Val:  ", dict(zip(*np.unique(y_val, return_counts=True))))
    print("Test: ", dict(zip(*np.unique(y_test, return_counts=True))))

    # --------------------------------------------------------
    # Encode labels
    # --------------------------------------------------------

    encoder = LabelEncoder()

    y_train_encoded = encoder.fit_transform(y_train)
    y_val_encoded = encoder.transform(y_val)
    y_test_encoded = encoder.transform(y_test)

    print("\nClass encoding:")
    for i, class_name in enumerate(encoder.classes_):
        print(f"  {i} = {class_name}")

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    print("\nNormalizing features...")

    (
        X_train,
        X_val,
        X_test,
        scaler,
    ) = normalize_sequences(
        X_train,
        X_val,
        X_test,
    )

    # --------------------------------------------------------
    # Build CNN
    # --------------------------------------------------------

    print("\nBuilding CNN...")

    model = build_model()

    model.summary()

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        ),
    ]

    print("\nStarting training...")

    history = model.fit(
        X_train,
        y_train_encoded,
        validation_data=(
            X_val,
            y_val_encoded,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # --------------------------------------------------------
    # Test evaluation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)

    loss, accuracy = model.evaluate(
        X_test,
        y_test_encoded,
        verbose=0,
    )

    print(f"Test loss:     {loss:.4f}")
    print(f"Test accuracy: {accuracy:.4%}")

    probabilities = model.predict(
        X_test,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test_encoded,
            predictions,
            target_names=encoder.classes_,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test_encoded,
            predictions,
        )
    )

    # --------------------------------------------------------
    # Save model + preprocessing
    # --------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = os.path.join(
        MODEL_DIR,
        "exercise_cnn.keras",
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "exercise_cnn_scaler.pkl",
    )

    encoder_path = os.path.join(
        MODEL_DIR,
        "exercise_cnn_encoder.pkl",
    )

    model.save(model_path)

    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    print("\nSaved:")
    print(f"  {model_path}")
    print(f"  {scaler_path}")
    print(f"  {encoder_path}")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
