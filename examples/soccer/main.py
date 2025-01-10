import argparse
import json
from enum import Enum
from typing import Iterator, List, Dict, Any
from collections import defaultdict

import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_with_labels_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration
from pytesseract import pytesseract

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection-v2.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3
BALL_COLOR_ID = 4

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700', '#FFFFFF']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    JERSEY_DETECTION = 'JERSEY_DETECTION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray,
    labels: List[str] = None
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)

    # Filter points and labels by team/color and draw
    for color_id in range(5):  # 5 possible colors
        team_mask = (color_lookup == color_id)
        team_xy = transformed_xy[team_mask]
        team_labels = [labels[i] for i in range(len(labels)) if team_mask[i]] if labels else None
        radar = draw_points_with_labels_on_pitch(
            config=CONFIG,
            xy=team_xy,
            face_color=sv.Color.from_hex(COLORS[color_id]),
            radius=2 if color_id == 5 else 20,
            pitch=radar,
            labels=team_labels
        )

    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
        overlap_ratio_wh=None,
        overlap_wh=(0, 0)
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)

    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)

        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_jersey_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run jersey number recognition with team classification, ensuring unique numbers per team.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on.

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    team_classifier = TeamClassifier(device=device)

    # First, collect crops to fit the team classifier
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        player_detections = detections[detections.class_id == PLAYER_CLASS_ID]
        crops += get_crops(frame, player_detections)

    team_classifier.fit(crops)

    # Now process frames
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    # Initialize jersey number history and assigned numbers per team
    jersey_numbers_history = defaultdict(lambda: defaultdict(int))  # {tracker_id: {jersey_number: count}}
    assigned_jersey_numbers = defaultdict(dict)  # {team_id: {tracker_id: jersey_number}}

    JERSEY_NUMBER_THRESHOLD = 5  # Number of times the same jersey number must be observed

    for frame in frame_generator:
        # Detect players
        player_result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        player_detections = sv.Detections.from_ultralytics(player_result)
        player_detections = tracker.update_with_detections(player_detections)

        # Get player crops
        players = player_detections[player_detections.class_id == PLAYER_CLASS_ID]
        player_crops = get_crops(frame, players)

        # Classify teams
        players_team_id = team_classifier.predict(player_crops)

        labels = []
        for tracker_id, team_id, crop in zip(players.tracker_id, players_team_id, player_crops):
            # Preprocess the crop to improve OCR accuracy
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Apply thresholding to isolate the numbers
            _, thresh_crop = cv2.threshold(
                gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Use OCR to read the jersey number
            jersey_number = pytesseract.image_to_string(
                thresh_crop,
                config='--psm 7 -c tessedit_char_whitelist=0123456789'
            ).strip()

            # Update jersey_numbers_history
            counts = jersey_numbers_history[tracker_id]
            counts[jersey_number] += 1

            # Decide whether to assign jersey number
            assigned_jersey_number = None
            # Check if any jersey number has reached the threshold
            for number, count in counts.items():
                if count >= JERSEY_NUMBER_THRESHOLD and number.isdigit():
                    # Check if the jersey number is already assigned to another player in the same team
                    team_numbers = assigned_jersey_numbers[team_id]
                    if number not in team_numbers.values():
                        assigned_jersey_numbers[team_id][tracker_id] = number
                        assigned_jersey_number = number
                        break  # Exit loop once the jersey number is confirmed

            # Build label
            if tracker_id in assigned_jersey_numbers[team_id]:
                label = assigned_jersey_numbers[team_id][tracker_id]
            else:
                label = ""

            labels.append(label)

        # Annotate frames
        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, players)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(
            annotated_frame, players, labels=labels
        )
        yield annotated_frame


def convert_numpy_types(data: Any) -> Any:
    """
    Recursively convert numpy types in data (dict, list, etc.) to native Python types.

    Args:
        data (Any): Input data that may contain numpy types.

    Returns:
        Any: Data with numpy types converted to Python native types.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()  # Convert numpy arrays to lists
    elif isinstance(data, np.generic):  # Handle numpy scalar types
        return data.item()
    elif isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    return data  # Return native types as is


all_frames = []


def save_all_frames_to_json(json_file_path: str) -> None:
    """
    Save all collected frames to a JSON file with a root object.

    Args:
        json_file_path (str): Path to the output JSON file.
    """
    # Convert all numpy types in the frames data to native Python types
    converted_frames = convert_numpy_types(all_frames)

    # Now save the converted data to the JSON file
    with open(json_file_path, 'w') as json_file:
        json.dump({"frames": converted_frames}, json_file, indent=4)


def estimate_speed(position_history, frame_rate):
    """
    Estimate the speed of a player given their position history.
    :param position_history: List of (x, y) positions.
    :param frame_rate: Frame rate of the video (frames per second).
    :return: Speed in meters per second (m/s).
    """
    if len(position_history) < 2:
        return 0.0  # Not enough data to calculate speed

    # Calculate distance between the last two positions
    delta_x = position_history[-1][0] - position_history[-2][0]
    delta_y = position_history[-1][1] - position_history[-2][1]
    distance = np.sqrt(delta_x**2 + delta_y**2)

    # Time between two frames
    time = 1 / frame_rate

    # Speed = distance / time
    speed = distance / time
    return speed

# Constants for team identification
TEAM_A_ID = 0
TEAM_B_ID = 1

# Initialize counters for possession
possession_counts = {
    TEAM_A_ID: 0,
    TEAM_B_ID: 0
}
total_frames = 0


# Helper function to compute Euclidean distance
def euclidean_distance(pt1: np.ndarray, pt2: np.ndarray) -> float:
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)


def create_radar_frame(frame: np.ndarray,
                       detections: sv.Detections,
                       color_lookup: np.ndarray,
                       labels: list,
                       keypoints: sv.KeyPoints) -> np.ndarray:
    '''
    A helper function that:
      1. Renders the radar using your existing logic (e.g., `render_radar`)
      2. Resizes it
      3. Overlays it onto the current frame
    '''

    h, w, _ = frame.shape
    # Render your radar using existing logic
    radar = render_radar(detections, keypoints, color_lookup, labels)
    radar = sv.resize_image(radar, (w // 2, h // 2))

    radar_h, radar_w, _ = radar.shape
    rect = sv.Rect(
        x=w // 2 - radar_w // 2,
        y=h - radar_h,
        width=radar_w,
        height=radar_h
    )

    # Overlay radar with some opacity
    annotated_frame = sv.draw_image(frame, radar, opacity=0.5, rect=rect)

    return annotated_frame


def run_radar(source_video_path: str, device: str, json_file_path: str) -> Iterator[np.ndarray]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    ball_tracker = BallTracker(buffer_size=20)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE
    )

    # Collect crops for team classification
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Initialize jersey number history and assigned numbers per team
    jersey_numbers_history = defaultdict(lambda: defaultdict(int))
    assigned_jersey_numbers = defaultdict(dict)
    JERSEY_NUMBER_THRESHOLD = 3  # Number of times the same jersey number must be observed

    # Initialize position history for speed calculation
    position_history = defaultdict(list)
    frame_rate = 30  # Assuming a default frame rate (update if known)

    # Start a new frame generator without the stride for actual processing
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    frame_index = 0  # To keep track of the frame number

    # Initialize last_ball_detections to store the last known ball positions
    last_ball_detections = None
    ball_missing_frames = 0
    BALL_MISSING_THRESHOLD = 15

    # We'll store output data here
    all_frames = []

    for frame in frame_generator:
        # ------------------- Pitch Detection -------------------
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # Create a ViewTransformer for this frame using the keypoints
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        transformer = ViewTransformer(
            source=keypoints.xy[0][mask].astype(np.float32),
            target=np.array(CONFIG.vertices)[mask].astype(np.float32)
        )

        # ------------------- Player Detection & Tracking -------------------
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)

        labels = []
        for tracker_id, team_id, crop in zip(players.tracker_id, players_team_id, crops):
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, thresh_crop = cv2.threshold(
                gray_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            jersey_number = pytesseract.image_to_string(
                thresh_crop,
                config='--psm 7 -c tessedit_char_whitelist=0123456789'
            ).strip()

            counts = jersey_numbers_history[tracker_id]
            counts[jersey_number] += 1

            for number, count in counts.items():
                if count >= JERSEY_NUMBER_THRESHOLD and number.isdigit():
                    team_numbers = assigned_jersey_numbers[team_id]
                    if number not in team_numbers.values():
                        assigned_jersey_numbers[team_id][tracker_id] = number
                        break

            if tracker_id in assigned_jersey_numbers[team_id]:
                label = assigned_jersey_numbers[team_id][tracker_id]
            else:
                label = ''
            labels.append(label)

        # Transform player positions into pitch coordinates
        transformed_players_positions = transformer.transform_points(
            players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ) / 100.0

        # Estimate speed for each player
        for tracker_id, position, bbox in zip(
            players.tracker_id, transformed_players_positions, players.xyxy
        ):
            position_history[tracker_id].append(position)
            if len(position_history[tracker_id]) > 10:  # keep last 10 positions
                position_history[tracker_id].pop(0)

            speed = estimate_speed(position_history[tracker_id], frame_rate)
            # Annotate speed on the frame
            if tracker_id is not None:
                x, y = int(bbox[0]), int(bbox[1])
                cv2.putText(
                    frame,
                    f'{speed:.2f} m/s',
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        # ------------------- Goalkeepers & Referees -------------------
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers
        )
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # ------------------- Ball Detection & Tracking -------------------
        ball_result = ball_detection_model(frame, imgsz=640, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_result)
        ball_class_id = 1
        ball_detections = ball_detections[ball_detections.class_id == ball_class_id]
        ball_detections = ball_tracker.update(ball_detections)

        if len(ball_detections) == 0:
            if last_ball_detections is not None and ball_missing_frames < BALL_MISSING_THRESHOLD:
                ball_detections = last_ball_detections
                ball_missing_frames += 1
            else:
                last_ball_detections = None
                ball_missing_frames = 0
        else:
            last_ball_detections = ball_detections
            ball_missing_frames = 0

        if ball_detections.tracker_id is None:
            ball_detections.tracker_id = np.arange(len(ball_detections))

        # Merge all detections for final annotation
        detections = sv.Detections.merge([players, goalkeepers, referees, ball_detections])
        color_lookup = np.array(
            players_team_id.tolist() +
            goalkeepers_team_id.tolist() +
            [REFEREE_CLASS_ID] * len(referees) +
            [BALL_COLOR_ID] * len(ball_detections)
        )
        labels = labels + [''] * (len(goalkeepers) + len(referees) + len(ball_detections))

        transformed_goalkeepers_positions = transformer.transform_points(
            goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ) / 100.0
        transformed_referees_positions = transformer.transform_points(
            referees.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        ) / 100.0

        # ------------------- Ball Position in Pitch Coordinates -------------------
        transformed_ball_positions = transformer.transform_points(
            ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        )  # not scaled by /100 so you can decide how you measure closeness

        # ------------------- Who has the ball this frame? -------------------
        # We'll only consider if we see exactly one ball (common scenario).
        global total_frames, possession_counts
        current_ball_possession_team = None

        if len(transformed_ball_positions) == 1 and len(transformed_players_positions) > 0:
            ball_pos = transformed_ball_positions[0]
            min_dist = float('inf')
            closest_player_team_id = None

            # Find the closest player in pitch space
            for pos, t_id in zip(transformed_players_positions, players_team_id):
                dist = euclidean_distance(ball_pos, pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_player_team_id = t_id

            # Increase possession counter for the closest player's team
            if closest_player_team_id is not None:
                possession_counts[closest_player_team_id] += 1
                current_ball_possession_team = closest_player_team_id

        total_frames += 1

        # ------------------- Logging possession text -------------------
        # By user request: 'team A has it' or 'team B has it'
        if current_ball_possession_team == TEAM_A_ID:
            cv2.putText(
                frame,
                'team A has it',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )
        elif current_ball_possession_team == TEAM_B_ID:
            cv2.putText(
                frame,
                'team B has it',
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

        # ------------------- Calculate possession % for overlay -------------------
        pos_a_percent = 0.0
        pos_b_percent = 0.0
        if total_frames > 0:
            pos_a_percent = (possession_counts[TEAM_A_ID] / total_frames) * 100.0
            pos_b_percent = (possession_counts[TEAM_B_ID] / total_frames) * 100.0

        # Show in top-left corner, below "team A has it" / "team B has it"
        cv2.putText(
            frame,
            f'Team A: {pos_a_percent:.1f}% - Team B: {pos_b_percent:.1f}%',
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # ------------------- Annotate Frame -------------------
        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup
        )
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup
        )

        # Use the helper function to add the radar overlay
        annotated_frame = create_radar_frame(
            annotated_frame, detections, color_lookup, labels, keypoints
        )

        # Save data for JSON
        frame_data = {
            'frame_index': frame_index,
            'players': [
                {
                    'id': int(tracker_id),
                    'team_id': int(team_id),
                    'position': list(pos),
                    'jersey_number': assigned_jersey_numbers[team_id].get(tracker_id, None)
                }
                for tracker_id, team_id, pos in zip(
                    players.tracker_id, players_team_id, transformed_players_positions
                )
            ],
            'goalkeepers': [
                {
                    'id': int(tracker_id),
                    'team_id': int(team_id),
                    'position': list(pos),
                    'jersey_number': '1'
                }
                for tracker_id, team_id, pos in zip(
                    goalkeepers.tracker_id,
                    goalkeepers_team_id,
                    transformed_goalkeepers_positions
                )
            ],
            'referees': [
                {'position': list(pos)}
                for pos in transformed_referees_positions
            ],
            'balls': [
                {'position': list(pos)}
                for pos in (transformed_ball_positions / 100.0)  # scale if you wish
            ]
        }
        all_frames.append(frame_data)

        frame_index += 1
        yield annotated_frame

    # Finally, save everything to JSON
    save_all_frames_to_json(json_file_path)


def main(source_video_path: str, target_video_path: str, device: str, mode: Mode, json_file_path: str) -> None:
    if mode == Mode.PITCH_DETECTION:
        frame_generator = run_pitch_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_DETECTION:
        frame_generator = run_player_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.BALL_DETECTION:
        frame_generator = run_ball_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.PLAYER_TRACKING:
        frame_generator = run_player_tracking(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.TEAM_CLASSIFICATION:
        frame_generator = run_team_classification(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.JERSEY_DETECTION:
        frame_generator = run_jersey_detection(
            source_video_path=source_video_path, device=device)
    elif mode == Mode.RADAR:
        frame_generator = run_radar(
            source_video_path=source_video_path, device=device, json_file_path=json_file_path)
    else:
        raise NotImplementedError(f"Mode {mode} is not implemented.")

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in frame_generator:
            sink.write_frame(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--source_video_path', type=str, required=True)
    parser.add_argument('--target_video_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--mode', type=Mode, default=Mode.PLAYER_DETECTION)
    parser.add_argument('--json_file_path', type=str, default='output.json')
    args = parser.parse_args()
    main(
        source_video_path=args.source_video_path,
        target_video_path=args.target_video_path,
        device=args.device,
        mode=args.mode,
        json_file_path=args.json_file_path  # Pass JSON file path to the main function
    )
