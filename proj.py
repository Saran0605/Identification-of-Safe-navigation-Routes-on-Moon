import cv2
import numpy as np
import os
import random
from queue import PriorityQueue

def create_output_folder(output_folder):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

def process_images(input_folder, output_folder):
    create_output_folder(output_folder)  # Ensure output folder is ready
    
    # Process each image in the folder
    for subdir, _, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(subdir, file)
                print(f"Processing: {image_path}")
                image = cv2.imread(image_path)
                output_image = identify_obstacles_and_find_safe_path(image)
                
                # Save the processed image to the output folder
                output_path = os.path.join(output_folder, file)
                cv2.imwrite(output_path, output_image)
                print(f"Saved processed image to: {output_path}")

def identify_obstacles_and_find_safe_path(image):
    # Convert to grayscale and blur the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect edges to identify obstacles
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find contours to outline obstacles
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    obstacle_mask = np.zeros_like(gray)  # Mask to mark obstacle areas

    for contour in contours:
        # Draw each obstacle on the mask and apply a transparent red overlay
        cv2.drawContours(obstacle_mask, [contour], -1, (255), thickness=cv2.FILLED)
        # Create a red overlay
        overlay = image.copy()
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), thickness=cv2.FILLED)  # Red for obstacles
        # Blend the overlay with the original image (0.3 is the transparency factor)
        cv2.addWeighted(overlay, 0.3, image, 1 - 0.3, 0, image)  # Adjust the second argument for blending strength

    # Define start and end points for the path
    start_point = (10, 10)
    end_point = (image.shape[1] - 10, image.shape[0] - 10)

    # Find a safe path from start to end avoiding obstacles
    safe_path = find_path(obstacle_mask, start_point, end_point)

    # Draw path and mark resources and stops along the path
    mark_path_with_stops_and_resources(image, safe_path, num_stops=5)

    return image

def find_path(obstacle_mask, start, end):
    # Implement A* pathfinding algorithm to find the shortest path in clear areas
    rows, cols = obstacle_mask.shape
    open_set = PriorityQueue()
    open_set.put((0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, end)}

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    while not open_set.empty():
        _, current = open_set.get()
        
        if current == end:
            return reconstruct_path(came_from, current)

        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if obstacle_mask[neighbor] == 255:
                    continue  # Skip if it's an obstacle
                
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end)
                    open_set.put((f_score[neighbor], neighbor))

    return []

def heuristic(a, b):
    # Heuristic for A* (Manhattan distance)
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def reconstruct_path(came_from, current):
    # Reconstruct the path from end to start
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.reverse()
    return path

def mark_path_with_stops_and_resources(image, path, num_stops):
    if not path:
        return

    # Calculate interval for stops
    interval = len(path) // (num_stops + 1)
    stop_points = [path[i * interval] for i in range(1, num_stops + 1)]
    
    # Draw the path with blue lines
    for i in range(len(path) - 1):
        cv2.line(image, path[i], path[i + 1], (255, 0, 0), 2)  # Blue for the path

    # Mark each stop with a green circle and label
    for i, stop in enumerate(stop_points):
        cv2.circle(image, stop, 6, (0, 255, 0), -1)  # Green for stopping points
        cv2.putText(image, f"Stop {i+1}", (stop[0] + 5, stop[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Mark resources (randomly chosen points for demonstration)
    resource_points = [path[random.randint(1, len(path) - 2)] for _ in range(3)]
    for resource in resource_points:
        cv2.circle(image, resource, 8, (0, 255, 255), -1)  # Yellow for resources
        cv2.putText(image, "Resource", (resource[0] + 5, resource[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

# Run the process
input_folder = "D:\\Saran\\Minor project\\python\\Scripts\\moon_imgs"
output_folder = "D:\\Saran\\Minor project\\python\\Scripts\\processed_images"
process_images(input_folder, output_folder)
