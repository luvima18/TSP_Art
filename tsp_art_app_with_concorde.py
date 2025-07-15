import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
from scipy.spatial import KDTree, Voronoi
from skimage.draw import polygon

# ---------- Configuration ----------
st.set_page_config(page_title="TSP Art Generator", layout="centered")
st.title("🎨 TSP Art Generator (Refined Voronoi + Adaptive TSP)")

# ---------- Paths ----------
working_dir = r"C:\\Users\\luvim\\TSP_Art"
tsp_path = os.path.join(working_dir, "tsp_instance.tsp")

# ---------- Step 1: Upload Image ----------
uploaded_file = st.file_uploader("Upload an image (JPEG/PNG)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 0: Human Detection (HOG + SVM)
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    bodies, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
    if len(bodies) > 0:
        largest = max(bodies, key=lambda b: b[2] * b[3])
        x, y, w_box, h_box = largest

        # Resize image to standard height while maintaining aspect ratio
        target_height = 600
        scale_factor = target_height / img.shape[0]
        img = cv2.resize(img, (int(img.shape[1] * scale_factor), target_height))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gamma = 0.5
    look_up = np.array([((i / 255.0) ** gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(gray, look_up)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    eq_img = clahe.apply(gamma_corrected)
    laplacian = cv2.Laplacian(eq_img, cv2.CV_64F)
    enhanced = cv2.convertScaleAbs(eq_img + 0.3 * laplacian)
    pil_img = Image.fromarray(enhanced).filter(ImageFilter.DETAIL).filter(ImageFilter.EDGE_ENHANCE_MORE)
    contrast_strength = st.slider("Adjust Contrast Enhancement", 1.0, 4.0, 3.5, 0.1)
    pil_img = ImageEnhance.Contrast(pil_img).enhance(contrast_strength).convert("L")
    processed = np.array(pil_img)
    h, w = processed.shape

    st.subheader("1. Processed Grayscale Image")
    st.image(processed, channels="GRAY", use_column_width=True)

    # Step 2: Generate weighted stippling map from darkness and edges
    st.subheader("2. Point Sampling")
    edge_map = cv2.Canny(processed, 50, 150) / 255.0
    darkness_weight = ((255 - processed) / 255.0) ** 4
    lightness_weight = (processed / 255.0) ** 2 * 0.05
    combined_map = (0.93 * darkness_weight + 0.05 * edge_map + lightness_weight)

    combined_map = np.clip(combined_map, 0.0, 1.0)
    if combined_map.sum() == 0:
        st.warning("The image has no dark or edge-detected regions to stipple.")
        st.stop()

    combined_map /= combined_map.sum()

    # Adjust number of points based on contrast (automated)
    detail_score = np.std(processed) / 255.0
    estimated_points = int((detail_score * 150000) + 4000)
    num_points = estimated_points

    st.markdown(f"**Number of cities/points used:** {num_points}")

    flat_weights = combined_map.flatten()
    coords = np.array([(x % w, x // w) for x in range(h * w)])
    indices = np.random.choice(np.arange(h * w), size=num_points, replace=False, p=flat_weights)
    points = coords[indices]

    def voronoi_relaxation(points, image, iterations=3):
        new_points = points.copy()
        for _ in range(iterations):
            vor = Voronoi(new_points)
            updated_points = []
            for region_index in vor.point_region:
                region = vor.regions[region_index]
                if not -1 in region and len(region) > 0:
                    polygon_pts = np.array([vor.vertices[i] for i in region])
                    rr, cc = polygon(polygon_pts[:, 1], polygon_pts[:, 0], shape=image.shape)
                    if rr.size > 0:
                        values = image[rr, cc]
                        total_intensity = values.sum()
                        if total_intensity > 0:
                            x_mean = (cc * values).sum() / total_intensity
                            y_mean = (rr * values).sum() / total_intensity
                            updated_points.append([x_mean, y_mean])
            if len(updated_points) > 10:
                new_points = np.array(updated_points)
        return np.clip(new_points, [0, 0], [w - 1, h - 1])

    points = voronoi_relaxation(points, combined_map)

    fig1, ax1 = plt.subplots(figsize=(6, 6))
    ax1.imshow(gray, cmap="gray")
    ax1.scatter(points[:, 0], points[:, 1], s=0.5, c="red")
    ax1.set_axis_off()
    st.pyplot(fig1)

    # Step 3: Greedy TSP drawing with distance-aware fading
    st.markdown("---")
    st.subheader("3. Final TSP Art Output (Greedy TSP with Distance Filter < 30)")

    tree = KDTree(points)
    n = len(points)
    visited = np.zeros(n, dtype=bool)
    tour = [0]
    visited[0] = True

    for _ in range(1, n):
        current_index = tour[-1]
        current_point = points[current_index]
        found = False
        for radius in range(10, 100, 10):
            neighbors = tree.query_ball_point(current_point, r=radius)
            candidates = [i for i in neighbors if not visited[i]]
            if candidates:
                dists = [np.linalg.norm(points[i] - current_point) for i in candidates]
                next_index = candidates[np.argmin(dists)]
                found = True
                break
        if not found:
            unvisited = np.where(~visited)[0]
            dists = np.linalg.norm(points[unvisited] - current_point, axis=1)
            next_index = unvisited[np.argmin(dists)]

        visited[next_index] = True
        tour.append(next_index)

    ordered_coords = points[tour]
    ordered_coords = np.vstack([ordered_coords, ordered_coords[0]])

    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
    for i in range(len(ordered_coords) - 1):
        p1 = ordered_coords[i]
        p2 = ordered_coords[i + 1]
        dist = np.linalg.norm(p2 - p1)
        if dist < 30:
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=0.25)
        else:
            ax2.plot([p1[0], p2[0]], [p1[1], p2[1]], color="gray", linestyle="dashed", linewidth=0.2)

    ax2.set_axis_off()
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.set_xlim(ax2.get_xlim())
    ax2.set_aspect('equal')
    st.pyplot(fig2)

    buf = BytesIO()
    fig2.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    buf.seek(0)
    st.download_button("📎 Download TSP Art", data=buf.read(), file_name="tsp_art.png", mime="image/png")
