import gc
import os
import time
from datetime import datetime

import gradio as gr
import numpy as np
import onnxruntime as ort
import trimesh
from PIL import Image

# --- Open3D is optional: gate mesh logic & UI if not installed ---
try:
    import open3d as o3d
    O3D_AVAILABLE = True
    O3D_IMPORT_ERR = ""
except Exception as _e:
    o3d = None
    O3D_AVAILABLE = False
    O3D_IMPORT_ERR = f"{type(_e).__name__}: {_e}"

# ---------- Model paths ----------
MODEL_MAP = {
    "Small": "./unik3d_vits_pred_518x518.onnx",
    "Base":  "./unik3d_vitb_pred_518x518.onnx",
    "Large": "./unik3d_vitl_pred_518x518.onnx",
}

TARGET_H, TARGET_W = 518, 518  # exporter size (×14 multiple)

# ---------- Core helpers ----------

def load_session(onnx_path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    in_name = sess.get_inputs()[0].name
    out_pts = sess.get_outputs()[0].name
    # 2nd output often confidence; unused here but session supports it.
    return sess, in_name, out_pts, None

def letterbox_image(img_np, target_h=TARGET_H, target_w=TARGET_W):
    """
    Letterbox (preserve aspect ratio) to target size with black padding.
    Returns:
      canvas_img  : (target_h, target_w, 3) uint8
      roi         : (top, left, new_h, new_w) region where the resized image lies
      scale       : uniform scale factor applied
    """
    h0, w0 = img_np.shape[:2]
    if h0 == 0 or w0 == 0:
        raise ValueError("Invalid input image size.")

    scale = min(target_w / w0, target_h / h0)
    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    im = Image.fromarray(img_np)
    im = im.resize((new_w, new_h), resample=Image.BILINEAR)
    resized = np.array(im)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    canvas[top:top+new_h, left:left+new_w] = resized
    return canvas, (top, left, new_h, new_w), scale

def preprocess_for_onnx(img_np):
    """Letterbox to (TARGET_H, TARGET_W), return NCHW float32 0..255 and the canvas/ROI for later crop."""
    canvas_img, roi, _ = letterbox_image(img_np, TARGET_H, TARGET_W)
    x = canvas_img.astype(np.float32)
    x = np.transpose(x, (2, 0, 1))[None, ...]  # NCHW
    return x, canvas_img, roi

def crop_to_roi(arr_hw3, roi):
    """Crop (H,W,3) array to the letterboxed ROI."""
    top, left, new_h, new_w = roi
    return arr_hw3[top:top+new_h, left:left+new_w, :]

def compute_distance_stats(points_hw3):
    """Return (dmin, dmax, d95) on finite distances."""
    v = points_hw3.reshape(-1, 3)
    d = np.linalg.norm(v, axis=1)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return 0.0, 1.0, 1.0
    dmin = float(np.percentile(d, 0.0))
    dmax = float(np.percentile(d, 100.0))
    d95  = float(np.percentile(d, 95.0))
    if not np.isfinite(dmin): dmin = 0.0
    if not np.isfinite(dmax) or dmax <= dmin: dmax = dmin + 1.0
    if not np.isfinite(d95) or d95 <= dmin: d95 = (dmin + dmax) * 0.5
    return dmin, dmax, d95

def orient_vertices(vertices, front_view=True):
    """
    Fixed transform for glTF conventions: +Y up, viewer looks toward -Z.
    'Front & upright' => flip Y (upright) and Z (front-facing).
    """
    if not front_view:
        return vertices
    v = vertices.copy()
    v[:, 1] *= -1.0  # Y -> -Y (upright)
    v[:, 2] *= -1.0  # Z -> -Z (front-facing)
    return v

def filter_points_for_view(points_hw3, image_hw3, max_distance=None, mask_black_bg=False, front_view=True):
    """
    Apply the SAME filtering/orientation used by the point view.
    Returns (vertices Nx3 float32, colors Nx3 uint8)
    """
    img = image_hw3.astype(np.uint8)
    pts = points_hw3.astype(np.float32)

    vertices = pts.reshape(-1, 3)
    colors   = img.reshape(-1, 3)

    keep = np.isfinite(vertices).all(axis=1)
    if max_distance is not None and np.isfinite(max_distance):
        d = np.linalg.norm(vertices, axis=1)
        keep = keep & (d <= float(max_distance))

    if mask_black_bg:
        keep = keep & (colors.sum(axis=1) >= 16)

    vertices = vertices[keep]
    colors   = colors[keep]

    # Apply the front/upright orientation so mesh matches the visible cloud
    vertices = orient_vertices(vertices, front_view=front_view)

    return vertices, colors

def predictions_to_glb_numpy(predictions, front_view=True, max_distance=None, mask_black_bg=False):
    """Point-cloud viewer GLB from filtered/oriented points."""
    vertices, colors = filter_points_for_view(
        predictions["points"], predictions["image"],
        max_distance=max_distance, mask_black_bg=mask_black_bg, front_view=front_view
    )
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=vertices, colors=colors))
    return scene

def run_onnx_once(image_path, onnx_path):
    """
    Run ONNX with letterbox input (AR preserved), then crop outputs to ROI so the
    point cloud matches the original image aspect ratio.
    """
    img = np.array(Image.open(image_path))
    sess, in_name, out_pts, _ = load_session(onnx_path)

    x, canvas_img, roi = preprocess_for_onnx(img)  # letterbox
    pts_3d = sess.run([out_pts], {in_name: x})[0]  # (1,3,H,W)
    pts_3d = np.transpose(pts_3d[0], (1, 2, 0))    # (H,W,3) on letterboxed canvas

    # Crop both points and image to active ROI (undo padding; AR preserved)
    pts_3d = crop_to_roi(pts_3d, roi)
    disp_img = crop_to_roi(canvas_img, roi)

    dmin, dmax, d95 = compute_distance_stats(pts_3d)
    return {"points": pts_3d, "image": disp_img}, (dmin, dmax, d95)

# ---------- Mesh normals fix (by "front" viewpoint) ----------
if O3D_AVAILABLE:
    def fix_mesh_normals_to_front_view(mesh, camera_pos=(0.0, 0.0, 10.0)):
        """
        Make triangle winding consistent per connected component, then orient each component
        so its average normal faces the camera at `camera_pos` (front view).
        """
        # Clean topology
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
        if len(mesh.triangles) == 0:
            return mesh

        # Consistent winding
        mesh.orient_triangles()

        # Per-triangle normals & centroids
        mesh.compute_triangle_normals()
        tris = np.asarray(mesh.triangles)
        verts = np.asarray(mesh.vertices)
        if tris.size == 0 or verts.size == 0:
            return mesh
        tri_normals = np.asarray(mesh.triangle_normals)      # (T,3)
        tri_centroids = verts[tris].mean(axis=1)             # (T,3)

        # Connected components (lists in Open3D 0.18)
        labels, cluster_n_triangles, _ = mesh.cluster_connected_triangles()
        labels = np.asarray(labels)
        n_clusters = len(cluster_n_triangles) if hasattr(cluster_n_triangles, '__len__') else int(cluster_n_triangles)

        cam = np.array(camera_pos, dtype=np.float64)
        tris_np = tris.copy()

        # Flip whole components that face away from the camera
        for cid in range(n_clusters):
            idx = np.where(labels == cid)[0]
            if idx.size == 0:
                continue
            n_mean = tri_normals[idx].mean(axis=0)
            c_mean = tri_centroids[idx].mean(axis=0)
            view_vec = c_mean - cam    # camera -> component
            if np.dot(n_mean, view_vec) > 0:
                tris_np[idx] = tris_np[idx][:, [0, 2, 1]]

        mesh.triangles = o3d.utility.Vector3iVector(tris_np)
        mesh.compute_triangle_normals()
        mesh.compute_vertex_normals()
        return mesh
else:
    def fix_mesh_normals_to_front_view(mesh, camera_pos=(0.0, 0.0, 10.0)):
        return mesh  # no-op if Open3D is unavailable

# ---------- Gradio glue ----------

def update_gallery_on_upload(input_image):
    if input_image is None:
        return None, "Please upload an image first."
    gc.collect()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tmpdir = os.environ.get("TMPDIR", "/tmp")
    target_dir = os.path.join(tmpdir, f"unik3d_onnx_{ts}")
    os.makedirs(target_dir, exist_ok=True)
    dst = os.path.join(target_dir, "image.png")
    Image.fromarray(input_image).save(dst)
    return target_dir, "Upload complete. Click **Run UniK3D (ONNX)**."

def run_pipeline(target_dir, model_size, front_view, mask_black_bg):
    if not os.path.isdir(target_dir) or target_dir == "None":
        return None, "No valid target directory. Please upload.", None, gr.update()

    image_names = [x for x in os.listdir(target_dir) if x.lower().endswith(".png")]
    if not image_names:
        return None, "No image found in working folder.", None, gr.update()
    image_path = os.path.join(target_dir, sorted(image_names)[-1])

    onnx_path = MODEL_MAP[model_size]
    if not os.path.exists(onnx_path):
        return None, f"ONNX not found: {onnx_path}", None, gr.update()

    start = time.time()
    gc.collect()

    results, (dmin, dmax, d95) = run_onnx_once(image_path, onnx_path)

    # Save predictions for re-filtering and mesh recon
    npz_path = os.path.join(target_dir, "predictions.npz")
    np.savez(npz_path, points=results["points"], image=results["image"])

    # First render with a conservative default (95th percentile)
    scene = predictions_to_glb_numpy(
        results,
        front_view=front_view,
        max_distance=d95,
        mask_black_bg=mask_black_bg,
    )
    glb_path = os.path.join(target_dir, "glbscene.glb")
    scene.export(file_obj=glb_path)

    took = time.time() - start
    log = (
        f"Success (ONNX). Inference & export took **{took:.2f}s**.\n\n"
        f"Distance range: min **{dmin:.3f}**, max **{dmax:.3f}**. "
        f"Slider initialized at **{d95:.3f}** (95th percentile)."
    )

    # Configure distance slider for the current result
    slider_update = gr.update(minimum=float(dmin), maximum=float(dmax), value=float(d95), visible=True)

    return glb_path, log, npz_path, slider_update

def refilter_with_slider(npz_path, max_distance, front_view, mask_black_bg):
    if not npz_path or not os.path.exists(npz_path):
        return None, "No predictions available. Please run the model first."
    data = np.load(npz_path)
    results = {"points": data["points"], "image": data["image"]}
    scene = predictions_to_glb_numpy(
        results,
        front_view=front_view,
        max_distance=float(max_distance) if max_distance is not None else None,
        mask_black_bg=mask_black_bg,
    )
    out_glb = os.path.join(os.path.dirname(npz_path), f"glbscene_maxd_{float(max_distance):.3f}.glb")
    scene.export(file_obj=out_glb)
    return out_glb, f"Filtered points with distance ≤ **{float(max_distance):.3f}**."

# ---------- Mesh reconstruction (Ball-Pivoting path) ----------
if O3D_AVAILABLE:
    def reconstruct_mesh_bpa(
        npz_path,
        max_distance, front_view, mask_black_bg,
        density_cut_pct,
        bpa_base_radius_factor,
        bpa_max_radius_factor,
        normals_radius_factor,
        normals_consistency_k,
        fix_normals_to_view,
    ):
        """
        Build mesh from the EXACT points visible in the point viewer.
        Noise control: drop the bottom 'density_cut_pct' percentile by point density (NN distance).
        BPA radii: geometric ladder starting at base = median_NN * bpa_base_radius_factor,
                   doubling up to base*bpa_max_radius_factor.
        Normals: estimated with radius = median_NN * normals_radius_factor, then globally oriented with
                 orient_normals_consistent_tangent_plane(k). Optionally flip components toward camera.
        """
        if not npz_path or not os.path.exists(npz_path):
            return None, "No predictions available. Please run the model first."
        data = np.load(npz_path)
        pts_hw3, img_hw3 = data["points"], data["image"]

        # 1) Filter/orient points to match what's visible
        vertices, colors = filter_points_for_view(
            pts_hw3, img_hw3,
            max_distance=float(max_distance) if max_distance is not None else None,
            mask_black_bg=mask_black_bg,
            front_view=front_view,
        )
        if vertices.shape[0] < 50:
            return None, "Too few points after filtering to reconstruct a mesh."

        # 2) Build Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector((colors.astype(np.float32) / 255.0))

        # 3) Compute NN distances for density metric
        nn_all = np.asarray(pcd.compute_nearest_neighbor_distance())
        if nn_all.size != len(pcd.points):
            return None, "Failed to compute point densities."
        med_nn = float(np.median(nn_all)) if np.isfinite(nn_all).any() else 1e-3

        # 4) Density cut (remove sparsest X%)
        if density_cut_pct > 0:
            thresh = np.percentile(nn_all, 100.0 - float(density_cut_pct))  # keep densest (100 - pct)%
            keep = nn_all <= thresh
            pcd = pcd.select_by_index(np.where(keep)[0])
            if len(pcd.points) < 50:
                return None, "Too few points after density cut."

        # Recompute median NN after cut for scale-adaptive radii/normals
        nn_cut = np.asarray(pcd.compute_nearest_neighbor_distance())
        med_nn = float(np.median(nn_cut)) if np.isfinite(nn_cut).any() else med_nn

        # --- 5) Normals: estimate, then force global consistency ---
        nrad = max(med_nn * float(normals_radius_factor), 1e-6)
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=nrad, max_nn=64)
        )
        try:
            pcd.orient_normals_consistent_tangent_plane(int(normals_consistency_k))
        except Exception:
            pcd.orient_normals_consistent_tangent_plane(int(max(10, min(50, normals_consistency_k))))
        if front_view:
            pcd.orient_normals_towards_camera_location(
                camera_location=np.array([0.0, 0.0, 10.0], dtype=np.float64)
            )

        # --- 6) Ball-Pivoting radii ladder ---
        base_r = max(med_nn * float(bpa_base_radius_factor), 1e-6)
        max_fac = max(float(bpa_max_radius_factor), 1.0)
        rlist = [base_r]
        while rlist[-1] * 2.0 <= base_r * max_fac + 1e-12:
            rlist.append(rlist[-1] * 2.0)
        radii = o3d.utility.DoubleVector(rlist)
        if len(radii) == 0 or base_r <= 0:
            return None, "Invalid BPA radii. Try increasing the base radius factor."

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, radii)

        # --- Clean topology & weld tiny cracks (vertex merging) ---
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        # Weld vertices that are extremely close (cracks from numeric noise)
        try:
            weld_eps = max(0.25 * base_r, 1e-6)    # tweak 0.2–0.5 * base_r if needed
            mesh.merge_close_vertices(weld_eps)
            mesh.remove_duplicated_triangles()
            mesh.remove_non_manifold_edges()
        except AttributeError:
            # Older Open3D fallback
            mesh = mesh.simplify_vertex_clustering(
                voxel_size=max(0.25 * base_r, 1e-6),
                contraction=o3d.geometry.SimplificationContraction.Average
            )

        # 7) Transfer colors to mesh vertices with k-NN blending (reduces seam lines)
        k_color = 5
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        vtx = np.asarray(mesh.vertices)
        pcd_colors = np.asarray(pcd.colors)
        out_colors = np.zeros((len(vtx), 3), dtype=np.float32)
        for i, v in enumerate(vtx):
            cnt, idx, dist = pcd_tree.search_knn_vector_3d(v, k_color)
            if cnt == 0:
                out_colors[i] = (1.0, 1.0, 1.0)
                continue
            idx = np.array(idx[:cnt], dtype=int)
            dist = np.array(dist[:cnt], dtype=np.float32)
            w = 1.0 / (1e-8 + dist)
            w = w / np.sum(w)
            out_colors[i] = np.sum(pcd_colors[idx] * w[:, None], axis=0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(out_colors)

        # 8) Optional: fix normals/winding per connected component to face front view
        if fix_normals_to_view:
            mesh = fix_mesh_normals_to_front_view(mesh, camera_pos=(0.0, 0.0, 10.0))

        # 9) Export mesh GLB
        out_glb = os.path.join(
            os.path.dirname(npz_path),
            f"mesh_bpa_b{bpa_base_radius_factor:g}_m{bpa_max_radius_factor:g}_n{normals_radius_factor:g}_d{density_cut_pct:g}"
            f"_k{int(normals_consistency_k)}{'_normfix' if fix_normals_to_view else ''}.glb"
        )
        o3d.io.write_triangle_mesh(out_glb, mesh, write_triangle_uvs=False)

        msg = (f"Mesh reconstructed (BPA). Used {len(pcd.points)} points. "
               f"Radii: {', '.join(f'{r:.4g}' for r in rlist)}; normals radius ~ {nrad:.4g}; "
               f"consistency k = {int(normals_consistency_k)}."
               f"{' (Normals fixed to front view)' if fix_normals_to_view else ''}")
        return out_glb, msg
else:
    def reconstruct_mesh_bpa(*args, **kwargs):
        return (None,
                "Mesh reconstruction is unavailable: Open3D is not installed.\n"
                "Install with:  pip install open3d==0.18.0\n"
                f"(Import error was: {O3D_IMPORT_ERR})")

# ---------- Launch UI ----------

if __name__ == "__main__":
    theme = gr.themes.Citrus()
    theme.set(
        checkbox_label_background_fill_selected="*button_primary_background_fill",
        checkbox_label_text_color_selected="*button_primary_text_color",
    )

    with gr.Blocks(theme=theme) as demo:
        gr.HTML("<h1>UniK3D (ONNX) — AR-Preserving Front View, Distance Filter & Ball-Pivoting Mesh</h1>")

        # Hidden state
        target_dir_output = gr.Textbox(label="Target Dir", visible=False, value="None")
        reconstruction_npy = gr.File(label="Predictions (.npz)", type="filepath", visible=False)

        with gr.Row():
            with gr.Column():
                model_size = gr.Dropdown(
                    choices=["Large", "Base", "Small"], label="Model Variant", value="Large"
                )
                front_view = gr.Checkbox(label="Snap to front & upright", value=True)
                mask_black_bg = gr.Checkbox(label="Filter black background", value=False)

                # Distance filter slider (configured after inference)
                distance_slider = gr.Slider(
                    label="Max distance (filter far points)",
                    minimum=0.0, maximum=1.0, step=0.01, value=1.0,
                    visible=True, interactive=True
                )

                # Mesh controls (visible only if Open3D is available)
                density_cut = gr.Slider(
                    label="Density cut (drop lowest-density points, %)",
                    minimum=0.0, maximum=20.0, step=0.5, value=2.0,
                    visible=O3D_AVAILABLE
                )
                bpa_base_radius_factor = gr.Slider(
                    label="BPA base radius (× median NN)",
                    minimum=0.5, maximum=3.0, step=0.05, value=1.2,
                    info="Increase to bridge gaps; decrease for fine details.",
                    visible=O3D_AVAILABLE
                )
                bpa_max_radius_factor = gr.Slider(
                    label="BPA max radius factor (× base)",
                    minimum=1.0, maximum=12.0, step=0.5, value=6.0,
                    info="Largest ball is base × this. Larger values fill wider gaps.",
                    visible=O3D_AVAILABLE
                )
                normals_radius_factor = gr.Slider(
                    label="Normals neighborhood (× median NN)",
                    minimum=1.0, maximum=8.0, step=0.25, value=4.0,
                    info="Larger neighborhood stabilizes normals for smoother BPA.",
                    visible=O3D_AVAILABLE
                )
                normals_consistency_k = gr.Slider(
                    label="Normals consistency (k neighbors)",
                    minimum=10, maximum=200, step=5, value=80,
                    info="Propagates consistent point-normal orientation across the cloud before BPA.",
                    visible=O3D_AVAILABLE
                )
                fix_normals_cb = gr.Checkbox(
                    label="Fix mesh normals to front view", value=True,
                    info="Make triangle winding consistent per component, then flip components toward the viewer.",
                    visible=O3D_AVAILABLE
                )

                mesh_btn = gr.Button("Reconstruct Mesh (Ball-Pivoting)", variant="secondary",
                                     visible=O3D_AVAILABLE)

                if not O3D_AVAILABLE:
                    gr.Markdown(
                        f"> ⚠️ Mesh reconstruction is disabled because Open3D isn’t installed. "
                        f"Install it with `pip install open3d==0.18.0`.\n\n"
                        f"_Importer said:_ `{O3D_IMPORT_ERR}`"
                    )

            with gr.Column(scale=1):
                input_image = gr.Image(label="Upload Image")
                gr.Markdown("**3D Estimation (ONNX)**")
                log_output = gr.Markdown("Upload an image and click **Run UniK3D (ONNX)**.")

            with gr.Column(scale=2):
                reconstruction_output = gr.Model3D(label="Point Cloud", height=520, zoom_speed=0.5, pan_speed=0.5)
                mesh_output = gr.Model3D(label="Mesh (BPA)", height=520, zoom_speed=0.5, pan_speed=0.5,
                                         visible=O3D_AVAILABLE)
                with gr.Row():
                    submit_btn = gr.Button("Run UniK3D (ONNX)", scale=1, variant="primary")
                    clear_btn = gr.ClearButton(
                        [input_image, reconstruction_output, mesh_output, log_output, target_dir_output, reconstruction_npy],
                        scale=1
                    )

        # Upload hook
        input_image.change(
            fn=update_gallery_on_upload,
            inputs=[input_image],
            outputs=[target_dir_output, log_output]
        )

        # Run model
        submit_btn.click(
            fn=lambda: None, inputs=[], outputs=[reconstruction_output]
        ).then(
            fn=lambda: "Running ONNX…", inputs=[], outputs=[log_output]
        ).then(
            fn=run_pipeline,
            inputs=[target_dir_output, model_size, front_view, mask_black_bg],
            outputs=[reconstruction_output, log_output, reconstruction_npy, distance_slider],
        )

        # Interactive re-filtering without re-running ONNX
        distance_slider.release(
            fn=refilter_with_slider,
            inputs=[reconstruction_npy, distance_slider, front_view, mask_black_bg],
            outputs=[reconstruction_output, log_output]
        )
        front_view.change(
            fn=refilter_with_slider,
            inputs=[reconstruction_npy, distance_slider, front_view, mask_black_bg],
            outputs=[reconstruction_output, log_output]
        )
        mask_black_bg.change(
            fn=refilter_with_slider,
            inputs=[reconstruction_npy, distance_slider, front_view, mask_black_bg],
            outputs=[reconstruction_output, log_output]
        )

        # Mesh reconstruction (uses CURRENT slider/toggles)
        if O3D_AVAILABLE:
            mesh_btn.click(
                fn=reconstruct_mesh_bpa,
                inputs=[
                    reconstruction_npy,
                    distance_slider, front_view, mask_black_bg,
                    density_cut,
                    bpa_base_radius_factor, bpa_max_radius_factor, normals_radius_factor,
                    normals_consistency_k,
                    fix_normals_cb
                ],
                outputs=[mesh_output, log_output],
            )

        demo.launch(show_error=True)
