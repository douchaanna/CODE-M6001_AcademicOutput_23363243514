"""
Blender rendering for the elaboration measure.

Every model is centred, scaled so its bounding sphere has radius 1, and photographed
with an orthographic camera whose width comes from that sphere. A bounding sphere is
identical from every direction, so each model occupies the same fraction of frame in
all six views and across every model. Nothing clips, and apparent size carries no
information — which matters because elaboration is pixel entropy, and a perspective
camera would let entropy depend partly on how large a model happened to render.
"""
import json
import struct
import subprocess

import numpy as np
import pandas as pd
from pathlib import Path

VIEWS = [("V1", 45, 25), ("V2", 135, 25), ("V3", 225, 25),
         ("V4", 315, 25), ("V5", 45, 60),  ("V6", 225, 60)]
VIEW_NAMES = [f"{name}_{az}az_{el}el" for name, az, el in VIEWS]

RESOLUTION     = 1024
MARGIN         = 1.12   # padding around the bounding sphere
WORLD_STRENGTH = 0.12   # ambient fill
SUN_ENERGY     = 4.5    # key light

BLENDER_PATHS = ["/Applications/Blender.app/Contents/MacOS/blender",
                 "/usr/local/bin/blender", "/usr/bin/blender"]


def find_blender():
    for candidate in BLENDER_PATHS:
        if Path(candidate).exists():
            return candidate
    which = subprocess.run(["which", "blender"], capture_output=True, text=True)
    if which.returncode == 0:
        return which.stdout.strip()
    return None


def glb_part_count(glb_path):
    """Parts recorded inside the GLB itself, for cross-checking against the metadata."""
    with open(glb_path, "rb") as f:
        magic, _, _ = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError("NOT_A_GLB")
        chunk_length, _ = struct.unpack("<II", f.read(8))
        gltf = json.loads(f.read(chunk_length).decode("utf-8").rstrip("\x00"))
    return sum(len(mesh.get("primitives", [])) for mesh in gltf.get("meshes", []))


def build_manifest(primary, render_dir):
    rows = []
    for idea in primary.itertuples():
        if not idea.glb_exists:
            continue
        folder = f"{idea.participant_id}_{idea.idea_name}"
        try:
            parts_in_glb = glb_part_count(idea.glb_path)
        except Exception:
            parts_in_glb = np.nan
        rows.append({
            "folder_name":    folder,
            "render_dir":     str(render_dir / folder),
            "participant":    idea.participant,
            "participant_id": idea.participant_id,
            "condition":      idea.condition,
            "idea_name":      idea.idea_name,
            "glb_path":       idea.glb_path,
            "glb_n_parts":    parts_in_glb,
            "parts_json":     idea.parts_json,
        })
    manifest = pd.DataFrame(rows)
    if manifest.folder_name.duplicated().any():
        raise ValueError("RENDER_FOLDER_COLLISION")
    return manifest


def views_on_disk(render_dir):
    folder = Path(render_dir)
    return sum(1 for view in VIEW_NAMES if (folder / f"{view}.png").exists())


BLENDER_SCRIPT = r'''
import bpy, math, os, sys, csv
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
manifest_path, res, margin, WORLD, SUN = (
    argv[0], int(argv[1]), float(argv[2]), float(argv[3]), float(argv[4]))

VIEWS = [("V1",45,25),("V2",135,25),("V3",225,25),
         ("V4",315,25),("V5",45,60),("V6",225,60)]

def purge():
    for o in list(bpy.data.objects):
        if o.type == "MESH":
            bpy.data.objects.remove(o, do_unlink=True)
    for blocks in (bpy.data.meshes, bpy.data.materials):
        for b in list(blocks):
            blocks.remove(b)

def make_matte():
    m = bpy.data.materials.new("Matte")
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (0.42, 0.50, 0.62, 1.0)
        for key, value in (("Roughness", 0.75), ("Metallic", 0.0), ("Alpha", 1.0)):
            if key in bsdf.inputs:
                bsdf.inputs[key].default_value = value
    return m

def setup_scene():
    sc = bpy.context.scene
    sc.render.resolution_x = sc.render.resolution_y = res
    sc.render.film_transparent = True
    sc.render.image_settings.file_format = "PNG"
    sc.render.image_settings.color_mode = "RGBA"
    if sc.world is None:
        sc.world = bpy.data.worlds.new("World")
    sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (1, 1, 1, 1)
        bg.inputs[1].default_value = WORLD

    cam_data = bpy.data.cameras.get("RenderCam") or bpy.data.cameras.new("RenderCam")
    cam_data.type = "ORTHO"
    cam_data.clip_start, cam_data.clip_end = 0.01, 100.0
    cam = bpy.data.objects.get("RenderCam")
    if cam is None:
        cam = bpy.data.objects.new("RenderCam", cam_data)
        bpy.context.collection.objects.link(cam)
    cam.data = cam_data
    sc.camera = cam

    sun_data = bpy.data.lights.get("Key") or bpy.data.lights.new("Key", type="SUN")
    sun_data.energy = SUN
    sun = bpy.data.objects.get("Key")
    if sun is None:
        sun = bpy.data.objects.new("Key", sun_data)
        bpy.context.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(50), 0, math.radians(35))
    return cam

def bbox_centre_and_radius(obj):
    mw = obj.matrix_world
    pts = [mw @ v.co for v in obj.data.vertices]
    if not pts:
        return Vector((0,0,0)), 0.0
    xs = [p.x for p in pts]; ys = [p.y for p in pts]; zs = [p.z for p in pts]
    centre = Vector(((min(xs)+max(xs))/2, (min(ys)+max(ys))/2, (min(zs)+max(zs))/2))
    return centre, max((p - centre).length for p in pts)

def render_one(glb_path, output_dir, cam, matte):
    os.makedirs(output_dir, exist_ok=True)
    try:
        bpy.ops.import_scene.gltf(filepath=glb_path)
    except Exception as e:
        print("IMPORT_FAILED %s" % e); return 0
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        print("NO_MESH"); return 0

    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.object

    obj.data.materials.clear()
    obj.data.materials.append(matte)

    # Source GLBs arrive in units ranging from about 1 to over 300, so normalising to a
    # unit bounding sphere keeps camera distance and clipping planes predictable.
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    centre, radius = bbox_centre_and_radius(obj)
    if radius <= 0:
        print("DEGENERATE_GEOMETRY"); return 0
    obj.location = -centre
    bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)
    obj.scale = (1.0/radius, 1.0/radius, 1.0/radius)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    _, unit_radius = bbox_centre_and_radius(obj)

    cam.data.ortho_scale = 2.0 * unit_radius * margin
    distance = 10.0
    saved = 0
    for name, az, el in VIEWS:
        a, e = math.radians(az), math.radians(el)
        cam.location = (distance*math.cos(e)*math.cos(a),
                        distance*math.cos(e)*math.sin(a),
                        distance*math.sin(e))
        cam.rotation_euler = (Vector((0,0,0)) - cam.location).to_track_quat("-Z","Y").to_euler()
        bpy.context.scene.render.filepath = os.path.join(
            output_dir, "%s_%daz_%del.png" % (name, az, el))
        bpy.ops.render.render(write_still=True)
        saved += 1
    return saved

cam = setup_scene()
with open(manifest_path, newline="") as f:
    rows = list(csv.DictReader(f))
for n, row in enumerate(rows, 1):
    print("[%d/%d] %s" % (n, len(rows), os.path.basename(row["glb_path"])), flush=True)
    purge()
    print("  %d views" % render_one(row["glb_path"], row["render_dir"], cam, make_matte()), flush=True)
print("RENDER_BATCH_DONE")
'''


def render(manifest, render_dir, timeout=7200):
    """Render every idea in the manifest. Ideas whose six views already exist are left
    alone, so an interrupted run resumes rather than repeating ~45 minutes of work."""
    render_dir = Path(render_dir)
    render_dir.mkdir(parents=True, exist_ok=True)

    manifest = manifest.copy()
    manifest["n_views_saved"] = manifest.render_dir.map(views_on_disk)
    outstanding = manifest[manifest.n_views_saved < len(VIEWS)]
    if outstanding.empty:
        return manifest, "ALL_RENDERED"

    blender = find_blender()
    if blender is None:
        raise RuntimeError("BLENDER_NOT_FOUND")

    script_path = render_dir / "render_batch.py"
    script_path.write_text(BLENDER_SCRIPT)
    batch_path = render_dir / "_batch.csv"
    outstanding.to_csv(batch_path, index=False)

    run = subprocess.run(
        [blender, "-b", "-P", str(script_path), "--", str(batch_path),
         str(RESOLUTION), str(MARGIN), str(WORLD_STRENGTH), str(SUN_ENERGY)],
        capture_output=True, text=True, timeout=timeout)
    log = run.stdout + run.stderr

    manifest["n_views_saved"] = manifest.render_dir.map(views_on_disk)
    status = "RENDER_BATCH_DONE" if "RENDER_BATCH_DONE" in log else "BLENDER_INCOMPLETE"
    if status != "RENDER_BATCH_DONE":
        print(log[-2000:])
    return manifest, status


def frame_report(manifest):
    """Per-image checks that caught the original framing fault: anything touching the
    frame edge, and the intensity range actually used."""
    from PIL import Image
    rows = []
    for idea in manifest.itertuples():
        for view in VIEW_NAMES:
            image_path = Path(idea.render_dir) / f"{view}.png"
            if not image_path.exists():
                continue
            pixels = np.array(Image.open(image_path).convert("RGBA"))
            mask = pixels[..., 3] > 8
            grey = np.array(Image.fromarray(pixels[..., :3]).convert("L"))[pixels[..., 3] > 128]
            rows.append({
                "folder":  idea.folder_name,
                "clipped": bool(mask[0, :].any() or mask[-1, :].any()
                                or mask[:, 0].any() or mask[:, -1].any()),
                "fill":    mask.mean(),
                "lo":      grey.min() if grey.size else np.nan,
                "hi":      grey.max() if grey.size else np.nan,
            })
    return pd.DataFrame(rows)
