#!/usr/bin/env python3
"""Regenerate the small M1 OBJ/MTL/PNG fixtures (Python standard library only).

These are original test assets, not part of the NVIDIA Emerald scene.
Coordinates are metres, Y is up. No model normalization is requested.
"""
from pathlib import Path
import json
import math
import struct
import zlib

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "model/mesh/m1"
SCENES = ROOT / "scene"


def png(path, width, height, sample):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    rows = b"".join(b"\0" + b"".join(bytes(sample(x, y)) for x in range(width)) for y in range(height))
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", width, height, 8, 6, 0, 0, 0)) +
                     chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


class Obj:
    def __init__(self):
        self.lines = ["# Original M1 fixture; metres, +Y up", "mtllib m1.mtl"]
        self.count = 0

    def quad(self, name, material, points, normals=None):
        a, b, c = points[:3]
        u, v = [b[i] - a[i] for i in range(3)], [c[i] - a[i] for i in range(3)]
        n = [u[1]*v[2]-u[2]*v[1], u[2]*v[0]-u[0]*v[2], u[0]*v[1]-u[1]*v[0]]
        scale = math.sqrt(sum(q*q for q in n))
        if scale == 0:
            raise ValueError("degenerate fixture quad")
        n = [q / scale for q in n]
        normals = normals or [n] * 4
        self.lines.extend(["o " + name, "usemtl " + material])
        for p, uv, normal in zip(points, [(0, 0), (1, 0), (1, 1), (0, 1)], normals):
            self.lines.append("v " + " ".join(format(q, ".9g") for q in p))
            self.lines.append("vt " + " ".join(map(str, uv)))
            self.lines.append("vn " + " ".join(format(q, ".9g") for q in normal))
        vtx = [self.count + i + 1 for i in range(4)]
        self.lines.append("f " + " ".join(f"{i}/{i}/{i}" for i in vtx))
        self.count += 4

    def box(self, name, material, lo, hi):
        x, y, z = lo
        X, Y, Z = hi
        faces = [
            [(x,y,z),(x,y,Z),(x,Y,Z),(x,Y,z)],
            [(X,y,Z),(X,y,z),(X,Y,z),(X,Y,Z)],
            [(x,y,z),(X,y,z),(X,y,Z),(x,y,Z)],
            [(x,Y,Z),(X,Y,Z),(X,Y,z),(x,Y,z)],
            [(X,y,z),(x,y,z),(x,Y,z),(X,Y,z)],
            [(x,y,Z),(X,y,Z),(X,Y,Z),(x,Y,Z)],
        ]
        for i, points in enumerate(faces):
            self.quad(f"{name}_{i}", material, points)

    def write(self, path):
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")


def object_desc(name, file, translation=(0,0,0), scale=(1,1,1), rotation=(0,0,0,1)):
    return {"name": name, "type": "mesh", "file": "m1/" + file,
            "TRS": {"transform": translation, "scale": scale, "rotation": rotation},
            "placeCenterBBoxAtOrigin": False, "placeOnGround": False, "normalizeScale": False}


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    SCENES.mkdir(parents=True, exist_ok=True)
    png(ASSETS / "checker_srgb.png", 32, 32,
        lambda x,y: (205,80,35,255) if (x//8+y//8)%2 else (50,150,225,255))
    png(ASSETS / "alpha_srgb.png", 32, 32,
        lambda x,y: (100,205,65,255 if (x//8+y//8)%2 else 0))
    # Legacy importer maps map_Ks to its packed (AO,roughness,metallic) texture.
    # These are LINEAR data textures, not conventional Phong specular maps.
    png(ASSETS / "metal_rm.png", 4, 4, lambda x,y: (255,96,255,255))
    png(ASSETS / "plastic_rm.png", 4, 4, lambda x,y: (255,160,0,255))
    (ASSETS / "m1.mtl").write_text("""# Original M1 materials. Kd/Ke factors are linear RGB.
newmtl Gray
Kd 0.5 0.5 0.5
newmtl Checker
Kd 1 1 1
map_Kd checker_srgb.png
newmtl RoughMetal
Kd 0.85 0.55 0.15
map_Ks metal_rm.png
newmtl RoughPlastic
Kd 0.06 0.25 0.65
map_Ks plastic_rm.png
newmtl ClearGlass
Kd 1 1 1
Ni 1.5
illum 7
newmtl AlphaCard
Kd 1 1 1
map_Kd alpha_srgb.png
newmtl Lamp
Kd 0 0 0
Ke 5 5 5
""", encoding="utf-8")
    courtyard = Obj()
    courtyard.quad("floor", "Checker", [(-3,0,3),(3,0,3),(3,0,-3),(-3,0,-3)])
    courtyard.quad("back", "Gray", [(-3,0,-3),(3,0,-3),(3,3,-3),(-3,3,-3)])
    courtyard.quad("left", "Gray", [(-3,0,3),(-3,0,-3),(-3,3,-3),(-3,3,3)])
    courtyard.box("metal", "RoughMetal", (-1.75,0,-.75), (-.55,1.4,.45))
    courtyard.box("plastic", "RoughPlastic", (.0,0,-2.0), (.9,.9,-1.1))
    courtyard.box("glass", "ClearGlass", (.4,0,.25), (1.25,1.45,1.1))
    courtyard.quad("alpha_card", "AlphaCard", [(1.7,.3,-.8),(2.7,.3,-.8),(2.7,1.8,-.8),(1.7,1.8,-.8)])
    courtyard.write(ASSETS / "courtyard.obj")
    lamp = Obj()
    lamp.quad("area_emitter", "Lamp", [(-.5,0,-.5),(.5,0,-.5),(.5,0,.5),(-.5,0,.5)])
    lamp.write(ASSETS / "lamp.obj")
    objects = [object_desc("Courtyard", "courtyard.obj")]
    lamps = [object_desc("Area A", "lamp.obj", (0,2.9,.2), (1.4,1,.8)),
             object_desc("Area B", "lamp.obj", (1.8,2.3,-1.5), (.7,1,1.1),
                         (0,0,-math.sin(math.radians(10)),math.cos(math.radians(10))))]
    for lighting in ("area", "sun", "combined"):
        for mode in ("mis", "bsdf", "nee"):
            scene = {
                "objects": objects + (lamps if lighting != "sun" else []),
                "camera": {"from": [5,3.5,7], "at": [0,1,0], "up": [0,1,0], "fov": 42,
                           "focalLength": 50, "fValue": 8, "pintDist": 8, "sensitivity": 1},
                "environment": {"file": ""},
                "integrator": {"type": "Path Tracing", "applySpectralRendering": True,
                               "m1Surface": True, "spp": 4, "maxBounce": 12, "isAccumulate": True,
                               "m1DirectMode": mode, "m1RrDepth": 5, "m1Seed": 17,
                               "m1SunEnabled": lighting != "area", "m1SunScale": 1,
                               "m1SunAngularRadiusDeg": .25, "m1SunZenithDeg": 35,
                               "m1SunAzimuthDeg": 65, "m1RayEpsilon": .0001},
                "spectrum": {"m1AssetDirectory": "spectrum/m1"},
                "enableGroundRendering": False,
            }
            (SCENES / f"m1_{lighting}_{mode}.json").write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
    print("Wrote 9 M1 scenes, OBJ/MTL geometry, and four small PNG textures.")


if __name__ == "__main__":
    main()
