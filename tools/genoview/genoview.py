"""View or record BVH motion on the Geno character, side by side, with audio.

    python tools/genoview/genoview.py a.bvh [b.bvh ...] [--labels A B] [--colors "#ffa100" "#3d3d3d"]
                                      [--audio speech.wav] [--record out.mp4]

ZEGGS-skeleton BVHs (zeggs-generate output) are retargeted to Geno on the fly; Geno BVHs are shown as they are.
Without --audio, a .wav next to the first BVH with the same name is used.

Window: Ctrl + left drag rotates, Ctrl + right drag pans, wheel zooms. Space plays/pauses, Left/Right step a frame,
Home/End jump, Up/Down change speed, F5 reloads, and dropping .bvh files onto the window replaces the characters.

Rendering and the inspection tools come from GenoViewPython by Daniel Holden (MIT); see genoview_lib.py.
"""
import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from genoview_lib import *  # noqa: E402,F401,F403  (raylib API and GenoView helpers)
from raylib import ffi as rl_ffi  # noqa: E402  (raylib's own FFI, which knows struct Color)
from zeggs.anim import bvh  # noqa: E402  (after the star import, which also exports GenoView's bvh module)
from zeggs.retarget import PARAMS, retarget  # noqa: E402

COLORS = ["#ffa100", "#3d3d3d", "#00a087", "#b07aa1"]  # orange, charcoal, teal, mauve

# The Geno mesh is free for non-commercial research use and is not redistributed here: fetch it from its source.
GENO_URL = ("https://raw.githubusercontent.com/orangeduck/GenoViewPython/"
            "3cee22df7d958860d0ceba2c949c0d52e4e7d9b3/resources/Geno.bin")
GENO_SHA256 = "79fab76fcd5f394462d64ef63aa41f4b823b038fba534cf1ef9b43ac868c03cf"


def resource(name):
    return os.path.join(HERE, "resources", name)


def ensure_geno():
    path = resource("Geno.bin")
    if not os.path.exists(path):
        print(f"Downloading the Geno character (non-commercial research use) from\n  {GENO_URL}")
        data = urllib.request.urlopen(GENO_URL, timeout=60).read()
        if hashlib.sha256(data).hexdigest() != GENO_SHA256:
            raise RuntimeError("Geno.bin checksum mismatch; not saving it")
        with open(path, "wb") as f:
            f.write(data)
    return path


def hex_rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def load_motion(path, lane_x, geno_offsets):
    """GenoView animation dict (LoadAnimation layout) for a BVH, with the first-frame hips moved to (lane_x, ., 0)."""
    data = bvh.load(path)
    if np.abs(data["offsets"][1:] - geno_offsets[1:]).max() > 0.5:  # not Geno's proportions: a ZEGGS skeleton
        data = retarget(data)
    lpos = 0.01 * data["positions"].astype(np.float32)
    lrot = quat.unroll(quat.from_euler(np.radians(data["rotations"]), order=data["order"]))
    grot, gpos = quat.fk(lrot, lpos, data["parents"])
    gpos = gpos - gpos[0, 0] * np.array([1, 0, 1]) + np.array([lane_x, 0, 0])
    frame_time = data["frametime"]
    names = list(data["names"])
    feet = [(label, names.index(name)) for label, name in
            [("L Toe", "LeftToeBase"), ("R Toe", "RightToeBase"), ("L Foot", "LeftFoot"), ("R Foot", "RightFoot")]]
    return dict(path=path, name=os.path.basename(path), parents=data["parents"], globalRotations=grot,
                globalPositions=gpos, frameTime=frame_time,
                footJoints=feet, footContacts={j: ComputeFootContacts(gpos, j, frame_time) for _, j in feet})


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("bvh", nargs="+", help="BVH files (ZEGGS or Geno skeleton)")
    p.add_argument("--labels", nargs="*", help="One label per BVH, drawn on the floor in front of each character")
    p.add_argument("--colors", nargs="*", help="One hex colour per character (default: orange, charcoal, teal, mauve)")
    p.add_argument("--audio", help="WAV to play and mux into recordings (default: <first bvh>.wav if it exists)")
    p.add_argument("--record", help="Write an MP4 (or GIF) instead of opening a window")
    p.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    p.add_argument("--end", type=float, help="End time in seconds (default: shortest clip)")
    p.add_argument("--fps", type=int, default=30, help="Recording frame rate")
    p.add_argument("--size", type=int, nargs=2, default=(1280, 720), metavar=("W", "H"))
    p.add_argument("--spacing", type=float, default=1.5, help="Metres between characters")
    p.add_argument("--azimuth", type=float, default=0.0, help="Camera azimuth (radians, 0 = front)")
    p.add_argument("--altitude", type=float, default=0.25, help="Camera altitude (radians)")
    p.add_argument("--distance", type=float, help="Camera distance (default fits all characters)")
    p.add_argument("--foot-markers", action="store_true", help="Draw foot-slide markers (recordings; toggle in window)")
    p.add_argument("--onion", action="store_true", help="Draw onion-skin ghosts (recordings; toggle in window)")
    return p.parse_args()


def main():
    args = parse_args()
    width, height = args.size
    if args.audio is None and os.path.exists(os.path.splitext(args.bvh[0])[0] + ".wav"):
        args.audio = os.path.splitext(args.bvh[0])[0] + ".wav"
    # Colour structs the window's colour picker edits in place; defaults fill in for characters without --colors.
    hexes = list(args.colors or []) + COLORS[len(args.colors or []):]
    colors = [rl_ffi.new("Color *", (*hex_rgb(c), 255)) for c in hexes]
    geno_offsets = np.load(PARAMS)["offsets"]
    paths, labels = list(args.bvh), list(args.labels or [])

    def load_all(paths):
        lanes = (np.arange(len(paths)) - (len(paths) - 1) / 2) * args.spacing
        return [load_motion(p, x, geno_offsets) for p, x in zip(paths, lanes)]

    anims = load_all(paths)
    geno_path = ensure_geno()

    SetConfigFlags(FLAG_WINDOW_HIDDEN if args.record else FLAG_VSYNC_HINT)
    InitWindow(width, height, b"GenoView")
    SetTargetFPS(60)

    def shader(vs, fs):
        return LoadShaderVersioned(resource(vs), resource(fs))

    shadowShader = shader("shadow.vs", "shadow.fs")
    skinnedShadowShader = shader("skinnedShadow.vs", "shadow.fs")
    skinnedBasicShader = shader("skinnedBasic.vs", "basic.fs")
    basicShader = shader("basic.vs", "basic.fs")
    lightingShader = shader("post.vs", "lighting.fs")
    ssaoShader = shader("post.vs", "ssao.fs")
    blurShader = shader("post.vs", "blur.fs")
    fxaaShader = shader("post.vs", "fxaa.fs")
    ghostShader = shader("skinnedBasic.vs", "ghost.fs")
    ghostColorLoc = GetShaderLocation(ghostShader, b"ghostColor")

    def loc(s, name):
        return GetShaderLocation(s, name.encode())

    def set_float(s, name, value):
        ptr = ffi.new("float*")
        ptr[0] = value
        SetShaderValue(s, loc(s, name), ptr, SHADER_UNIFORM_FLOAT)

    def set_vec(s, name, value, kind):
        SetShaderValue(s, loc(s, name), ffi.addressof(value), kind)

    groundModel = LoadModelFromMesh(GenMeshPlane(40.0, 40.0, 10, 10))
    groundPosition = Vector3(0.0, -0.01, 0.0)
    genoModel = LoadGenoModel(geno_path.encode())
    genoPosition = Vector3(0.0, 0.0, 0.0)
    bindPos, bindRot = GetModelBindPoseAsNumpyArrays(genoModel)

    camera = Camera()
    camera.azimuth, camera.altitude = args.azimuth, args.altitude
    camera.distance = args.distance or 2.5 + args.spacing * len(anims)
    rlSetClipPlanes(0.01, 50.0)

    lightDir = Vector3Normalize(Vector3(0.35, -1.0, -0.35))
    shadowLight = ShadowLight()
    shadowLight.up = Vector3(0.0, 1.0, 0.0)
    shadowLight.near, shadowLight.far = 0.01, 10.0
    shadowMap = LoadShadowMap(1024, 1024)
    shadowInvResolution = Vector2(1.0 / 1024, 1.0 / 1024)

    gbuffer = LoadGBuffer(width, height)
    lighted = LoadRenderTexture(width, height)
    ssaoFront = LoadRenderTexture(width, height)
    ssaoBack = LoadRenderTexture(width, height)
    final = LoadRenderTexture(width, height)

    toggles = {name: ffi.new("bool*") for name in ("skeleton", "transforms", "foot", "onion", "colors")}
    toggles["foot"][0], toggles["onion"][0] = args.foot_markers, args.onion

    def draw_characters(frame, s):
        genoModel.materials[0].shader = s
        for i, a in enumerate(anims):
            f = min(frame, len(a["globalPositions"]) - 1)
            UpdateModelPoseFromNumpyArrays(genoModel, bindPos, bindRot, a["globalPositions"][f], a["globalRotations"][f])
            DrawModel(genoModel, genoPosition, 1.0, colors[i % len(colors)][0])

    def render(frame, interactive):
        """Draw `frame` into `final` (scene, overlays and labels); returns the per-foot slide readouts."""
        hips = np.mean([a["globalPositions"][min(frame, len(a["globalPositions"]) - 1), 0] for a in anims], axis=0)
        target = Vector3(float(hips[0]), 0.0, float(hips[2]))
        shadowLight.width = shadowLight.height = 5.0 + args.spacing * len(anims)
        shadowLight.target = target
        shadowLight.position = Vector3Add(target, Vector3Scale(lightDir, -5.0))
        drag = interactive and IsKeyDown(KEY_LEFT_CONTROL)
        delta = GetMouseDelta()
        camera.update(
            Vector3(float(hips[0]), 0.75, float(hips[2])),
            delta.x if drag and IsMouseButtonDown(0) else 0.0,
            delta.y if drag and IsMouseButtonDown(0) else 0.0,
            delta.x if drag and IsMouseButtonDown(1) else 0.0,
            delta.y if drag and IsMouseButtonDown(1) else 0.0,
            GetMouseWheelMove() if interactive else 0.0,
            GetFrameTime() if interactive else 0.0)

        rlDisableColorBlend()

        BeginShadowMap(shadowMap, shadowLight)
        lightViewProj = MatrixMultiply(rlGetMatrixModelview(), rlGetMatrixProjection())
        lightNear, lightFar = rlGetCullDistanceNear(), rlGetCullDistanceFar()
        for s in (shadowShader, skinnedShadowShader):
            set_float(s, "lightClipNear", lightNear)
            set_float(s, "lightClipFar", lightFar)
        groundModel.materials[0].shader = shadowShader
        DrawModel(groundModel, groundPosition, 1.0, WHITE)
        draw_characters(frame, skinnedShadowShader)
        EndShadowMap()

        BeginGBuffer(gbuffer, camera.cam3d)
        camView, camProj = rlGetMatrixModelview(), rlGetMatrixProjection()
        camInvProj = MatrixInvert(camProj)
        camInvViewProj = MatrixInvert(MatrixMultiply(camView, camProj))
        camNear, camFar = rlGetCullDistanceNear(), rlGetCullDistanceFar()
        for s in (basicShader, skinnedBasicShader):
            set_float(s, "specularity", 0.5)
            set_float(s, "glossiness", 10.0)
            set_float(s, "camClipNear", camNear)
            set_float(s, "camClipFar", camFar)
        groundModel.materials[0].shader = basicShader
        DrawModel(groundModel, groundPosition, 1.0, Color(190, 190, 190, 255))
        draw_characters(frame, skinnedBasicShader)
        EndGBuffer(width, height)

        BeginTextureMode(ssaoFront)
        BeginShaderMode(ssaoShader)
        SetShaderValueTexture(ssaoShader, loc(ssaoShader, "gbufferNormal"), gbuffer.normal)
        SetShaderValueTexture(ssaoShader, loc(ssaoShader, "gbufferDepth"), gbuffer.depth)
        SetShaderValueMatrix(ssaoShader, loc(ssaoShader, "camView"), camView)
        SetShaderValueMatrix(ssaoShader, loc(ssaoShader, "camProj"), camProj)
        SetShaderValueMatrix(ssaoShader, loc(ssaoShader, "camInvProj"), camInvProj)
        SetShaderValueMatrix(ssaoShader, loc(ssaoShader, "camInvViewProj"), camInvViewProj)
        SetShaderValueMatrix(ssaoShader, loc(ssaoShader, "lightViewProj"), lightViewProj)
        SetShaderValueShadowMap(ssaoShader, loc(ssaoShader, "shadowMap"), shadowMap)
        set_vec(ssaoShader, "shadowInvResolution", shadowInvResolution, SHADER_UNIFORM_VEC2)
        set_float(ssaoShader, "camClipNear", camNear)
        set_float(ssaoShader, "camClipFar", camFar)
        set_float(ssaoShader, "lightClipNear", lightNear)
        set_float(ssaoShader, "lightClipFar", lightFar)
        set_vec(ssaoShader, "lightDir", lightDir, SHADER_UNIFORM_VEC3)
        ClearBackground(WHITE)
        DrawTextureRec(ssaoFront.texture, Rectangle(0, 0, width, -height), Vector2(0.0, 0.0), WHITE)
        EndShaderMode()
        EndTextureMode()

        invRes = Vector2(1.0 / width, 1.0 / height)
        for src, dst, direction in ((ssaoFront, ssaoBack, Vector2(1.0, 0.0)), (ssaoBack, ssaoFront, Vector2(0.0, 1.0))):
            BeginTextureMode(dst)
            BeginShaderMode(blurShader)
            SetShaderValueTexture(blurShader, loc(blurShader, "gbufferNormal"), gbuffer.normal)
            SetShaderValueTexture(blurShader, loc(blurShader, "gbufferDepth"), gbuffer.depth)
            SetShaderValueTexture(blurShader, loc(blurShader, "inputTexture"), src.texture)
            SetShaderValueMatrix(blurShader, loc(blurShader, "camInvProj"), camInvProj)
            set_float(blurShader, "camClipNear", camNear)
            set_float(blurShader, "camClipFar", camFar)
            set_vec(blurShader, "invTextureResolution", invRes, SHADER_UNIFORM_VEC2)
            set_vec(blurShader, "blurDirection", direction, SHADER_UNIFORM_VEC2)
            DrawTextureRec(dst.texture, Rectangle(0, 0, width, -height), Vector2(0, 0), WHITE)
            EndShaderMode()
            EndTextureMode()

        BeginTextureMode(lighted)
        BeginShaderMode(lightingShader)
        SetShaderValueTexture(lightingShader, loc(lightingShader, "gbufferColor"), gbuffer.color)
        SetShaderValueTexture(lightingShader, loc(lightingShader, "gbufferNormal"), gbuffer.normal)
        SetShaderValueTexture(lightingShader, loc(lightingShader, "gbufferDepth"), gbuffer.depth)
        SetShaderValueTexture(lightingShader, loc(lightingShader, "ssao"), ssaoFront.texture)
        set_vec(lightingShader, "camPos", camera.cam3d.position, SHADER_UNIFORM_VEC3)
        SetShaderValueMatrix(lightingShader, loc(lightingShader, "camInvViewProj"), camInvViewProj)
        set_vec(lightingShader, "lightDir", lightDir, SHADER_UNIFORM_VEC3)
        set_vec(lightingShader, "sunColor", Vector3(253.0 / 255.0, 255.0 / 255.0, 232.0 / 255.0), SHADER_UNIFORM_VEC3)
        set_vec(lightingShader, "skyColor", Vector3(174.0 / 255.0, 183.0 / 255.0, 190.0 / 255.0), SHADER_UNIFORM_VEC3)
        for name, value in (("sunStrength", 0.25), ("skyStrength", 0.15), ("groundStrength", 0.1),
                            ("ambientStrength", 1.0), ("exposure", 0.9), ("camClipNear", camNear),
                            ("camClipFar", camFar)):
            set_float(lightingShader, name, value)
        ClearBackground(RAYWHITE)
        DrawTextureRec(gbuffer.color, Rectangle(0, 0, width, -height), Vector2(0, 0), WHITE)
        EndShaderMode()

        # GenoView's inspection overlays, per character
        readouts = []
        BeginMode3D(camera.cam3d)
        for a in anims:
            gpos, grot = a["globalPositions"], a["globalRotations"]
            f = min(frame, len(gpos) - 1)
            if toggles["onion"][0]:
                DrawOnionGhosts(genoModel, bindPos, bindRot, gpos, grot, f, len(gpos) - 1, ghostShader, ghostColorLoc)
            if toggles["skeleton"][0]:
                DrawSkeletonOverlay(gpos[f], a["parents"])
            if toggles["transforms"][0]:
                DrawSkeleton(gpos[f], grot[f], a["parents"], GRAY)
            if toggles["foot"][0]:
                for label, j in a["footJoints"]:
                    pos, speed, grounded, sliding, below = MeasureFootSlide(gpos, f, j, a["footContacts"][j])
                    DrawFootSlide(pos, speed, grounded, sliding, below)
                    readouts.append((label, speed, grounded, sliding, below))
        EndMode3D()
        EndTextureMode()

        # FXAA and labels go into `final`, so recordings and the window show the same pixels.
        BeginTextureMode(final)
        BeginShaderMode(fxaaShader)
        SetShaderValueTexture(fxaaShader, loc(fxaaShader, "inputTexture"), lighted.texture)
        set_vec(fxaaShader, "invTextureResolution", invRes, SHADER_UNIFORM_VEC2)
        DrawTextureRec(lighted.texture, Rectangle(0, 0, width, -height), Vector2(0, 0), WHITE)
        EndShaderMode()
        rlEnableColorBlend()
        for i, label in enumerate(labels[:len(anims)]):
            hip = anims[i]["globalPositions"][min(frame, len(anims[i]["globalPositions"]) - 1), 0]
            p = GetWorldToScreenEx(Vector3(float(hip[0]), 0.0, float(hip[2]) + 0.45), camera.cam3d, width, height)
            size = max(20, height // 32)
            text = label.encode()
            DrawText(text, int(p.x - MeasureText(text, size) / 2), int(p.y), size, Color(40, 40, 40, 255))
        EndTextureMode()
        return readouts

    def blit_final():
        # Copy without blending: the FXAA pass leaves alpha at 0, which would make the scene invisible.
        rlDisableColorBlend()
        ClearBackground(RAYWHITE)
        DrawTextureRec(final.texture, Rectangle(0, 0, width, -height), Vector2(0, 0), WHITE)
        rlDrawRenderBatchActive()
        rlEnableColorBlend()

    fps = 1.0 / anims[0]["frameTime"]
    nframes = min(len(a["globalPositions"]) for a in anims)
    first = int(args.start * fps)
    last = min(nframes, int(args.end * fps)) if args.end else nframes

    if args.record:
        record(args, render, blit_final, final, width, height, fps, first, last)
    else:
        interactive(args, anims, paths, labels, load_all, render, blit_final, toggles, colors, width, height, fps, first,
                    last)
    CloseWindow()


def record(args, render, blit_final, final, width, height, fps, first, last):
    import imageio_ffmpeg

    audio_path = None
    if args.audio:
        from scipy.io import wavfile
        sr, wav = wavfile.read(args.audio)
        audio_path = os.path.join(tempfile.mkdtemp(), "audio.wav")
        wavfile.write(audio_path, sr, wav[int(first / fps * sr): int(last / fps * sr)])
    if args.record.lower().endswith(".gif"):
        # palette from the clip itself; GIFs carry no audio
        writer = imageio_ffmpeg.write_frames(
            args.record, (width, height), fps=args.fps, codec="gif", pix_fmt_out="pal8", macro_block_size=1,
            output_params=["-vf", "split[a][b];[a]palettegen[p];[b][p]paletteuse", "-loop", "0"])
    else:
        writer = imageio_ffmpeg.write_frames(
            args.record, (width, height), fps=args.fps, quality=8, macro_block_size=1,
            audio_path=audio_path, audio_codec="aac", output_params=["-shortest"])
    writer.send(None)
    frames = np.arange(first, last, fps / args.fps).astype(int)
    for k, frame in enumerate(frames):
        BeginDrawing()
        render(int(frame), interactive=False)
        blit_final()
        EndDrawing()
        image = LoadImageFromTexture(final.texture)
        pixels = np.frombuffer(ffi.buffer(image.data, width * height * 4), dtype=np.uint8)
        writer.send(pixels.reshape(height, width, 4)[::-1, :, :3].copy())  # render textures are stored bottom-up
        UnloadImage(image)
        if k % 100 == 0:
            print(f"\rframe {k + 1}/{len(frames)}", end="", flush=True)
    writer.close()
    print(f"\rwrote {args.record} ({len(frames)} frames)")


def interactive(args, anims, paths, labels, load_all, render, blit_final, toggles, colors, width, height, fps, first,
                last):
    music = None
    if args.audio:
        InitAudioDevice()
        music = LoadMusicStream(args.audio.encode())
        PlayMusicStream(music)
        SeekMusicStream(music, first / fps)
    frame_ptr = ffi.new("float*")
    frame_ptr[0] = float(first)
    speed_ptr = ffi.new("float*")
    speed_ptr[0] = 1.0
    mode_ptr = ffi.new("int*")  # 0 Loop, 1 PingPong, 2 Once
    playing, direction = True, 1
    selected = 0  # character whose colour the picker edits

    def seek_audio():
        if music:
            SeekMusicStream(music, frame_ptr[0] / fps)

    while not WindowShouldClose():
        new_paths = None
        if IsFileDropped():
            dropped = LoadDroppedFiles()
            new_paths = [ffi.string(dropped.paths[i]).decode() for i in range(dropped.count)]
            new_paths = [p for p in new_paths if p.lower().endswith(".bvh")] or None
            UnloadDroppedFiles(dropped)
        if IsKeyPressed(KEY_F5):
            new_paths = list(paths)
        if new_paths:
            try:
                anims[:] = load_all(new_paths)
                if new_paths != paths:
                    paths[:] = new_paths
                    labels[:] = [os.path.splitext(os.path.basename(p))[0] for p in new_paths] if len(new_paths) > 1 else []
                frame_ptr[0], direction = 0.0, 1
                seek_audio()
            except Exception as e:  # keep the viewer alive on a bad file
                print(f"Failed to load {new_paths}: {e}")
        end = float(max(min(min(len(a["globalPositions"]) for a in anims), last) - 1, 0))

        if IsKeyPressed(KEY_SPACE):
            playing = not playing
        if IsKeyPressed(KEY_RIGHT):
            playing, frame_ptr[0] = False, min(end, float(int(frame_ptr[0]) + 1))
        if IsKeyPressed(KEY_LEFT):
            playing, frame_ptr[0] = False, max(0.0, float(int(frame_ptr[0]) - 1))
        if IsKeyPressed(KEY_HOME):
            frame_ptr[0] = 0.0
            seek_audio()
        if IsKeyPressed(KEY_END):
            frame_ptr[0] = end
        if IsKeyPressed(KEY_UP):
            speed_ptr[0] = min(4.0, speed_ptr[0] + 0.25)
        if IsKeyPressed(KEY_DOWN):
            speed_ptr[0] = max(0.1, speed_ptr[0] - 0.25)

        # Audio drives the playhead at 1x in Loop/Once; otherwise it is paused and the clock is used.
        audio_on = bool(music) and playing and mode_ptr[0] != 1 and abs(speed_ptr[0] - 1.0) < 1e-6
        if music:
            if audio_on and not IsMusicStreamPlaying(music):
                seek_audio()
                ResumeMusicStream(music)
            elif not audio_on and IsMusicStreamPlaying(music):
                PauseMusicStream(music)
            UpdateMusicStream(music)

        if playing and end > 0:
            if audio_on:
                frame_ptr[0] = GetMusicTimePlayed(music) * fps
            else:
                frame_ptr[0] += speed_ptr[0] * min(GetFrameTime(), 0.1) * fps * direction
            if mode_ptr[0] == 1:
                if frame_ptr[0] >= end:
                    frame_ptr[0], direction = end, -1
                elif frame_ptr[0] <= 0:
                    frame_ptr[0], direction = 0.0, 1
            elif frame_ptr[0] >= end:
                if mode_ptr[0] == 0:
                    frame_ptr[0] = 0.0
                    seek_audio()
                else:
                    frame_ptr[0], playing = end, False
        frame_ptr[0] = min(end, max(0.0, frame_ptr[0]))
        frame = int(frame_ptr[0])

        BeginDrawing()
        readouts = render(frame, interactive=True)
        blit_final()

        # UI on the window only (not in recordings)
        GuiGroupBox(Rectangle(20, 10, 190, 100), b"Camera")
        GuiLabel(Rectangle(30, 20, 170, 20), b"Ctrl + Left Click - Rotate")
        GuiLabel(Rectangle(30, 40, 170, 20), b"Ctrl + Right Click - Pan")
        GuiLabel(Rectangle(30, 60, 170, 20), b"Mouse Scroll - Zoom")
        GuiLabel(Rectangle(30, 80, 170, 20), b"Drop .bvh files - F5 reload")
        GuiGroupBox(Rectangle(width - 260, 10, 240, 144), b"Rendering")
        for k, (name, text) in enumerate((("skeleton", b"Skeleton"), ("transforms", b"Draw Transforms"),
                                          ("foot", b"Foot Slide"), ("onion", b"Onion Skin"), ("colors", b"Colours"))):
            GuiCheckBox(Rectangle(width - 250, 20 + 26 * k, 20, 20), text, toggles[name])
        panel_top = 164
        if toggles["colors"][0]:
            GuiGroupBox(Rectangle(width - 260, panel_top, 240, 214), b"Colours")
            selected = min(selected, len(anims) - 1)
            for i in range(len(anims)):  # one swatch per character; click to pick which one to edit
                swatch = Rectangle(width - 250 + 30 * i, panel_top + 12, 24, 24)
                DrawRectangleRec(swatch, colors[i % len(colors)][0])
                DrawRectangleLinesEx(swatch, 2 if i == selected else 1, BLACK if i == selected else GRAY)
                if IsMouseButtonPressed(0) and CheckCollisionPointRec(GetMousePosition(), swatch):
                    selected = i
            c = colors[selected % len(colors)]
            GuiColorPicker(Rectangle(width - 250, panel_top + 44, 180, 140), b"", c)
            GuiLabel(Rectangle(width - 250, panel_top + 188, 220, 18),
                     f"character {selected + 1}: #{c.r:02x}{c.g:02x}{c.b:02x}".encode())
            panel_top += 224
        if toggles["foot"][0] and readouts:
            GuiGroupBox(Rectangle(width - 260, panel_top, 240, 30 + 18 * len(readouts)), b"Foot Slide (m/s)")
            for k, (label, speed, grounded, sliding, below) in enumerate(readouts):
                state = b"BELOW FLOOR" if below else b"SLIDING" if sliding else b"planted" if grounded else b"air"
                char = f"{k // 4 + 1} " if len(anims) > 1 else ""
                GuiLabel(Rectangle(width - 250, panel_top + 10 + 18 * k, 230, 18),
                         f"{char}{label:7s} {speed:5.3f}  ".encode() + state)

        panel_y = height - 96
        GuiGroupBox(Rectangle(20, panel_y, width - 40, 86), b"Playback")
        scrub_w = width - 72 - 140
        before = frame_ptr[0]
        GuiSliderBar(Rectangle(36, panel_y + 16, scrub_w, 16), b"", b"", frame_ptr, 0.0, end)
        if frame_ptr[0] != before:  # scrubbed: move the audio too
            seek_audio()
        GuiLabel(Rectangle(46 + scrub_w, panel_y + 14, 140, 20), f"{frame / fps:6.2f} s  ({frame})".encode())
        by = panel_y + 46
        if GuiButton(Rectangle(36, by, 36, 26), b"|<"):
            frame_ptr[0] = 0.0
            seek_audio()
        if GuiButton(Rectangle(76, by, 36, 26), b"<"):
            playing, frame_ptr[0] = False, max(0.0, float(frame - 1))
        if GuiButton(Rectangle(116, by, 74, 26), b"Pause" if playing else b"Play"):
            playing = not playing
        if GuiButton(Rectangle(194, by, 36, 26), b">"):
            playing, frame_ptr[0] = False, min(end, float(frame + 1))
        if GuiButton(Rectangle(234, by, 36, 26), b">|"):
            frame_ptr[0] = end
        GuiSliderBar(Rectangle(344, by + 4, 150, 16), b"Speed", f"{speed_ptr[0]:.2f}x".encode(), speed_ptr, 0.1, 4.0)
        GuiComboBox(Rectangle(width - 150, by, 114, 26), b"Loop;PingPong;Once", mode_ptr)
        EndDrawing()


if __name__ == "__main__":
    main()
