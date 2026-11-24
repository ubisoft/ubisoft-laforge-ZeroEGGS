# Helpers from GenoViewPython (https://github.com/orangeduck/GenoViewPython, MIT, Daniel Holden),
# taken unchanged from the viewer-enhancements branch (PR #1, macOS rendering fixes and inspection tools).

from pyray import (
    Vector2, Vector3, Vector4, Transform, Matrix, Camera3D, 
    Color, Rectangle, Model, ModelAnimation, Mesh, BoneInfo, 
    Texture, RenderTexture)
from raylib import *

import bvh
import quat
import numpy as np
import struct
import os
import sys
import subprocess
import cffi
ffi = cffi.FFI()

#----------------------------------------------------------------------------------
# Camera
#----------------------------------------------------------------------------------

class Camera:

    def __init__(self):
        self.cam3d = Camera3D()
        self.cam3d.position = Vector3(2.0, 3.0, 5.0)
        self.cam3d.target = Vector3(-0.5, 1.0, 0.0)
        self.cam3d.up = Vector3(0.0, 1.0, 0.0)
        self.cam3d.fovy = 45.0
        self.cam3d.projection = CAMERA_PERSPECTIVE
        self.azimuth = 0.0
        self.altitude = 0.4
        self.distance = 4.0
        self.offset = Vector3Zero()
    
    def update(
        self,
        target,
        azimuthDelta,
        altitudeDelta,
        offsetDeltaX,
        offsetDeltaY,
        mouseWheel,
        dt):

        self.azimuth = self.azimuth + 1.0 * dt * -azimuthDelta
        self.altitude = Clamp(self.altitude + 1.0 * dt * altitudeDelta, 0.0, 0.4 * PI)
        self.distance = Clamp(self.distance +  20.0 * dt * -mouseWheel, 0.1, 100.0)
        
        rotationAzimuth = QuaternionFromAxisAngle(Vector3(0, 1, 0), self.azimuth)
        position = Vector3RotateByQuaternion(Vector3(0, 0, self.distance), rotationAzimuth)
        axis = Vector3Normalize(Vector3CrossProduct(position, Vector3(0, 1, 0)))

        rotationAltitude = QuaternionFromAxisAngle(axis, self.altitude)

        localOffset = Vector3(dt * offsetDeltaX, dt * -offsetDeltaY, 0.0)
        localOffset = Vector3RotateByQuaternion(localOffset, rotationAzimuth)

        self.offset = Vector3Add(self.offset, Vector3RotateByQuaternion(localOffset, rotationAltitude))

        cameraTarget = Vector3Add(self.offset, target)
        eye = Vector3Add(cameraTarget, Vector3RotateByQuaternion(position, rotationAltitude))

        self.cam3d.target = cameraTarget
        self.cam3d.position = eye        

#----------------------------------------------------------------------------------
# Shadow Maps
#----------------------------------------------------------------------------------

class ShadowLight:
    
    def __init__(self):
        
        self.target = Vector3Zero()
        self.position = Vector3Zero()
        self.up = Vector3(0.0, 1.0, 0.0)
        self.target = Vector3Zero()
        self.width = 0
        self.height = 0
        self.near = 0.0
        self.far = 1.0


def LoadShadowMap(width, height):

    target = RenderTexture()
    target.id = rlLoadFramebuffer()
    target.texture.width = width
    target.texture.height = height
    assert target.id != 0
    
    rlEnableFramebuffer(target.id)

    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)
    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target

def UnloadShadowMap(target):
    
    if target.id > 0:
        rlUnloadFramebuffer(target.id)
        

def BeginShadowMap(target, shadowLight):
    
    BeginTextureMode(target)
    ClearBackground(WHITE)
    
    rlDrawRenderBatchActive()      # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)    # Switch to projection matrix
    rlPushMatrix()                 # Save previous matrix, which contains the settings for the 2d ortho projection
    rlLoadIdentity()               # Reset current matrix (projection)

    rlOrtho(
        -shadowLight.width/2, shadowLight.width/2, 
        -shadowLight.height/2, shadowLight.height/2, 
        shadowLight.near, shadowLight.far)

    rlMatrixMode(RL_MODELVIEW)     # Switch back to modelview matrix
    rlLoadIdentity()               # Reset current matrix (modelview)

    # Setup Camera view
    matView = MatrixLookAt(shadowLight.position, shadowLight.target, shadowLight.up)
    rlMultMatrixf(MatrixToFloatV(matView).v)      # Multiply modelview matrix by view matrix (camera)

    rlEnableDepthTest()            # Enable DEPTH_TEST for 3D    


def EndShadowMap():
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlMatrixMode(RL_PROJECTION)     # Switch to projection matrix
    rlPopMatrix()                   # Restore previous matrix (projection) from matrix stack

    rlMatrixMode(RL_MODELVIEW)      # Switch back to modelview matrix
    rlLoadIdentity()                # Reset current matrix (modelview)

    rlDisableDepthTest()            # Disable DEPTH_TEST for 2D

    EndTextureMode()

def SetShaderValueShadowMap(shader, locIndex, target):
    if locIndex > -1:
        rlEnableShader(shader.id)
        slotPtr = ffi.new('int*'); slotPtr[0] = 10  # Can be anything 0 to 15, but 0 will probably be taken up
        rlActiveTextureSlot(slotPtr[0])
        rlEnableTexture(target.depth.id)
        rlSetUniform(locIndex, slotPtr, SHADER_UNIFORM_INT, 1)

#----------------------------------------------------------------------------------
# GBuffer
#----------------------------------------------------------------------------------

class GBuffer:
    
    def __init__(self):
        self.id = 0              # OpenGL framebuffer object id
        self.color = Texture()   # Color buffer attachment texture 
        self.normal = Texture()  # Normal buffer attachment texture 
        self.depth = Texture()   # Depth buffer attachment texture


def LoadGBuffer(width, height):
    
    target = GBuffer()
    target.id = rlLoadFramebuffer()
    assert target.id
    
    rlEnableFramebuffer(target.id)

    target.color.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8, 1)
    target.color.width = width
    target.color.height = height
    target.color.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    target.color.mipmaps = 1
    rlFramebufferAttach(target.id, target.color.id, RL_ATTACHMENT_COLOR_CHANNEL0, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.normal.id = rlLoadTexture(ffi.NULL, width, height, PIXELFORMAT_UNCOMPRESSED_R16G16B16A16, 1)
    target.normal.width = width
    target.normal.height = height
    target.normal.format = PIXELFORMAT_UNCOMPRESSED_R16G16B16A16
    target.normal.mipmaps = 1
    rlFramebufferAttach(target.id, target.normal.id, RL_ATTACHMENT_COLOR_CHANNEL1, RL_ATTACHMENT_TEXTURE2D, 0)
    
    target.depth.id = rlLoadTextureDepth(width, height, False)
    target.depth.width = width
    target.depth.height = height
    target.depth.format = 19       #DEPTH_COMPONENT_24BIT?
    target.depth.mipmaps = 1
    rlFramebufferAttach(target.id, target.depth.id, RL_ATTACHMENT_DEPTH, RL_ATTACHMENT_TEXTURE2D, 0)

    assert rlFramebufferComplete(target.id)

    rlDisableFramebuffer()

    return target


def UnloadGBuffer(target):

    if target.id > 0:
        rlUnloadFramebuffer(target.id)


def BeginGBuffer(target, camera):
    
    rlDrawRenderBatchActive()       # Update and draw internal render batch

    rlEnableFramebuffer(target.id)  # Enable render target
    rlActiveDrawBuffers(2) 

    # Set viewport and RLGL internal framebuffer size
    rlViewport(0, 0, target.color.width, target.color.height)
    rlSetFramebufferWidth(target.color.width)
    rlSetFramebufferHeight(target.color.height)

    ClearBackground(BLACK)

    # Use raylib's own camera setup for the projection/view matrices. The
    # hand-rolled rlFrustum + rlMultMatrixf path silently culls all geometry
    # on macOS's OpenGL driver, so we let BeginMode3D build the matrices
    # (it pushes the projection matrix; EndMode3D pops it in EndGBuffer).
    BeginMode3D(camera)


def EndGBuffer(windowWidth, windowHeight):

    EndMode3D()                     # Restore matrices (pairs with BeginMode3D in BeginGBuffer)

    rlActiveDrawBuffers(1)
    rlDisableFramebuffer()          # Disable render target (fbo)


#----------------------------------------------------------------------------------
# Geno Character and Animation
#----------------------------------------------------------------------------------

def FileRead(out, size, f):
    ffi.memmove(out, f.read(size), size)

def LoadGenoModel(fileName):

    model = Model()
    model.transform = MatrixIdentity()
  
    with open(fileName, "rb") as f:
        
        model.materialCount = 1
        model.materials = MemAlloc(model.materialCount * ffi.sizeof(Mesh()))
        model.materials[0] = LoadMaterialDefault()

        model.meshCount = 1
        model.meshMaterial = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshMaterial[0] = 0

        model.meshes = MemAlloc(model.meshCount * ffi.sizeof(Mesh()))
        model.meshes[0].vertexCount = struct.unpack('I', f.read(4))[0]
        model.meshes[0].triangleCount = struct.unpack('I', f.read(4))[0]
        model.boneCount = struct.unpack('I', f.read(4))[0]

        model.meshes[0].boneCount = model.boneCount
        model.meshes[0].vertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].texcoords = MemAlloc(model.meshes[0].vertexCount * 2 * ffi.sizeof("float"))
        model.meshes[0].normals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].boneIds = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("unsigned char"))
        model.meshes[0].boneWeights = MemAlloc(model.meshes[0].vertexCount * 4 * ffi.sizeof("float"))
        model.meshes[0].indices = MemAlloc(model.meshes[0].triangleCount * 3 * ffi.sizeof("unsigned short"))
        model.meshes[0].animVertices = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.meshes[0].animNormals = MemAlloc(model.meshes[0].vertexCount * 3 * ffi.sizeof("float"))
        model.bones =  MemAlloc(model.boneCount * ffi.sizeof(BoneInfo()))
        model.bindPose =  MemAlloc(model.boneCount * ffi.sizeof(Transform()))
        
        FileRead(model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].texcoords, ffi.sizeof("float") * model.meshes[0].vertexCount * 2, f)
        FileRead(model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3, f)
        FileRead(model.meshes[0].boneIds, ffi.sizeof("unsigned char") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].boneWeights, ffi.sizeof("float") * model.meshes[0].vertexCount * 4, f)
        FileRead(model.meshes[0].indices, ffi.sizeof("unsigned short") * model.meshes[0].triangleCount * 3, f)
        ffi.memmove(model.meshes[0].animVertices, model.meshes[0].vertices, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        ffi.memmove(model.meshes[0].animNormals, model.meshes[0].normals, ffi.sizeof("float") * model.meshes[0].vertexCount * 3)
        FileRead(model.bones, ffi.sizeof(BoneInfo()) * model.boneCount, f)
        FileRead(model.bindPose, ffi.sizeof(Transform()) * model.boneCount, f)
        
        model.meshes[0].boneMatrices = MemAlloc(model.boneCount * ffi.sizeof(Matrix()))
        for i in range(model.boneCount):
            model.meshes[0].boneMatrices[i] = MatrixIdentity()
    
    UploadMesh(ffi.addressof(model.meshes[0]), True)
    
    return model


def GetModelBindPoseAsNumpyArrays(model):
    
    bindPos = np.zeros([model.boneCount, 3])
    bindRot = np.zeros([model.boneCount, 4])
    
    for boneId in range(model.boneCount):
        bindTransform = model.bindPose[boneId]
        bindPos[boneId] = (bindTransform.translation.x, bindTransform.translation.y, bindTransform.translation.z)
        bindRot[boneId] = (bindTransform.rotation.w, bindTransform.rotation.x, bindTransform.rotation.y, bindTransform.rotation.z)
        
    return bindPos, bindRot
    
    
def UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot, animPos, animRot):
    
    meshPos = quat.mul_vec(animRot, quat.inv_mul_vec(bindRot, -bindPos)) + animPos
    meshRot = quat.mul_inv(animRot, bindRot)
    
    matArray = np.frombuffer(ffi.buffer(
        model.meshes[0].boneMatrices, model.boneCount * 4 * 4 * 4), 
        dtype=np.float32).reshape([model.boneCount, 4, 4])
    
    matArray[:,:3,:3] = quat.to_xform(meshRot)
    matArray[:,:3,3] = meshPos


#----------------------------------------------------------------------------------
# Debug Draw
#----------------------------------------------------------------------------------

def DrawTransform(position, rotation, scale):
    
    rotMatrix = QuaternionToMatrix(Vector4(*rotation))
  
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m0, scale * rotMatrix.m1, scale * rotMatrix.m2)),
        RED)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m4, scale * rotMatrix.m5, scale * rotMatrix.m6)),
        GREEN)
        
    DrawLine3D(
        Vector3(*position),
        Vector3Add(Vector3(*position), Vector3(scale * rotMatrix.m8, scale * rotMatrix.m9, scale * rotMatrix.m10)),
        BLUE)

def DrawSkeleton(positions, rotations, parents, color):
    
    for i in range(len(positions)):
    
        DrawSphereWires(
            Vector3(*positions[i]),
            0.01,
            4,
            6,
            color)

        DrawTransform(positions[i], rotations[i], 0.1)

        if parents[i] != -1:

            DrawLine3D(
                Vector3(*positions[i]),
                Vector3(*positions[parents[i]]),
                color)


def DrawSkeletonOverlay(positions, parents):
    # Clean skeleton: solid bones (joint -> parent) with joint markers, drawn as
    # an overlay so it reads clearly over the skinned mesh.
    boneColor = Color(235, 235, 240, 255)
    jointColor = Color(255, 190, 40, 255)
    for i in range(len(positions)):
        par = parents[i]
        if par != -1:
            DrawCylinderEx(Vector3(*positions[par]), Vector3(*positions[i]),
                           0.007, 0.007, 6, boneColor)
    for i in range(len(positions)):
        DrawSphere(Vector3(*positions[i]), 0.013, jointColor)


#----------------------------------------------------------------------------------
# Foot Slide Detection
#----------------------------------------------------------------------------------

# A foot is "planted" only when its contact joint stays close to the ground for
# a sustained window of frames - not just for the instant it passes low through
# the bottom of a step. This is what separates a real foot SLIDE (a planted foot
# that drifts) from a foot that is simply moving/stepping near the ground.

FOOT_CONTACT_BAND   = 0.03   # m above the joint's lowest point that counts as touching the ground
FOOT_CONTACT_FRAMES = 5      # +/- frames the foot must stay grounded to count as planted (~contact persistence)
FOOT_SLIDE_SPEED    = 0.10   # m/s: a planted foot moving faster than this is flagged as sliding
FOOT_SLIDE_FULL     = 0.30   # m/s: slide speed mapped to a fully-red marker
FOOT_FLOOR_LEVEL    = -0.02  # m: a foot joint below this has penetrated the floor (small tolerance)


def ComputeFootContacts(globalPositions, jointIndex, dt):
    # Pre-computes, for one foot joint over the whole clip:
    #   hSpeed[f] - horizontal speed (m/s)
    #   planted[f] - sustained ground contact (grounded across a centred window)
    p = globalPositions[:, jointIndex, :]
    h = p[:, 1]
    hSpeed = np.empty(len(h), np.float32)
    hSpeed[0] = 0.0
    hSpeed[1:] = np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=1) / dt

    groundedH = h < (h.min() + FOOT_CONTACT_BAND)
    # planted[f] = groundedH for every frame in [f - W, f + W]  (via prefix sums)
    W = FOOT_CONTACT_FRAMES
    csum = np.concatenate([[0], np.cumsum(groundedH.astype(np.int32))])
    F = len(h)
    planted = np.zeros(F, bool)
    for f in range(F):
        a = max(0, f - W); b = min(F, f + W + 1)
        planted[f] = (csum[b] - csum[a]) == (b - a)
    return hSpeed, planted


def MeasureFootSlide(globalPositions, frame, jointIndex, contacts):
    # contacts = (hSpeedArray, plantedArray) from ComputeFootContacts
    hSpeedArr, plantedArr = contacts
    f = min(frame, len(hSpeedArr) - 1)
    hSpeed = float(hSpeedArr[f])
    grounded = bool(plantedArr[f])
    sliding = grounded and (hSpeed > FOOT_SLIDE_SPEED)
    p = globalPositions[frame][jointIndex]
    penetrating = float(p[1]) < FOOT_FLOOR_LEVEL
    return Vector3(float(p[0]), float(p[1]), float(p[2])), hSpeed, grounded, sliding, penetrating


def DrawFootSlide(pos, hSpeed, grounded, sliding, penetrating):
    if penetrating:
        # foot is below the floor - flag it (magenta), regardless of slide state
        DrawSphere(pos, 0.028, MAGENTA)
        DrawLine3D(pos, Vector3(pos.x, 0.0, pos.z), MAGENTA)
        DrawCircle3D(Vector3(pos.x, 0.001, pos.z), 0.06, Vector3(1.0, 0.0, 0.0), 90.0, MAGENTA)
    elif grounded:
        # green (planted, still) -> red, ramping only once it is actually sliding
        t = max(0.0, min(1.0, (hSpeed - FOOT_SLIDE_SPEED) / (FOOT_SLIDE_FULL - FOOT_SLIDE_SPEED)))
        col = Color(int(40 + 215 * t), int(200 * (1.0 - t) + 40 * t), 55, 255)
        DrawSphere(pos, 0.025, col)
        DrawLine3D(pos, Vector3(pos.x, 0.0, pos.z), col)   # drop line to the floor
        if sliding:
            # small flat ring on the floor marking the sliding contact
            r = min(0.07, 0.035 + hSpeed * 0.06)
            DrawCircle3D(Vector3(pos.x, 0.001, pos.z), r, Vector3(1.0, 0.0, 0.0), 90.0, RED)
    else:
        DrawSphere(pos, 0.018, Fade(SKYBLUE, 0.45))        # airborne: faint blue


#----------------------------------------------------------------------------------
# Onion Skinning (ghost frames)
#----------------------------------------------------------------------------------

# Draw the pose at frame +/- k*stride as translucent "ghosts" so the motion is
# legible in a still frame. Past frames are tinted cool, future frames warm, and
# both fade out the further they are in time. Drawn as a forward translucent
# overlay (the deferred G-buffer can only hold one opaque surface per pixel).

ONION_STRIDE       = 6      # frames between successive ghosts
ONION_COUNT        = 3      # number of ghosts per side (past / future)
ONION_MAX_ALPHA    = 0.32   # opacity of the nearest ghost
ONION_PAST_COLOR   = (0.35, 0.55, 0.95)   # cool blue  - earlier frames
ONION_FUTURE_COLOR = (0.95, 0.50, 0.25)   # warm orange - later frames


def DrawOnionGhosts(model, bindPos, bindRot, globalPositions, globalRotations,
                    frame, maxFrame, ghostShader, ghostColorLoc):
    model.materials[0].shader = ghostShader
    colPtr = ffi.new('float[4]')
    rlEnableColorBlend()
    for side, base in ((-1, ONION_PAST_COLOR), (1, ONION_FUTURE_COLOR)):
        for k in range(1, ONION_COUNT + 1):
            fg = frame + side * k * ONION_STRIDE
            if fg < 0 or fg > maxFrame:
                continue
            alpha = ONION_MAX_ALPHA * (1.0 - (k - 1) / float(ONION_COUNT))
            colPtr[0], colPtr[1], colPtr[2], colPtr[3] = base[0], base[1], base[2], alpha
            SetShaderValue(ghostShader, ghostColorLoc, colPtr, SHADER_UNIFORM_VEC4)
            UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot,
                                           globalPositions[fg], globalRotations[fg])
            DrawModel(model, Vector3Zero(), 1.0, WHITE)
    rlDisableColorBlend()
    # restore the model to the current frame's pose for subsequent passes
    UpdateModelPoseFromNumpyArrays(model, bindPos, bindRot,
                                   globalPositions[frame], globalRotations[frame])


#----------------------------------------------------------------------------------
# Animation loading (supports runtime swapping / hot-reload)
#----------------------------------------------------------------------------------

def LoadAnimation(path):
    # Loads a BVH and pre-computes everything the viewer needs for it. Returns a
    # dict so the same routine can be used at start-up and for runtime reloads.
    bvhData = bvh.load(path)
    parents = bvhData['parents']
    localPositions = 0.01 * bvhData['positions'].copy().astype(np.float32)
    localRotations = quat.unroll(quat.from_euler(np.radians(bvhData['rotations']), order=bvhData['order']))
    globalRotations, globalPositions = quat.fk(localRotations, localPositions, parents)

    frameTime = bvhData.get('frametime', 1.0 / 60.0)
    if frameTime <= 0.0:
        frameTime = 1.0 / 60.0

    names = bvhData['names']
    footSpec = [("L Toe", "LeftToeBase"), ("R Toe", "RightToeBase"),
                ("L Foot", "LeftFoot"), ("R Foot", "RightFoot")]
    footJoints = [(label, names.index(name)) for (label, name) in footSpec if name in names]
    # foot speeds are in m/s, so use the clip's real frame time
    footContacts = {j: ComputeFootContacts(globalPositions, j, frameTime) for (_, j) in footJoints}

    return {
        'path': path,
        'name': os.path.basename(path),
        'parents': parents,
        'localPositions': localPositions,
        'globalRotations': globalRotations,
        'globalPositions': globalPositions,
        'footJoints': footJoints,
        'footContacts': footContacts,
        'frameTime': frameTime,
    }


def ScanBvhFolder(folder):
    # Returns a sorted list of full .bvh paths in the folder (empty if missing).
    try:
        return [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith(".bvh")]
    except OSError:
        return []


def NativeChoose(folder=False):
    # Opens a native macOS file/folder picker via osascript and returns the chosen
    # POSIX path (or None if cancelled / unsupported). Used by the Open buttons.
    if sys.platform != "darwin":
        print("Native file picker is only wired up for macOS; use drag-and-drop instead.")
        return None
    # NOTE: we deliberately don't pass `of type {"bvh"}` - .bvh has no registered
    # UTI on macOS, which makes the dialog grey out every file. Instead we let any
    # file be chosen and validate the extension afterwards.
    if folder:
        script = 'POSIX path of (choose folder with prompt "Select a folder of BVH files")'
    else:
        script = 'POSIX path of (choose file with prompt "Select a BVH file")'
    try:
        out = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True, timeout=300)
        path = out.stdout.strip()
        return path or None
    except Exception as e:
        print("File picker failed:", e)
        return None


#----------------------------------------------------------------------------------
# Shader loading
#----------------------------------------------------------------------------------

def LoadShaderVersioned(vsPath, fsPath):
    # Loads shaders while adapting the GLSL "#version" line to the GL backend
    # raylib was built against: desktop GL (>= 3.3) wants "#version 330", while
    # web / mobile want "#version 300 es". The shader sources stay written as
    # GLSL ES so they work on every platform; only the version line is rewritten.
    header = "#version 330"
    if rlGetVersion() in (RL_OPENGL_ES_20, RL_OPENGL_ES_30):
        header = "#version 300 es"

    def patched(path):
        with open(path) as f:
            lines = f.read().split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith("#version"):
                lines[i] = header
                break
        return "\n".join(lines).encode()

    return LoadShaderFromMemory(patched(vsPath), patched(fsPath))


#----------------------------------------------------------------------------------
# App
#----------------------------------------------------------------------------------

