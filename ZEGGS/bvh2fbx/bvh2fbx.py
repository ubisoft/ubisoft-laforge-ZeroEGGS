import sys, os, logging
import pyfbsdk

# Customs
# MoBu env
sys.path.append("C:/Users/sghorbani/Anaconda3/envs/mobu2/Lib/site-packages/")


import numpy as np

logging.basicConfig(
    filename="compile_results.log",
    filemode="w",
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.DEBUG,
)

console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
logging.getLogger("").addHandler(console)


def bvh2fbx(animation_file, output_file, template_file, sound_file=None):

    pyfbsdk.FBApplication().FileNew()
    logging.info("Loading %s..." % str(template_file))

    if not pyfbsdk.FBApplication().FileOpen(str(template_file)):
        raise IOError("Could not open file: {}".format(str(template_file)))

    if sound_file is not None:
        # Load Audio
        logging.info("Loading %s..." % str(sound_file))
        audio = pyfbsdk.FBAudioClip(sound_file)
        if audio is None:
            raise IOError("Could not open file: {}".format(str(sound_file)))

        # Rescale Timespan
        pyfbsdk.FBSystem().CurrentTake.LocalTimeSpan = pyfbsdk.FBTimeSpan(
            pyfbsdk.FBTime(0), audio.Duration
        )

    # Set FPS
    pyfbsdk.FBPlayerControl().SetTransportFps(pyfbsdk.FBTimeMode.kFBTimeMode60Frames)
    pyfbsdk.FBPlayerControl().SnapMode = (
        pyfbsdk.FBTransportSnapMode.kFBTransportSnapModeSnapOnFrames
    )

    # Load BVH
    if not pyfbsdk.FBApplication().FileImport(animation_file, True):
        raise IOError("Could not open file: {}".format(str(animation_file)))

    # Save FBX
    pyfbsdk.FBApplication().FileSave(output_file)


if True:
    try:

        logging.info("======")
        logging.info("BVH2FBX")
        logging.info("======")

        results_path = "./Rendered"
        template_file = "./LaForgeFemale.fbx"

        # Characterizing all bvh files
        for animation_file in [f for f in os.listdir(results_path) if f.endswith(".bvh")]:
            sound_file = animation_file.replace(".bvh", ".wav")
            sound_file = results_path + "/" + sound_file
            if not os.path.exists(sound_file):
                sound_file = None
            bvh2fbx(
                results_path + "/" + animation_file,
                results_path + "/" + animation_file.replace(".bvh", ".fbx"),
                template_file,
                sound_file
            )

        pyfbsdk.FBApplication().FileExit()

    except Exception as e:
        logging.exception("FAILED:")
        raise e
