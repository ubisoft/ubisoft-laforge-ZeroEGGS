import os, platform, subprocess

# setup dependencies on import
# check for any missing that the user needs to install and inform them
if platform.system() == "Windows":
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'sox', 'windows')
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'ffmpeg', 'windows')
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'windows')


elif platform.system() == "Linux":
    os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'linux')
    try:
        output = subprocess.check_output(['sox', '--version'])
    except Exception as e:
       raise Exception("Error calling sox dependency: " + str(e) + "\n\nHave you installed sox with 'apt-get install sox'?\n")

    try:
        output = subprocess.check_output(['ffmpeg', '-version'])
    except Exception as e:
       raise Exception("Error calling ffmpeg dependency: " + str(e) + "\n\nHave you installed ffmpeg with 'apt-get install ffmpeg'?\n")

elif platform.system() == "Darwin":
    # Skipping because not sure if used at all
    # os.environ['PATH'] += ";" + os.path.join(os.path.dirname(__file__), 'bin', 'psola', 'linux')
    try:
        output = subprocess.check_output(['sox', '--version'])
    except Exception as e:
       raise Exception("Error calling sox dependency: " + str(e) + "\n\nHave you installed sox with 'brew install sox'?\n")

    try:
        output = subprocess.check_output(['ffmpeg', '-version'])
    except Exception as e:
       raise Exception("Error calling ffmpeg dependency: " + str(e) + "\n\nHave you installed ffmpeg with 'brew install ffmpeg'?\n")


else:
    raise Exception("Unsupported platform: {}".format(platform.system()))