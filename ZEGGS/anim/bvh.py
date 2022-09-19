import re
import numpy as np

def load(filename, start=None, end=None, order=None):
    
    channelmap = {
        'Xrotation' : 'x',
        'Yrotation' : 'y',
        'Zrotation' : 'z'   
    }
    
    f = open(filename, "r")

    i = 0
    active = -1
    end_site = False
    state = 'definition'
    
    names   = []
    offsets = np.empty(shape=[0, 3], dtype=np.float32)
    parents = np.empty(shape=[0],    dtype=np.int32)
    
    for line in f:
        
        if state == 'definition':
        
            if "HIERARCHY" in line: continue
            if "MOTION" in line: continue

            rmatch = re.match(r"ROOT (\w+)", line)
            if rmatch:
                names.append(rmatch.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]], dtype=np.float32), axis=0)
                parents = np.append(parents, np.array([active], dtype=np.int32))
                active  = parents.shape[0]-1
                continue

            if "{" in line: continue

            if "}" in line:
                if end_site: end_site = False
                else: active = parents[active]
                continue
            
            offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
            if offmatch:
                if not end_site:
                    offsets[active] = np.array(list(map(float, offmatch.groups())))
                continue
               
            chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
            if chanmatch:
                channels = int(chanmatch.group(1))
                if order is None:
                    channelis = 0 if channels == 3 else 3
                    channelie = 3 if channels == 3 else 6
                    parts = line.split()[2+channelis:2+channelie]
                    if any([p not in channelmap for p in parts]):
                        continue
                    order = "".join([channelmap[p] for p in parts])
                continue

            jmatch = re.match("\s*JOINT\s+(\w+)", line)
            if jmatch:
                names.append(jmatch.group(1))
                offsets = np.append(offsets, np.array([[0,0,0]], dtype=np.float32), axis=0)
                parents = np.append(parents, np.array([active], dtype=np.int32))
                active  = (parents.shape[0]-1)
                continue
            
            if "End Site" in line:
                end_site = True
                continue
                  
            fmatch = re.match("\s*Frames:\s+(\d+)", line)
            if fmatch:
                if start and end:
                    fnum = (end - start)-1
                else:
                    fnum = int(fmatch.group(1))
                jnum = parents.shape[0]
                positions = offsets[np.newaxis].repeat(fnum, axis=0)
                rotations = np.zeros([fnum, jnum, 3], dtype=np.float32)
                continue
            
            fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
            if fmatch:
                frametime = float(fmatch.group(1))
                state = 'body'
                continue
            
        elif state == 'body':
            
            if (start and end) and (i < start or i >= end-1):
                i += 1
                continue
            
            dmatch = line.strip().split()
            if dmatch:
                
                fi = i - start if start else i
                data_block = np.asarray(tuple(map(float, dmatch)))
                N = parents.shape[0]
                if   channels == 3:
                    positions[fi,0] = data_block[0:3]
                    rotations[fi,:] = data_block[3: ].reshape([N, 3])
                elif channels == 6:
                    data_block = data_block.reshape([N, 6])
                    positions[fi,:] = data_block[:,0:3]
                    rotations[fi,:] = data_block[:,3:6]
                elif channels == 9:
                    positions[fi,0] = data_block[0:3]
                    data_block = data_block[3:].reshape([N-1, 9])
                    rotations[fi,1:] = data_block[:,3:6]
                    positions[fi,1:] = positions[fi,1:] + data_block[:,0:3] * data_block[:,6:9]
                else:
                    raise Exception("Too many channels! %i" % channels)

                i += 1
        
        else:
        
            raise Exception()
        
    f.close()
    
    return {
        'rotations': rotations,
        'positions': positions,
        'offsets': offsets,
        'parents': parents,
        'names': names,
        'order': order,
        'frametime': frametime
    }

def save(filename, data, translations=False):

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }

    rots, poss, offsets, parents = [
        data['rotations'],
        data['positions'],
        data['offsets'],
        data['parents']]
    
    names = data.get('names', ["joint_" + str(i) for i in range(len(parents))])
    order = data.get('order', 'zyx')
    frametime = data.get('frametime', 1.0/60.0)
    
    with open(filename, 'w') as f:

        t = ""
        f.write("%sHIERARCHY\n" % t)
        f.write("%sROOT %s\n" % (t, names[0]))
        f.write("%s{\n" % t)
        t += '\t'

        f.write("%sOFFSET %f %f %f\n" % ((t,) + tuple(offsets[0])))
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % 
            (t, channelmap_inv[order[0]], 
                channelmap_inv[order[1]], 
                channelmap_inv[order[2]]))
        jseq = [0]       
        for i in range(len(parents)):
            if parents[i] == 0:
                t, jseq = save_joint(f, offsets, order, parents, names, t, i, jseq, translations=translations)

        t = t[:-1]
        f.write("%s}\n" % t)
        f.write("MOTION\n")
        f.write("Frames: %i\n" % len(rots))
        f.write("Frame Time: %f\n" % frametime)
        
        for i in range(rots.shape[0]):
            for j in jseq:
                
                if translations or j == 0:
                    f.write("%f %f %f %f %f %f " % (
                        poss[i,j,0], poss[i,j,1], poss[i,j,2], 
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))
                
                else:   
                    f.write("%f %f %f " % (
                        rots[i,j,0], rots[i,j,1], rots[i,j,2]))

            f.write("\n")
    
def save_joint(f, offsets, order, parents, names, t, i, jseq, translations=False):

    jseq.append(i)

    channelmap_inv = {
        'x': 'Xrotation',
        'y': 'Yrotation',
        'z': 'Zrotation',
    }
    
    f.write("%sJOINT %s\n" % (t, names[i]))
    f.write("%s{\n" % t)
    t += '\t'
  
    f.write("%sOFFSET %f %f %f\n" % ((t,) + tuple(offsets[i])))
    
    if translations:
        f.write("%sCHANNELS 6 Xposition Yposition Zposition %s %s %s \n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    else:
        f.write("%sCHANNELS 3 %s %s %s\n" % (t, 
            channelmap_inv[order[0]], channelmap_inv[order[1]], channelmap_inv[order[2]]))
    
    end_site = True
    
    for j in range(len(parents)):
        if parents[j] == i:
            t, jseq = save_joint(f, offsets, order, parents, names, t, j, jseq, translations=translations)
            end_site = False
    
    if end_site:
        f.write("%sEnd Site\n" % t)
        f.write("%s{\n" % t)
        t += '\t'
        f.write("%sOFFSET %f %f %f\n" % (t, 0.0, 0.0, 0.0))
        t = t[:-1]
        f.write("%s}\n" % t)
  
    t = t[:-1]
    f.write("%s}\n" % t)
    
    return t, jseq
