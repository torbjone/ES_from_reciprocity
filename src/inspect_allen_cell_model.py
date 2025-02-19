import os
from os.path import join
import json
import numpy as np
import matplotlib.pyplot as plt
import neuron
import LFPy
import src.cell_models

allen_models_folder = join(src.cell_models.__path__[0], "allen_cell_models")

def initiate_cell_models_folder():
    """
    If cell models folder do not exist, we make it and add two additional
    files needed in these simulations
    """

    print("Initiating cell models folder.")

    cwd = os.getcwd()
    if not os.path.isdir(allen_models_folder):
        # If the cell models folder do not exist yet, we make it
        os.makedirs(allen_models_folder)

    # Some morphologies come with axons, which are removed and replaced by a stub axon.
    # We make a .hoc file to do this
    if not os.path.isfile(join(allen_models_folder, "remove_axon.hoc")):
        file = open(join(allen_models_folder, "remove_axon.hoc"), 'w')
        file.write(
            """
            axon {delete_section()}
            
            create axon[2]
            axon[0]{
               L = 30
               diam = 1
               nseg = 1+2*int(L/40)
            }
            axon[1]{
               L = 30
               diam = 1
               nseg = 1+2*int(L/40)
            }
            
            nSecAxonal = 2
            connect axon(0), soma(0.5)
            connect axon[1](0), axon[0](1)
            access soma
            """)
        file.close()

    # Make a .mod file for stimulating the somas with synaptic step-currents
    # We make a .hoc file to do this
    if not os.path.isfile(join(allen_models_folder, "stim.mod")):
        file = open(join(allen_models_folder, "stim.mod"), 'w')
        file.write(
            """
            COMMENT
            Since this is an synapse current, positive values of i depolarize the cell
            and is a transmembrane current.
            ENDCOMMENT
            
            NEURON {
                POINT_PROCESS ISyn
                RANGE del, dur, amp, i
                NONSPECIFIC_CURRENT i
            }
            UNITS {
                (nA) = (nanoamp)
            }
            
            PARAMETER {
                del (ms)
                dur (ms)	<0,1e9>
                amp (nA)
            }
            ASSIGNED { i (nA) }
            
            INITIAL {
                i = 0
            }
            
            BREAKPOINT {
                at_time(del)
                at_time(del+dur)
                if (t < del + dur && t >= del) {
                    i = amp
                }else{
                    i = 0
                }
            }
            """)
        file.close()
        os.chdir(allen_models_folder)
        os.system("nrnivmodl")
        os.chdir(cwd)


def download_allen_model(cell_name="473863035"):
    '''
    Download model with id 'cell_name' from the Allen database
    '''


    import zipfile
    print("Downloading Allen model: ", cell_name)
    url = "https://api.brain-map.org/neuronal_model/download/%s" % cell_name

    cwd = os.getcwd()

    os.chdir(allen_models_folder)
    os.system("wget %s" % url)
    os.system("mv %s neuronal_model_%s.zip" % (cell_name, cell_name))

    os.chdir(cwd)

    myzip = zipfile.ZipFile(join(allen_models_folder, 'neuronal_model_{}.zip'.format(cell_name)), 'r')
    myzip.extractall(join(allen_models_folder, 'neuronal_model_{}'.format(cell_name)))
    myzip.close()
    os.remove(join(allen_models_folder, 'neuronal_model_{}.zip'.format(cell_name)))

    mod_folder = join(allen_models_folder, 'neuronal_model_{}'.format(cell_name), "modfiles")
    if not os.path.isdir(join(mod_folder, "x86_64")):
        print("Compiling mechanisms ...")
        cwd = os.getcwd()
        os.chdir(mod_folder)
        os.system("nrnivmodl")
        os.chdir(cwd)


def align_cell_to_axes(cell):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Rotates the cell such that it is aligned with the z-axis

    """
    from sklearn.decomposition import PCA

    points = np.array([cell.x.mean(axis=1), cell.y.mean(axis=1), cell.z.mean(axis=1)])
    pca = PCA(n_components=3)
    pca.fit(points[:3].T)
    x_axis = pca.components_[1]
    y_axis = np.asarray(pca.components_[2])

    y_axis = y_axis / np.linalg.norm(y_axis)

    dx = y_axis[0]
    dy = y_axis[1]
    dz = y_axis[2]

    x_angle = -np.arctan2(dz, dy)
    z_angle = np.arctan2(dx, np.sqrt(dy * dy + dz * dz))

    cell.set_rotation(x_angle, None, z_angle)
    if x_axis is None:
        return

    x_axis = np.asarray(x_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)

    Rx = rotation_matrix([1, 0, 0], x_angle)
    Rz = rotation_matrix([0, 0, 1], z_angle)

    x_axis = np.dot(x_axis, Rx)
    x_axis = np.dot(x_axis, Rz)

    dx = x_axis[0]
    dz = x_axis[2]

    y_angle = np.arctan2(dz, dx)
    cell.set_rotation(None, y_angle, None)

    if np.abs(np.min(cell.z)) > np.abs(np.max(cell.z)):
        cell.set_rotation(x=np.pi)


def rotation_matrix(axis, theta):
    """
    Based on code from: https://github.com/lastis/LFPy_util/
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    Uses the Euler-rodrigues formula
    """
    theta = -theta
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def return_allen_cell_model(cell_name, dt, tstop, make_passive=False):
    '''
    Loads a cell model from the Allen database. It reads the biophysical parameters from file
    to make an LFPy Cell object. The method has earlier been tested agains the AllenSDK and BMTK
    versions running the same cell models.
    '''

    model_folder = join(allen_models_folder, "neuronal_model_%s" % cell_name)

    if not os.path.isdir(allen_models_folder):
        initiate_cell_models_folder()

    print(model_folder)
    if not os.path.isdir(model_folder):
        download_allen_model(cell_name)

    mod_folder = join(model_folder, "modfiles")
    if not os.path.isdir(join(mod_folder, "x86_64")):
        print("Compiling mechanisms ...")
        cwd = os.getcwd()
        os.chdir(mod_folder)
        os.system("nrnivmodl")
        os.chdir(cwd)

    if not hasattr(neuron.h, "Ca_HVA"):
    # try:
        neuron.load_mechanisms(mod_folder)
    # except:
        # pass

    if not hasattr(neuron.h, "ISyn"):
        neuron.load_mechanisms(allen_models_folder)

    model_file = join(model_folder, "fit_parameters.json")
    manifest_file = join(model_folder, "manifest.json")
    metadata_file = join(model_folder, "model_metadata.json")
    morph_file = join(model_folder, "reconstruction.swc")

    params = json.load(open(model_file, 'r'))
    manifest = json.load(open(manifest_file, 'r'))
    metadata = json.load(open(metadata_file, 'r'))
    model_type = manifest["biophys"][0]["model_type"]

    Ra = params["passive"][0]["ra"]

    # For some reason the parameters are stored somewhat differently for
    # 'perisomatic' and 'all active' cell models
    if model_type == "Biophysical - perisomatic":
        e_pas = params["passive"][0]["e_pas"]
        cms = params["passive"][0]["cm"]

    celsius = params["conditions"][0]["celsius"]
    neuron.h.celsius = celsius

    reversal_potentials = params["conditions"][0]["erev"]
    v_init = params["conditions"][0]["v_init"]
    active_mechs = params["genome"]

    # Define cell parameters
    cell_parameters = {
        'morphology': morph_file,
        'v_init': -70,  # initial membrane potential
        'passive': False,  # if True, insert default passive parameters
        'nsegs_method': 'fixed_length',  # spatial discretization method
        'max_nsegs_length': 10.,
        'dt': dt,  # simulation time step size
        'tstart': -100,  # start time of simulation, recorders start at t=0
        'tstop': tstop,
        'pt3d': True,
        'extracellular': True,
        'custom_code': [join(allen_models_folder, 'remove_axon.hoc')]  # Custom file to delete axon if present
    }

    cell = LFPy.Cell(**cell_parameters)
    cell.metadata = metadata
    cell.manifest = manifest

    if make_passive and model_type != "Biophysical - perisomatic":
        raise RuntimeError("Not implemented!")

    for sec in neuron.h.allsec():
        sec.insert("pas")
        sectype = sec.name().split("[")[0]
        if model_type == "Biophysical - perisomatic":
            sec.e_pas = e_pas
            for cm_dict in cms:
                if cm_dict["section"] == sectype:
                    exec("sec.cm = {}".format(cm_dict["cm"]))
        sec.Ra = Ra

        if not make_passive:
            for sec_dict in active_mechs:
                if sec_dict["section"] == sectype:
                    if not sec_dict["mechanism"] == "":
                        if not sec.has_membrane(sec_dict["mechanism"]):
                            sec.insert(sec_dict["mechanism"])
                    exec("sec.{} = {}".format(sec_dict["name"], sec_dict["value"]))

            for sec_dict in reversal_potentials:
                if sec_dict["section"] == sectype:
                    for key in sec_dict.keys():
                        if not key == "section":
                            exec("sec.{} = {}".format(key, sec_dict[key]))

    neuron.h.secondorder = 0
    align_cell_to_axes(cell)
    cell.set_rotation(z=np.pi)
    cell.set_rotation(x=np.pi)
    return cell


def insert_current_stimuli(cell, amp):
    stim_params = {'amp': amp,
                   'idx': 0,
                   'pptype': "ISyn",
                   'dur': 1e9,
                   'delay': 5}
    synapse = LFPy.StimIntElectrode(cell, **stim_params)
    return synapse, cell



def run_cell_model(cell_name, dt, tstop):

    #download_allen_model(cell_name=str(cell_name))

    # Make cell model and run simulation
    cell = return_allen_cell_model(cell_name, dt, tstop)
    synapse, cell = insert_current_stimuli(cell, -0.3)
    cell.simulate(rec_vmem=True, rec_imem=True)

    # Extract some information about the cell model
    model_type = cell.manifest["biophys"][0]["model_type"].split("-")[1]
    cell_type = cell.metadata["specimen"]["specimen_tags"][1]["name"].split("-")[1]
    cell_region = cell.metadata["specimen"]["structure"]["name"]
    cell_layer = cell.metadata["specimen"]["structure"]["name"].split(",")[1]
    cell_info = model_type + cell_type + ' ' + cell_region

    print(cell_name, cell_info)

    # Make a control plot of the cell, its membrane potential, and extracellular potential at
    # a single nearby electrode
    plt.close("all")
    fig = plt.figure(figsize=[16, 10])
    fig.subplots_adjust(wspace=0.4)
    fig.suptitle(cell_info)
    ax_m = fig.add_subplot(141, xlabel="x (µm)", ylabel="z (µm)", aspect=1,
                           xlim=[-150, 150], ylim=[-150, 500])
    ax_m2 = fig.add_subplot(142, xlabel="y (µm)", ylabel="z (µm)", aspect=1,
                           xlim=[-150, 150], ylim=[-150, 500])

    ax_dist_x = fig.add_subplot(343, xlabel="NEURON dist", ylabel="x")
    ax_dist_y = fig.add_subplot(347, xlabel="NEURON dist", ylabel="y")
    ax_dist_z = fig.add_subplot(3,4,11, xlabel="NEURON dist", ylabel="z")

    ax_dist_x.grid(True)
    ax_dist_y.grid(True)
    ax_dist_z.grid(True)

    ax1 = fig.add_subplot(144, xlabel="time (ms)", ylabel=r"somatic $V_{\rm m}$ (mV)")

    # zips = []
    # for x1, x2 in cell.get_idx_polygons():
    #     zips.append(list(zip(x1, x2)))
    # polycol = PolyCollection(zips,
    #                          edgecolors='none',
    #                          facecolors='k',
    #                          rasterized=True)
    #
    # ax_m.add_collection(polycol)
    #
    #
    # zips = []
    # for x1, x2 in cell.get_idx_polygons(projection=('y', 'z')):
    #     zips.append(list(zip(x1, x2)))
    # polycol2 = PolyCollection(zips,
    #                          edgecolors='none',
    #                          facecolors='k',
    #                          rasterized=True)
    #
    # ax_m2.add_collection(polycol2)
    name_clr_dict = {'soma': 'k', 'apic': 'orange', 'dend': 'gray', 'axon': 'cyan'}

    c_idx = 0
    soma_sec = None
    for sec in neuron.h.allsec():
        if 'soma' in sec.name():
            soma_sec = sec
            #neuron.h.distance(0.5, sec=sec)
        for seg in sec:

            _, c_name, c_x = cell.get_idx_name(c_idx)
            dist = neuron.h.distance(soma_sec(0.5), sec(seg.x))
            print(c_name, c_x)
            c_name = c_name.split('[')[0]
            clr = name_clr_dict[c_name]

            ax_dist_x.plot(dist, cell.x[c_idx].mean(), 'o', c=clr)
            ax_dist_y.plot(dist, cell.y[c_idx].mean(), 'o', c=clr)
            ax_dist_z.plot(dist, cell.z[c_idx].mean(), 'o', c=clr)
            if c_idx == 0:
                ax_m.plot(cell.x[c_idx].mean(), cell.z[c_idx].mean(), 'o', c=clr)
                ax_m2.plot(cell.y[c_idx].mean(), cell.z[c_idx].mean(), 'o', c=clr)
            else:
                ax_m.plot(cell.x[c_idx], cell.z[c_idx], c=clr)
                ax_m2.plot(cell.y[c_idx], cell.z[c_idx], c=clr)
            c_idx += 1
    ax1.plot(cell.tvec, cell.somav, 'k')

    fig.savefig(join(allen_models_folder, "cell_name_%s.png" % cell_name))

    elec = None
    synapse = None
    cell.__del__()

if __name__ == "__main__":

    dt = 2**-5
    tstop = 200
    if not os.path.isdir(allen_models_folder):
        initiate_cell_models_folder()

    model_ids = [
        # 497232692,
        # 486052412,
        491623973,
        # 488462783,
        # 497232978,
        # 480631286,
        # 478513398
    ]
    for model_id in model_ids:
        run_cell_model(model_id, dt, tstop)