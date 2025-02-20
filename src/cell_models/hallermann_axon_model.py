import sys
import os
from os.path import join
import numpy as np
import neuron

h = neuron.h
import LFPy
import src.cell_models.hallermann_params as hp


soma_dict = hp.soma_dict
ais_dict = hp.ais_dict
node_dict = hp.node_dict
myelin_dict = hp.myelin_dict
axon_dict = hp.axon_dict

cell_models_folder = os.path.dirname(__file__)
hallermann_folder =  join(os.path.dirname(__file__), "HallermannEtAl2012")
mechanisms_folder = join(os.path.dirname(__file__), "HallermannEtAl2012")


def download_hallermann_model():

    print("Downloading Hallermann model")
    if sys.version < '3':
        from urllib2 import urlopen
    else:
        from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://modeldb.science/download/144526',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'HallermannEtAl2012_r1.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    # Remove NEURON GUI from model files:
    model_file_ = open(join(hallermann_folder, "Cell parameters.hoc"), 'r')
    new_lines = ""
    for line in model_file_:
        changes = line.replace('load_proc("nrn', '//load_proc("nrn')
        changes = changes.replace('load_file("nrn', '//load_file("nrn')
        new_lines += changes
    new_lines += "parameters()\ngeom_nseg()\ninit_channels()\n"

    model_file_.close()
    model_file_mod = open(join(hallermann_folder, "Cell parameters_mod.hoc"), 'w')
    model_file_mod.write(new_lines)
    model_file_mod.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hallermann_folder)

    if "win32" in sys.platform:
        warn("no autompile of NMODL (.mod) files on Windows.\n"
             + "Run mknrndll from NEURON bash in the folder "
               "L5bPCmodelsEH/mod and rerun example script")
        if not mod_pth in neuron.nrn_dll_loaded:
            neuron.h.nrn_load_dll(join(mod_pth, "nrnmech.dll"))
        neuron.nrn_dll_loaded.append(mod_pth)
    else:
        os.system('''
                  cd {}
                  nrnivmodl
                  '''.format(mod_pth))
        neuron.load_mechanisms(mod_pth)

def make_sec_from_dict(basename, sec_type_counter, param_dict,
                       startpos, endpos):

    secname = "{}[{}]".format(basename, sec_type_counter[basename])

    sec = h.Section(name=secname)

    sec_type_counter[basename] += 1
    sec.nseg = param_dict["nseg"]
    sec.Ra = param_dict["Ra"]
    sec.L = param_dict['morphology']["L"]

    for nseg, seg in enumerate(sec):
        seg.cm = param_dict["cm"][nseg]
        diam = param_dict['morphology']["diam"][nseg]
        seg.diam = diam

    if "pts3d" in param_dict['morphology']:
        for pt3d in param_dict['morphology']["pts3d"]:
            sec.pt3dadd(pt3d[0], pt3d[1], pt3d[2], pt3d[3])
    else:
        num_pts = len(param_dict['morphology']["diam"])
        if num_pts == 1:
            sec.pt3dadd(startpos[0], startpos[1], startpos[2], param_dict['morphology']["diam"][0])
            sec.pt3dadd(endpos[0], endpos[1], endpos[2], param_dict['morphology']["diam"][0])

        else:
            for pt_idx in range(num_pts):
                pos = startpos + pt_idx / (num_pts - 1) * (endpos - startpos)
                #print(pos, param_dict['morphology']["diam"][pt_idx])
                sec.pt3dadd(pos[0], pos[1], pos[2], param_dict['morphology']["diam"][pt_idx])
                #print(pos[0], pos[1], pos[2], param_dict['morphology']["diam"][pt_idx])
        # sec.pt3dadd(startpos[0], startpos[1], startpos[2], param_dict['morphology']["diam"][0])
        # sec.pt3dadd(endpos[0], endpos[1], endpos[2], param_dict['morphology']["diam"][-1])

    #print("param", param_dict['morphology']["L"], sec.L, len(param_dict['morphology']["diam"]))
    for mech in param_dict["density_mechs"]:
        sec.insert(mech)
        for param in param_dict["density_mechs"][mech]:
            for nseg, seg in enumerate(sec):
                value = param_dict["density_mechs"][mech][param][nseg]
                cmd = "seg.{}_{} = {}".format(param, mech, value)
                exec(cmd)

    for ion in param_dict["ions"]:
        for param in param_dict["ions"][ion]:
            for nseg, seg in enumerate(sec):
                value = param_dict["ions"][ion][param][nseg]
                cmd = "seg.{} = {}".format(param, value)
                exec(cmd)

    if h.ismembrane("ca_ion", sec=sec):
        sec.eca = 140
        h.ion_style("ca_ion",0,1,0,0,0)
        h.vshift_ca = 8

    sec.ena = 55
    sec.ek = -98

    return sec, sec_type_counter


def test_axon_extraction(dt, tstop):

    model_folder = join(mechanisms_folder)

    neuron.load_mechanisms(model_folder)

    # Define cell parameters
    cell_parameters = {          # various cell parameters,
        'morphology' : join(model_folder, '28_04_10_num19.hoc'),
        'v_init' : -80.,    # initial crossmembrane potential
        # 'e_pas' : -65.,     # reversal potential passive mechs
        'passive' : False,   # switch on passive mechs
        'nsegs_method' : 'lambda_f',
        'lambda_f' : 500.,
        'dt' : dt,   # [ms] dt's should be in powers of 2 for both,
        'tstart' : 0,    # start time of simulation, recorders start at t=0
        'tstop' : tstop,   # stop simulation at 200 ms. These can be overridden
                            # by setting these arguments i cell.simulation()
        "extracellular": False,
        "pt3d": False,
        'custom_code': [join(model_folder, 'Cell parameters.hoc'),
                        join(model_folder, 'charge_only_unmyelinated.hoc')]
    }

    cell = LFPy.Cell(**cell_parameters)
    # cell = scale_soma_diameter(cell, cell_parameters, 10)
    cell.set_rotation(x=np.pi/2, y=-0.1)
    #cell.set_pos(z=-np.max(cell.z) - 20)
    return cell


def get_sec_endpoint(sec):
    return sec.x3d(sec.n3d() - 1), sec.y3d(sec.n3d() - 1), sec.z3d(sec.n3d() - 1)


def make_myelinated_branch(parent_sec, parent_axis,
                           my_length, node_length, sec_list, split_angle_fraction,
                           sec_type_counter):

    parent_end = get_sec_endpoint(parent_sec)
    # parent_name = parent_sec.name()
    #parent_i = int(parent_name.split("[")[-1][:-1])

    # making branch # 1
    x_frac = np.random.random() # np.random.choice([0, 1])#
    y_frac = 1 - x_frac
    x_frac *= split_angle_fraction
    y_frac *= split_angle_fraction

    axon_axis_ = np.array([-x_frac, -y_frac, 0]) + parent_axis
    #axon_axis_ = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) #+ parent_axis
    axon_axis_ = axon_axis_ / np.sqrt(np.sum(axon_axis_**2))
    end_pos = parent_end + axon_axis_ * my_length
    # print(start_pos, end_pos)
    sec_myelin_left, sec_type_counter = make_sec_from_dict("my", sec_type_counter, myelin_dict, parent_end, end_pos)

    sec_myelin_left.connect(parent_sec(1), 0)
    sec_list.append(sec_myelin_left)

    start_pos = end_pos.copy()
    end_pos = start_pos + parent_axis * node_length
    sec_node_left, sec_type_counter = make_sec_from_dict("node", sec_type_counter, node_dict, start_pos, end_pos)

    sec_node_left.connect(sec_myelin_left(1), 0)
    sec_list.append(sec_node_left)

    # making branch # 2
    axon_axis_ = np.array([x_frac, y_frac, 0]) + parent_axis
    #axon_axis_ = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) #+ parent_axis
    axon_axis_ = axon_axis_ / np.sqrt(np.sum(axon_axis_**2))
    end_pos = parent_end + axon_axis_ * my_length
    # print(branc_point_start, end_pos)
    sec_myelin_right, sec_type_counter = make_sec_from_dict("my", sec_type_counter, myelin_dict, parent_end, end_pos)

    sec_myelin_right.connect(parent_sec(1), 0)
    sec_list.append(sec_myelin_right)

    start_pos = end_pos.copy()
    end_pos = start_pos + parent_axis * node_length
    sec_node_right, sec_type_counter = make_sec_from_dict("node", sec_type_counter, node_dict, start_pos, end_pos)

    sec_node_right.connect(sec_myelin_right(1), 0)
    sec_list.append(sec_node_right)

    return sec_list, sec_node_left, sec_node_right, sec_type_counter


def make_unmyelinated_branch(parent_sec, parent_axis,
                           axon_length, sec_list, split_angle_fraction,
                           sec_type_counter):

    parent_end = get_sec_endpoint(parent_sec)


    # making branch # 1
    x_frac = np.random.random() # np.random.choice([0, 1])#

    y_frac = 1 - x_frac
    x_frac *= split_angle_fraction
    y_frac *= split_angle_fraction

    axon_axis_ = np.array([-x_frac, -y_frac, 0]) + parent_axis
    #axon_axis_ = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) #+ parent_axis
    axon_axis_ = axon_axis_ / np.sqrt(np.sum(axon_axis_**2))
    end_pos = parent_end + axon_axis_ * axon_length
    # print(start_pos, end_pos)
    sec_axon_left, sec_type_counter = make_sec_from_dict("axon", sec_type_counter, axon_dict, parent_end, end_pos)

    sec_axon_left.connect(parent_sec(1), 0)
    sec_list.append(sec_axon_left)

    # making branch # 2
    axon_axis_ = np.array([x_frac, y_frac, 0]) + parent_axis
    #axon_axis_ = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(-1, 1)]) #+ parent_axis

    axon_axis_ = axon_axis_ / np.sqrt(np.sum(axon_axis_**2))
    end_pos = parent_end + axon_axis_ * axon_length

    sec_axon_right, sec_type_counter = make_sec_from_dict("axon", sec_type_counter, axon_dict, parent_end, end_pos)

    sec_axon_right.connect(parent_sec(1), 0)
    sec_list.append(sec_axon_right)

    return sec_list, sec_axon_left, sec_axon_right, sec_type_counter


def return_constructed_myelinated_axon(tstop, dt,
                                       axon_diam_scaling=1.0,
                                       my_length=30, axon_length=1000,
                                       make_passive=False):



    hp.set_additional_hallermann_params()

    node_length = node_dict['morphology']["L"]
    num_sec_repeats = int((axon_length) / my_length)# + 1
    my_length = my_length - node_length
    myelin_dict['morphology']["L"] = my_length

    axon_dict["morphology"]["diam"] = np.array(axon_dict["morphology"]["diam"]) * axon_diam_scaling
    node_dict["morphology"]["diam"] = np.array(node_dict["morphology"]["diam"]) * axon_diam_scaling
    myelin_dict["morphology"]["diam"] = np.array(myelin_dict["morphology"]["diam"]) * axon_diam_scaling

    axon_axis = np.array([0, 0, 1])

    sec_type_counter = {"soma": 0,
                        "axon": 0,
                        "my": 0,
                        "node": 0
                        }


    start_pos = np.array([0, 0, 0])
    end_pos = start_pos + axon_axis * node_length
    # axon, sec_type_counter = make_sec_from_dict("axon", sec_type_counter, ais_dict, start_pos, end_pos)
    sec_node, sec_type_counter = make_sec_from_dict("node", sec_type_counter,
                                                    node_dict, start_pos, end_pos)
    sec_list = []
    sec_list.append(sec_node)
    for i in range(int(num_sec_repeats)):

        end_pos = start_pos + axon_axis * my_length
        sec_myelin, sec_type_counter = make_sec_from_dict("my", sec_type_counter,
                                                          myelin_dict, start_pos, end_pos)

        sec_myelin.connect(sec_list[-1](1), 0)
        sec_list.append(sec_myelin)

        start_pos = end_pos.copy()
        end_pos = start_pos + axon_axis * node_length
        sec_node, sec_type_counter = make_sec_from_dict("node", sec_type_counter,
                                                        node_dict, start_pos, end_pos)

        sec_node.connect(sec_myelin(1), 0)

        start_pos = end_pos.copy()
        sec_list.append(sec_node)


    # neuron.h.define_shape()
    allsecs = h.SectionList(sec_list)

    cell_params = {
        'v_init': -77.55963918,
        'passive': False,
        'nsegs_method': "lambda_f",
        'lambda_f': 500.,
        'dt': dt,   # [ms] dt's should be in powers of 2 for both,
        'tstart': 0,    # start time of simulation, recorders start at t=0
        'tstop': tstop,
        "extracellular": True,
        "pt3d": False,
        "delete_sections": False}

    cell = LFPy.Cell(morphology=allsecs, **cell_params)
    if make_passive:
        from src.main import remove_active_mechanisms
        remove_list = node_dict["density_mechs"].keys()
        remove_list = [n for n in remove_list if not n == "pas"]
        remove_active_mechanisms(remove_list, cell)

    cell.somapos = [0, 0, 0]
    #cell.set_rotation(y=np.pi)
    #cell.set_pos(z=-np.max(cell.z) - 0)
    # print(cell.z[0])
    cell.sec_list = sec_list

    return cell


def return_constructed_unmyelinated_axon(tstop, dt,
                                       axon_diam_scaling=1.0,
                                       axon_length=1000,
                                       make_passive=False):


    hp.set_additional_hallermann_params()

    axon_sec_length = 20
    num_sec_repeats = int(axon_length / axon_sec_length)

    axon_dict["morphology"]["diam"] = np.array(axon_dict["morphology"]["diam"]) * axon_diam_scaling
    node_dict["morphology"]["diam"] = np.array(node_dict["morphology"]["diam"]) * axon_diam_scaling
    myelin_dict["morphology"]["diam"] = np.array(myelin_dict["morphology"]["diam"]) * axon_diam_scaling

    axon_axis = np.array([0, 0, 1])

    sec_type_counter = {"soma": 0,
                        "axon": 0,
                        "my": 0,
                        "node": 0
                        }

    start_pos = np.array([0, 0, 0])
    sec_list = []
    for i in range(num_sec_repeats):

        end_pos = start_pos + axon_axis * axon_sec_length
        sec, sec_type_counter = make_sec_from_dict("axon",
                                                   sec_type_counter,
                                                   axon_dict,
                                                   start_pos, end_pos)
        if i == 0:
            pass
        else:
            sec.connect(sec_list[-1](1), 0)

        start_pos = end_pos.copy()
        sec_list.append(sec)

    # neuron.h.define_shape()
    allsecs = h.SectionList(sec_list)
    cell_params = {
        'v_init': -77.55963918,    # initial crossmembrane potential
        'passive': False,   # switch on passive mechs
        'nsegs_method': "lambda_f",
        'lambda_f': 500.,
        'dt': dt,   # [ms] dt's should be in powers of 2 for both,
        'tstart': 0,    # start time of simulation, recorders start at t=0
        'tstop': tstop,
        "extracellular": True,
        "pt3d": False,
        "delete_sections": False}

    cell = LFPy.Cell(morphology=allsecs, **cell_params)
    if make_passive:
        from main import remove_active_mechanisms
        remove_list = axon_dict["density_mechs"].keys()
        remove_list = [n for n in remove_list if not n == "pas"]
        remove_active_mechanisms(remove_list, cell)

    cell.somapos = [0, 0, 0]
    #cell.set_rotation(y=np.pi)
    #cell.set_pos(z=-np.max(cell.z) - 0)
    cell.sec_list = sec_list

    return cell




def test_stim_cell(cell):

    stim_params = {'amp': -0.06,
                   'idx': 0,
                   'pptype': "ISyn",
                   'dur': 1e9,
                   'delay': 5}
    neuron.load_mechanisms("ext_stim_phase")
    synapse = LFPy.StimIntElectrode(cell, **stim_params)

    return cell, synapse



if not os.path.isdir(mechanisms_folder):
    download_hallermann_model()

neuron.load_mechanisms(mechanisms_folder)


if __name__ == '__main__':


    dt = 2**-6
    tstop = 15
    sim_type = ["extract_axon", "reconstruct_axon"][1]
    pid = os.fork()
    if pid == 0:
        cell_ext = test_axon_extraction(dt, tstop)

        print("EXTRACTED:")
        # for sec in cell.allseclist:
        #     print()
        #     print(sec.name())
        #     print(neuron.psection(sec))
            # print(sec.L, sec.diam, sec.Ra, sec.cm, sec.g_pas, sec.e_pas)


        cell_ext, synapse = test_stim_cell(cell_ext)
        # for sec in cell.allseclist:
        #     if "soma" in sec.name():
        #         stim = h.IClamp(0.5, sec=sec)
        #         stim.amp = 0.1
        #         stim.delay = 50
        #         stim.dur = 10
        cell_ext.simulate(rec_vmem=True, rec_imem=True)
        np.save("vmem_extracted.npy", cell_ext.vmem)


        os._exit(0)
    else:
        os.waitpid(pid, 0)

    pid = os.fork()
    if pid == 0:
        cell_ext = return_constructed_unmyelinated_axon(dt, tstop, 1)
        print("CONSTRUCTED:")
        # for sec in cell.allseclist:
        #     print()
        #     print(sec.name())
        #     print(neuron.psection(sec))
            # print(sec.L, sec.diam, sec.Ra, sec.cm, sec.g_pas, sec.e_pas)

        # for sec in cell.allseclist:
        #     if "soma" in sec.name():
        #         stim = h.IClamp(0.5, sec=sec)
        #         stim.amp = 0.1
        #         stim.delay = 50
        #         stim.dur = 10
        cell_ext, synapse = test_stim_cell(cell_ext)
        cell_ext.simulate(rec_vmem=True, rec_imem=True)
        np.save("vmem_constructed.npy", cell_ext.vmem)


        os._exit(0)
    else:
        os.waitpid(pid, 0)


        # cell = test_axon_extraction(dt, tstop)

    cell_ext = test_axon_extraction(dt, tstop)
    cell_rec = return_constructed_unmyelinated_axon(dt, tstop, 1)
    # cell_rec.set_rotation(x=np.pi / 2)

    for sec in cell_ext.allseclist:
        sec_dict = neuron.psection(sec)
        print(sec.name(), sec_dict)

    import matplotlib
    # matplotlib.use("AGG")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    # ax1 = fig.add_subplot(121, aspect=1, xlim=[-200, 200], ylim=[-900, 20])
    ax1 = fig.add_subplot(121, aspect=1, xlim=[-200, 200], ylim=[-150, 1400])
    ax2 = fig.add_subplot(122)

    sec_clrs = {"Myelin": 'olive',
        "dend": '0.6',
        "soma": 'k',
        'apic': '0.8',
        "axon": 'lightgreen',
        "node": 'r',
        "my": '0.3',
        "Unmyelin": 'salmon',
        "Node": 'r',
        "hilloc": 'lightblue',
        "hill": 'pink',}
    possible_names = ["Myelin", "axon", "Unmyelin", "Node", "node", "my",
                      "hilloc",
                      "hill", "apic", "dend", "soma"]
    used_clrs = []
    # # fig2 = plt.figure()
    # # ax_ = fig2.add_subplot(111)
    for idx in range(cell_ext.totnsegs):
        sec_name = cell_ext.get_idx_name(idx)[1]
        #print(sec_name)
        c = 'k'
        for ax_name in possible_names:
            if ax_name in sec_name:
                # print(ax_name, sec_name)
                c = sec_clrs[ax_name]
                if not ax_name in used_clrs:
                    used_clrs.append(ax_name)

        ax1.plot(cell_ext.x[idx], cell_ext.z[idx], '-',
                 c=c, clip_on=True, lw=np.sqrt(cell_ext.d[idx]) * 1)
    cell_ext.__del__()
    for idx in range(cell_rec.totnsegs):
        sec_name = cell_rec.get_idx_name(idx)[1]
        #print(sec_name)
        c = 'k'
        for ax_name in possible_names:
            if ax_name in sec_name:
                # print(ax_name, sec_name)
                c = sec_clrs[ax_name]
                if not ax_name in used_clrs:
                    used_clrs.append(ax_name)

        ax1.plot(cell_rec.x[idx], cell_rec.z[idx], '-',
                 c=c, clip_on=True, lw=np.sqrt(cell_rec.d[idx]) * 1)


    vmem_ext = np.load("vmem_extracted.npy")
    vmem_cst = np.load("vmem_constructed.npy")

    diff = np.max(np.abs(vmem_cst[[0, -1], :] - vmem_ext[[0, -1], :]))
    print("MAX DIFF:", diff)

    [ax2.plot(vmem_ext[idx], 'r') for idx in range(len(vmem_ext))]
    [ax2.plot(vmem_cst[idx], 'k--') for idx in range(len(vmem_cst))]
    plt.savefig("test_axon_construction_compare_unmyelinated.png")

    plt.show()