
import os
import sys
from os.path import join
import numpy as np
import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
import scipy.fftpack as ff
import LFPy
from lfpykit.cellgeometry import CellGeometry
from lfpykit.models import CurrentDipoleMoment
from lfpykit.eegmegcalc import NYHeadModel
import neuron
from src.plotting_convention import mark_subplots, simplify_axes, cmap_v_e
from src.inspect_allen_cell_model import return_allen_cell_model
import src.cell_models.hallermann_axon_model as ha
from mpl_toolkits.mplot3d import art3d
h = neuron.h

root_dir = join(os.path.dirname(os.path.realpath(__file__)), '..')
cell_models_folder = join(root_dir, 'src', 'cell_models')

cell_models_folder = os.path.abspath(join(cell_models_folder))
hay_folder = join(cell_models_folder, "L5bPCmodelsEH")
allen_folder = join(cell_models_folder, "allen_cell_models")
mrg_axon_folder = join(cell_models_folder, "MRGaxon")

os.makedirs(cell_models_folder, exist_ok=True)

neuron.load_mechanisms(cell_models_folder)

cell_input_subthreshold_amp = {'hay': 0.1,
                               'allen_491623973': 0.01,
                               'myelinated_axon': 0.001,
                               'MRG_axon': 0.001,
                               }

stim_clr = 'gray'
ipas_clr = 'purple'
icap_clr = 'orange'
im_clr = 'pink'

sigma_T = 0.3


def download_hay_model():

    print("Downloading Hay model")

    from urllib.request import urlopen
    import ssl
    from warnings import warn
    import zipfile
    #get the model files:
    u = urlopen('https://modeldb.science/download/139653',
                context=ssl._create_unverified_context())
    localFile = open(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'wb')
    localFile.write(u.read())
    localFile.close()
    #unzip:
    myzip = zipfile.ZipFile(join(cell_models_folder, 'L5bPCmodelsEH.zip'), 'r')
    myzip.extractall(cell_models_folder)
    myzip.close()

    #compile mod files every time, because of incompatibility with Mainen96 files:
    mod_pth = join(hay_folder, "mod/")

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


def return_MRG_axon_cell(tstop, dt, make_passive):
    neuron.load_mechanisms(mrg_axon_folder)
    cell_parameters = {
        'morphology': join(mrg_axon_folder, "MRGaxon_mod.hoc"),
        'passive': False,
        'nsegs_method': None,
        'dt': dt,
        'tstart': -100,
        'tstop': tstop,
        'v_init': -80,
        'pt3d': True,
        'extracellular': True,
    }
    cell = LFPy.Cell(**cell_parameters)

    if make_passive:
        for sec in cell.allseclist:
            for seg in sec:
                if hasattr(seg, "axnode"):
                    seg.gnabar_axnode = 0
                    seg.gnapbar_axnode = 0
                    seg.gkbar_axnode = 0

        #raise NotImplementedError()

    cell.set_rotation(y=-np.pi/2)
    return cell


def return_hay_cell(tstop, dt, make_passive=False):
    if not os.path.isfile(join(hay_folder, 'morphologies', 'cell1.asc')):
        download_hay_model()

    if make_passive:
        cell_params = {
            'morphology': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': True,
            'passive_parameters': {"g_pas": 1 / 15000,
                                   "e_pas": -70.},
            'nsegs_method': "lambda_f",
            "Ra": 100,
            "cm": 1.0,
            "lambda_f": 100,
            'dt': dt,
            'tstart': -1,
            'tstop': tstop,
            'v_init': -70,
            'pt3d': True,
            'extracellular': True,
        }

        cell = LFPy.Cell(**cell_params)
        cell.set_rotation(x=4.729, y=-3.166)

        return cell
    else:
        if not hasattr(neuron.h, "CaDynamics_E2"):
            neuron.load_mechanisms(join(hay_folder, 'mod'))
        cell_params = {
            'morphology': join(hay_folder, "morphologies", "cell1.asc"),
            'templatefile': [join(hay_folder, 'models', 'L5PCbiophys3.hoc'),
                             join(hay_folder, 'models', 'L5PCtemplate.hoc')],
            'templatename': 'L5PCtemplate',
            'templateargs': join(hay_folder, 'morphologies', 'cell1.asc'),
            'passive': False,
            'nsegs_method': None,
            'dt': dt,
            'tstart': -200,
            'tstop': tstop,
            'v_init': -75,
            'celsius': 34,
            'pt3d': True,
            'extracellular': True,
        }

        cell = LFPy.TemplateCell(**cell_params)

        cell.set_rotation(x=4.729, y=-3.166)
        return cell


def return_2comp_cell(tstop, dt, make_passive=False,
                      set_extracelluar=True):

    neuron.h.celsius = 33
    h("forall delete_section()")

    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1]

    proc topol() { local i
      basic_shape()
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -0., 1.)
      pt3dadd(0, 0, 200., 1.)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 2}

    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert pas // 'pas' for passive, 'hh' for Hodgkin-Huxley
        }
    """)
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100,
                'tstop': tstop,
                'pt3d': True,
                'extracellular': set_extracelluar,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(z=-cell.z[0])
    return cell


def return_stick_cell(tstop, dt, apic_diam=1,
                      set_extracelluar=True, make_passive=False):

    neuron.h.celsius = 33
    mech = "pas" if make_passive else "hh"
    h("forall delete_section()")

    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1]//, dend[1]

    proc topol() { local i
      basic_shape()
      //connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -0., 1.)
      pt3dadd(0, 0, 1000., 1.)}
      //dend[0] {pt3dclear()
      //pt3dadd(0, 0, 10., %s)
      //pt3dadd(0, 0, 1000, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        //dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 300}
    //dend[0] {nseg = 300}
    }
    proc biophys() {
    }
    celldef()

    Ra = 100.
    cm = 1.
    Rm = 15000.

    forall {
        insert %s // 'pas' for passive, 'hh' for Hodgkin-Huxley
        //g_pas = 1 / Rm
        }
    """ % (apic_diam, apic_diam, mech))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100,
                'tstop': tstop,
                'pt3d': True,
                'extracellular': set_extracelluar,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(z=-cell.z[0])
    return cell


def return_ball_and_stick_cell(tstop, dt):

    neuron.h.celsius = 33

    soma = h.Section(name='soma')
    soma.nseg = 1
    
    dend = h.Section(name='dend')
    dend.nseg = 600

    Ra = 100.
    cm = 1.
    Rm = 15000.

    dend.connect(soma(1), 0)
    
    all_sec = h.SectionList()
    for sec in h.allsec():
        # print(sec.name())
        all_sec.append(sec=sec)
        sec.insert("pas")

        sec.g_pas = 1 / Rm
        sec.Ra = Ra
        sec.cm = cm
        h.pt3dclear()
   
    h.pt3dadd(0, 0, -5, 10, sec=soma)
    h.pt3dadd(0, 0, 5, 10, sec=soma)
    h.pt3dadd(0, 0, 5., 1, sec=dend)
    h.pt3dadd(0, 0, 1005, 1, sec=dend)
    h.define_shape()
    cell_params = {
                'morphology': all_sec,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -dt,
                'tstop': tstop,
                'pt3d': True,
                'extracellular': True,
            }
    cell = LFPy.Cell(**cell_params)

    cell.seclist_ = [soma, dend]
    # cell.set_pos()
    # cell.set_pos(z=-cell.z[0])
    # cell.__soma__ = soma
    # cell.__dend__ = dend

    return cell


def return_axon_cell(tstop, dt, apic_diam=1,
                      set_extracelluar=True, make_passive=False):

    neuron.h.celsius = 33
    mech = "pas" if make_passive else "hh"
    h("forall delete_section()")

    h("""
    proc celldef() {
      topol()
      subsets()
      geom()
      biophys()
      geom_nseg()
    }

    create soma[1]//, dend[1]

    proc topol() { local i
      basic_shape()
      //connect dend(0), soma(1)
    }
    proc basic_shape() {
      soma[0] {pt3dclear()
      pt3dadd(0, 0, -0., 1.)
      pt3dadd(0, 0, 5000., 1.)}
      //dend[0] {pt3dclear()
      //pt3dadd(0, 0, 10., %s)
      //pt3dadd(0, 0, 1000, %s)}
    }

    objref all
    proc subsets() { local i
      objref all
      all = new SectionList()
        soma[0] all.append()
        //dend[0] all.append()

    }
    proc geom() {
    }
    proc geom_nseg() {
    soma[0] {nseg = 500}
    //dend[0] {nseg = 300}
    }
    proc biophys() {
    }
    celldef()

    Ra = 150.
    cm = 1.
    Rm = 30000.

    forall {
        insert %s // 'pas' for passive, 'hh' for Hodgkin-Huxley
        //g_pas = 1 / Rm
        }
    """ % (apic_diam, apic_diam, mech))
    cell_params = {
                'morphology': h.all,
                'delete_sections': False,
                'v_init': -70.,
                'passive': False,
                'nsegs_method': None,
                'dt': dt,
                'tstart': -100,
                'tstop': tstop,
                'pt3d': True,
                'extracellular': set_extracelluar,
            }
    cell = LFPy.Cell(**cell_params)
    cell.set_pos(z=-cell.z[0])
    return cell


def remove_active_mechanisms(remove_list, cell):
    # remove_list = ["Nap_Et2", "NaTa_t", "NaTs2_t", "SKv3_1",
    # "SK_E2", "K_Tst", "K_Pst", "KdShu2007",
    # "Im", "Ih", "CaDynamics_E2", "Ca_LVAst", "Ca", "Ca_HVA",
    #'StochKv']
    mt = h.MechanismType(0)
    for sec in h.allsec():
        for seg in sec:
            for mech in remove_list:
                mt.select(mech)
                mt.remove(sec=sec)
    return cell


def make_extracellular_stimuli(dbs_params, cell, insert=True):
    """ Function to calculate and apply external potential """

    if dbs_params['position'] is None:
        # Assuming uniform electric field, giving linear extracellular potential
        ext_field = np.vectorize(lambda x, y, z: z / 1000)
    else:
        x0, y0, z0 = dbs_params['position']

        ext_field = np.vectorize(lambda x, y, z: 1 / (4 * np.pi * sigma_T * np.sqrt(
            (x0 - x) ** 2 + (y0 - y) ** 2 + (z0 - z) ** 2)))

    ### MAKING THE EXTERNAL FIELD
    n_tsteps_ = int(cell.tstop / cell.dt + 1)

    t_ = np.arange(n_tsteps_) * cell.dt
    pulse = np.zeros(n_tsteps_)


    if dbs_params['stim_type'] == 'pulse':
        start_time = dbs_params['start_time']
        end_time = dbs_params['end_time']
        start_idx = np.argmin(np.abs(t_ - start_time))
        end_idx = np.argmin(np.abs(t_ - end_time))
        pulse[start_idx:end_idx] = dbs_params['amp']
    elif dbs_params['stim_type'] == 'sine':
        if not 'frequency' in dbs_params:
            raise ValueError('Need "frequency" argument!')
        pulse = dbs_params['amp'] * np.sin(2 * np.pi * dbs_params['frequency'] *
                                                              t_ / 1000)
    else:
        raise RuntimeError('Unknown stimulus type!')
    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
    v_cell_ext[:, :] = ext_field(cell.x.mean(axis=-1),
                                 cell.y.mean(axis=-1),
                                 cell.z.mean(axis=-1)
                                 ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps_)
    if insert:
        cell.insert_v_ext(v_cell_ext, t_)
    return ext_field, v_cell_ext, pulse, cell



def return_amp_and_phase(sig, num_tsteps, dt):
    """ Returns the amplitude and frequency of
    the input signal at given frequency"""

    sig = np.array(sig)
    if len(sig.shape) == 1:
        sig = np.array([sig])
    elif len(sig.shape) == 2:
        pass
    else:
        raise RuntimeError("Not compatible with given array shape!")

    sample_freq = ff.fftfreq(num_tsteps, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]

    Y = ff.fft(sig, axis=1)[:, pidxs[0]]

    phase = np.angle(Y[:, :], deg=True)
    amplitude = np.abs(Y[:, :])/Y.shape[1]
    return freqs, amplitude, phase



def make_sinusoidal_input(cell, freq, phase=0):
    """ White Noise input ala Linden 2010 is made """
    tot_ntsteps = round((cell.tstop ) / cell.dt + 1)
    #I = np.zeros(tot_ntsteps)
    tvec = np.arange(tot_ntsteps) * cell.dt
    I = np.sin(2 * np.pi * freq * tvec/1000. + phase)
    return I


def make_sinusoidal_stimuli(cell, input_idx, weight, freq, phase=0):

    input_scaling = 1
    np.random.seed(1234)
    input_array = input_scaling * make_sinusoidal_input(cell, freq, phase)

    noise_vec = (neuron.h.Vector(input_array) if weight is None
                 else neuron.h.Vector(input_array * weight))

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, np.array(noise_vec), noise_vec



def make_white_noise_stimuli(cell, input_idx, weight, freqs):

    input_scaling = 1
    np.random.seed(1234)
    tot_ntsteps = round((cell.tstop ) / cell.dt + 1)
    tvec = np.arange(tot_ntsteps) * cell.dt
    I = np.zeros(tot_ntsteps)
    for freq in freqs:
        I += np.cos(2 * np.pi * freq * tvec/1000. + np.random.uniform(0, 2 * np.pi))

    input_array = input_scaling * I

    noise_vec = (neuron.h.Vector(input_array) if weight is None
                 else neuron.h.Vector(input_array * weight))

    i = 0
    syn = None
    for sec in cell.allseclist:
        for seg in sec:
            if i == input_idx:
                print("Input inserted in ", sec.name())
                syn = neuron.h.ISyn(seg.x, sec=sec)
            i += 1
    if syn is None:
        raise RuntimeError("Wrong stimuli index")
    syn.dur = 1E9
    syn.delay = 0
    noise_vec.play(syn._ref_amp, cell.dt)
    return cell, syn, np.array(noise_vec), noise_vec





def do_RT_sim(cell_name, amp, sim_name, stim_idx, stim_freq,
              make_passive, stim_type, pos, tstop, cutoff, dt, results_folder, fig_folder):

    cell = return_cell_model(tstop, dt, cutoff, make_passive, cell_name)

    if stim_type == 'extracellular':
        dbs_params = {'position': pos,
                      'amp': amp,  # nA,
                      'stim_type': 'sine',
                      'frequency': stim_freq,
                      }
        ext_field, v_cell_ext, pulse, cell = make_extracellular_stimuli(
            dbs_params, cell)
    elif stim_type == 'intracellular':
        cell, syn1, pulse, b1 = make_sinusoidal_stimuli(cell, stim_idx,
                                                        -amp,
                                                        stim_freq)
    else:
        raise RuntimeError()

    cell.simulate(rec_vmem=True, rec_imem=True)

    cutoff_idx = np.argmin(np.abs(cell.tvec - cutoff))

    cell.tvec = cell.tvec[cutoff_idx:] - cell.tvec[cutoff_idx]
    pulse = pulse[cutoff_idx:]
    cell.imem = cell.imem[:, cutoff_idx:]
    cell.vmem = cell.vmem[:, cutoff_idx:]

    print("vmem", np.std(cell.vmem[0, :]))

    if stim_type == 'extracellular':
        dV = cell.vmem[stim_idx, :] - np.mean(cell.vmem[stim_idx, :])

    elif stim_type == 'intracellular':
        elec_params = {
            'sigma': sigma_T,
            'x': np.array([pos[0]]),
            'y': np.array([pos[1]]),
            'z': np.array([pos[2]]),
            'method': 'pointsource',
        }
        electrode = LFPy.RecExtElectrode(cell, **elec_params)
        M = electrode.get_transformation_matrix()
        V_ex = M @ cell.imem
        dV = V_ex[0] - np.mean(V_ex[0, :])

        plt.close("all")
        plt.subplot(131)
        plt.plot(cell.tvec, cell.vmem[stim_idx])
        plt.subplot(132)
        plt.plot(cell.tvec, dV)
        plt.subplot(133)
        plt.plot(cell.tvec, pulse)

        plt.savefig(join(fig_folder, f"control_{sim_name}_{stim_type}.png"))

    np.savez(join(results_folder, f"{sim_name}_{stim_type}.npz"),
             dV=dV, Vm=cell.vmem[stim_idx, :])


def make_RT_validation_plot(results_folder, fig_folder):
    """
    Run simulations to validate that there is zero error for applying the reciprocity
    theorem for passive cell models, and low errors for active models in the subthreshold regime.
    """

    cell_names = ["hay",
                  "allen_491623973",
                  "myelinated_axon",
                  ]

    cell_legend_names = {'hay': 'rat PC',
                         "allen_491623973": 'mouse IN',
                         "myelinated_axon": "axon terminal", 
                         "MRG_axon": "axon terminal",
                         }
    cell_clrs = {
        cell_name: plt.cm.tab10(n / 3)
            for n, cell_name in enumerate(cell_names)}
    amps = [100, 500, 1000, 5000, 10000, 50000, 100000]
    stim_freqs = np.array([1, 10, 100, 1000])

    distances = [25, 50, 100]

    dt = 2**-5
    stim_idx = 0
    tstop = 2000 - dt
    sample_freq = ff.fftfreq(int((tstop / dt + 1)), d=dt / 1000)

    def return_simname(cell_name, make_passive, dist, stim_freq, amp):

        return (f'{cell_name}_passive:{make_passive}_' +
                f'{dist}um_{stim_freq}Hz_{amp:0.03f}nA')

    run_neural_sim = True
    if run_neural_sim:
        for make_passive in [True, False]:
            for cell_name in cell_names:
                for dist in distances:
                    pos = [dist, 0, 0]
                    for stim_freq in stim_freqs:
                        if make_passive:
                            cutoff = 1000 + 1000 / stim_freq / 4
                        else:
                            cutoff = 6000 + 1000 / stim_freq / 4
                        
                        stim_freq_ = sample_freq[np.argmin(np.abs(sample_freq - stim_freq))]
                        if stim_freq != stim_freq_:
                            raise RuntimeError("Stimulation frequency not as expected ")
                        
                        amp_i = cell_input_subthreshold_amp[cell_name]
                        sim_name = return_simname(cell_name, make_passive, 
                                                  dist, stim_freq, amp_i)
                        pid = os.fork()
                        if pid == 0:
                            do_RT_sim(cell_name, amp_i,
                                    sim_name, stim_idx, stim_freq, make_passive, 
                                    "intracellular", pos, tstop, cutoff, dt, results_folder, fig_folder)
                            os._exit(0)
                        else:
                            os.waitpid(pid, 0)

                        for amp_e in amps:
                            sim_name = return_simname(cell_name, make_passive, dist, stim_freq, amp_e)

                            pid = os.fork()
                            if pid == 0:
                                do_RT_sim(cell_name, amp_e, sim_name, stim_idx, stim_freq, 
                                        make_passive, 'extracellular', pos, tstop, cutoff,
                                          dt, results_folder, fig_folder)
                                os._exit(0)
                            else:
                                os.waitpid(pid, 0)

                            sim_name_e = return_simname(cell_name, make_passive, 
                                                        dist, stim_freq, amp_e)
                            sim_name_i = return_simname(cell_name, make_passive, 
                                                        dist, stim_freq, amp_i)
                            dV_ctrl = np.load(join(results_folder,
                                                f"{sim_name_e}_extracellular.npz"))["dV"]

                            ctrl_soma_vm = np.load(join(results_folder,
                                                f"{sim_name_e}_extracellular.npz"))["Vm"]

                            dV_RT = np.load(join(results_folder,
                                                f"{sim_name_i}_intracellular.npz"))["dV"]
                            
                            dV_RT *= amp_e / amp_i
                            # error = np.max(np.abs(dV_RT - dV_ctrl))/np.std(dV_ctrl)
                            error = np.std(dV_RT - dV_ctrl)/np.std(dV_ctrl)

                            tvec = np.arange(len(dV_ctrl)) * dt
                            plt.close("all")
                            fig = plt.figure(figsize=[6, 6])
                            fig.subplots_adjust(top=0.9)
                            plt.subplot(311, ylabel="mV")
                            plt.plot(tvec, dV_ctrl, label="extracellular stim",
                                    lw=0.5)
                            plt.plot(tvec, dV_RT, '--', label="inracellular stim",
                                    lw=0.5)
                            plt.legend(frameon=False)
                            plt.subplot(312, ylabel="difference (mV)", 
                                        title=f"rel. error: {error:0.3f}")
                            plt.plot(tvec, dV_ctrl - dV_RT, 'k', lw=0.5)
            
                            plt.subplot(313, 
                                        title=f"Soma Vm, ext stim (SD: {np.std(ctrl_soma_vm)})",
                                        ylabel="mV")
                            plt.plot(tvec, ctrl_soma_vm, 'k', lw=0.5)

                            simplify_axes(fig.axes)
                            plt.savefig(join(fig_folder, f"control_{sim_name_e}.png"))

        
    plot_cell_name = cell_names[0]
    make_passive = False
    cell = return_cell_model(tstop, dt, 0, make_passive, plot_cell_name)
    amp_i = cell_input_subthreshold_amp[plot_cell_name]
    plot_dist = 50
    pos = [plot_dist, 0, 0]

    plt.close("all")
    fig = plt.figure(figsize=[6, 2.5])
    fig.subplots_adjust(top=0.9, bottom=0.12, right=0.95, 
                        left=-0.2, wspace=0.6)

    ax_m = fig.add_axes([0.01, 0.01, 0.12, 0.95], aspect=1, xlim=[-200, 200],
                          frameon=False, xticks=[], yticks=[])

    ax_amp = fig.add_axes([0.67, 0.15, 0.3, 0.6], 
                          xlabel=r"$V_{\rm m}$ amplitude (mV)",
                          ylabel="relative error", ylim=[-0.1, 1],
                          xlim=[5e-2, 10e1],
                          )

    ax_m.plot([100, 100], [300, 500], c='k', lw=1)

    ax_m.text(110, 400, "200\nµm", va='center', ha='left')
    ax_m.plot(cell.x.T, cell.z.T, c='k', lw=0.8)
    ax_m.plot(cell.x[0].mean(), cell.z[0].mean(), c='g', marker='o',
              mew=0.5, mec='k')
    ax_m.plot(pos[0], pos[2], 'o', c='orange', mew=0.5, mec='k')

    # axis cross
    ax_m.arrow(-150, -250, 100, 0, head_width=8, fc='k', clip_on=False)
    ax_m.arrow(-150, -250, 0, 100, head_width=8, fc='k', clip_on=False)
    ax_m.plot(-150, -250, 'k.', ms=0.5)
    ax_m.text(-150 + 50, -255, "$x$", va="top", ha="center")
    ax_m.text(-155, -250 + 50, "$z$", va="center", ha="right")

    sig_axes = []
    plot_amp_es = [1000, 10000, 10000]
    plot_freqs_es = [10, 1000, 10]
    p_markers = ['s', 'D', '^']

    for aidx, stim_freq in enumerate(plot_freqs_es):

        amp_e = plot_amp_es[aidx]

        if stim_freq == 1:
            tlim = 2000
        elif stim_freq == 10:
            tlim = 500
        elif stim_freq == 100:
            tlim = 60
        elif stim_freq == 1000:
            tlim = 10

        if aidx == 0:
            t_shift = 0
        else:
            t_shift = -290

        sim_name_e = return_simname(plot_cell_name, make_passive, plot_dist, 
                                    stim_freq, amp_e)
        sim_name_i = return_simname(plot_cell_name, make_passive, plot_dist, 
                                    stim_freq, amp_i)

        dV_ctrl = np.load(join(results_folder, 
                            f"{sim_name_e}_extracellular.npz"))["dV"]
        ctrl_soma_vm = np.load(join(results_folder,
                                            f"{sim_name_e}_extracellular.npz"))["Vm"]       
        dV_RT = np.load(join(results_folder, 
                            f"{sim_name_i}_intracellular.npz"))["dV"]
        dV_RT *= amp_e / amp_i

        f_, vm_amp, vm_phase = return_amp_and_phase(ctrl_soma_vm, len(ctrl_soma_vm), dt)
        freq_idx = np.argmin(np.abs(f_ - stim_freq))
        vm_amp = vm_amp[0, freq_idx]

        v_amp_ctrl = np.max(np.abs(dV_ctrl))
        v_amp_RT = np.max(np.abs(dV_RT))
        error = np.std(dV_RT - dV_ctrl)/np.std(dV_ctrl)

        ax_amp.plot(vm_amp, error, c=cell_clrs[plot_cell_name], 
                    marker=p_markers[aidx], zorder=1000)
        
        ax_ = fig.add_subplot(3, 3, 2 + aidx * 3, frameon=False,
                          xticks=[], yticks=[], 
                          xlim=[tstop - tlim + t_shift, tstop + t_shift],
                          ylim=[np.min(dV_ctrl)*1.1, np.max(dV_ctrl) * 1.1])
        sig_axes.append(ax_)

        cell.tvec = np.arange(len(dV_ctrl)) * cell.dt
        l_ctrl, = ax_.plot(cell.tvec, dV_ctrl, 'k')
        l_RT, = ax_.plot(cell.tvec, dV_RT, '--', c=cell_clrs[plot_cell_name])

        ax_.plot(cell.tvec[-1] - tlim + t_shift - tlim/10, np.min(dV_ctrl) * 0.5, 
                 c=cell_clrs[plot_cell_name], 
                    marker=p_markers[aidx], zorder=1000, clip_on=False)

        ax_.plot([cell.tvec[-1] - tlim / 4 + t_shift, cell.tvec[-1] + t_shift], 
                 [np.min(dV_ctrl) * 1.5, np.min(dV_ctrl) * 1.5], c='k', clip_on=False)
        ax_.text(cell.tvec[-1] - tlim / 8 + t_shift, np.min(dV_ctrl) * 1.7, 
                 f"{int(np.round(tlim / 4))} ms", 
                 ha="center", va="top", clip_on=False)

        scale_vm = int(np.round(v_amp_ctrl))
        if scale_vm == 0:
            scale_vm = v_amp_ctrl
        ax_.plot([cell.tvec[-1]+ t_shift + 1, cell.tvec[-1]+ t_shift + 1], 
                 [0, scale_vm], c='k', clip_on=False)
        ax_.text(cell.tvec[-1]+ t_shift + tlim*0.01 + 1, scale_vm / 2, 
                 f"{scale_vm}\nmV", 
                ha="left", va="center", clip_on=False)

        ax_.text(tstop - tlim + t_shift, 0, f"{stim_freq} Hz\n{int(amp_e /1000)} µA",
                 ha='right', va='bottom')

    fig.legend([l_ctrl, l_RT], ["extracellular stimulation", "reciprocity-based"],
               frameon=False, ncol=1, loc=[0.25, 0.88])

    lines = []
    line_names = []
    for cell_name in cell_names:
        plot_errors_amp = []
        plot_errors_freq = []
        plot_freqs = []
        plot_amps = []
        soma_vm_stds = []
        is_subthreshold = []
        #amp_e = 1000
        amp_i = cell_input_subthreshold_amp[cell_name]
        markers = []
        marker_sizes = []
        for make_passive in [True, False]:
            for stim_freq in stim_freqs:
                for dist in distances:
                    pos = [dist, 0, 0]
                    for amp_e in amps:

                        # sim_name = f'{cell_name}_passive:{make_passive}_{amp:0.03f}_{stim_freq}Hz'
                        sim_name_e = return_simname(cell_name, make_passive, dist, stim_freq, amp_e)
                        sim_name_i = return_simname(cell_name, make_passive, dist, stim_freq, amp_i)
                        dV_ctrl = np.load(join(results_folder,
                                            f"{sim_name_e}_extracellular.npz"))["dV"]
                        ctrl_soma_vm = np.load(join(results_folder,
                                            f"{sim_name_e}_extracellular.npz"))["Vm"]                
                        dV_RT = np.load(join(results_folder,
                                            f"{sim_name_i}_intracellular.npz"))["dV"]
                        
                        f_, vm_amp, vm_phase = return_amp_and_phase(ctrl_soma_vm, len(ctrl_soma_vm), dt)
                        freq_idx = np.argmin(np.abs(f_ - stim_freq))
                        vm_amp = vm_amp[0, freq_idx]

                        dV_RT *= amp_e /amp_i
                        error = np.std(dV_RT - dV_ctrl)/np.std(dV_ctrl)
                        plot_errors_amp.append(error)
                        soma_vm_stds.append(vm_amp)
                        plot_amps.append(amp_e)

                        is_subthreshold.append(vm_amp <= 5)
                        # if vm_amp <= 5:
                        plot_freqs.append(stim_freq)
                        plot_errors_freq.append(error)
                        markers.append('x' if make_passive else '.')
                        marker_sizes.append(4 if make_passive else 4)


        for idx in range(len(plot_errors_amp)):
            marker = markers[idx]#'o'# if is_subthreshold[idx] else 'x'
            ms = marker_sizes[idx]
            if marker != 'x':
                l, = ax_amp.semilogx(soma_vm_stds[idx], plot_errors_amp[idx],
                                marker, ms=ms, c=cell_clrs[cell_name], clip_on=True)

        lines.append(l)
        line_names.append(cell_legend_names[cell_name])

    mark_subplots(ax_m, "A", ypos=0.98, xpos=0.01)
    mark_subplots(sig_axes[:], "BCD")
    mark_subplots(ax_amp, "E")

    fig.legend(lines, line_names, frameon=False)
    simplify_axes([ax_amp])

    plt.savefig(join(fig_folder, f"RT_validation_quantified_passive:{make_passive}.png"))
    plt.savefig(join(fig_folder, f"RT_validation_quantified_passive:{make_passive}.pdf"))


def analytic_ext_pot(fig_folder):
    """
    Makes plot with analytic solution to ball and stick model
    """
    freqs = np.logspace(0, 4, 1000)
    omega = 2 * np.pi * freqs

    default_param_dict = dict(
        Ra = 100 * 1e-2,  # Ohm m
        d = 1 * 1e-6,  # m
        d_s = 10 * 1e-6,  # m
        Cm = 1. * 1e-6 / 1e-4, # F / m²
        Rm = 15000. * 1e-4, # Ohm m²
        l = 1000 * 1e-6, # m
    )

    param_mod_list = ["Rm", "Ra", "Cm", "l", "d", "d_s"]
    d_far = default_param_dict['l'] * 10
    d_close =  default_param_dict['d_s'] / 2

    dist = np.linspace(5, 10000, 1000)

    dt = 2**-4
    tstop = 1000 - dt
    cell = return_ball_and_stick_cell(tstop, dt,)
    num_tsteps = round(tstop / dt + 1)
    sample_freq = ff.fftfreq(num_tsteps, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs_numeric = sample_freq[pidxs]
    min_freq = 1
    max_freq = 10000
    freqs_idx0 = np.argmin(np.abs(freqs_numeric - min_freq))
    freqs_idx1 = np.argmin(np.abs(freqs_numeric - max_freq - 1))

    stim_freqs = freqs_numeric[freqs_idx0:freqs_idx1]

    cell, syn1, pulse, b1 = make_white_noise_stimuli(cell, 0,
                                                     1, stim_freqs)

    cell.simulate(rec_imem=True, rec_vmem=True)
    elec_params = {
        'sigma': sigma_T,
        'x': np.cos(np.deg2rad(-30)) * dist,
        'y': np.zeros(len(dist)),
        'z': np.sin(np.deg2rad(-30)) * dist,
        'method': 'pointsource',
    }
    electrode = LFPy.RecExtElectrode(cell, **elec_params)
    M = electrode.get_transformation_matrix()
    V_ex = M @ cell.imem * 1000
    t0 = 0
    V_ex = V_ex[:, t0:]
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
    cell.imem = cell.imem[:, t0:]

    f_, ve_amp, ve_phase = return_amp_and_phase(V_ex, len(V_ex[0]), dt)

    f_idx_numeric_0 = np.argmin(np.abs(f_ - min_freq))

    fig = plt.figure(figsize=[6, 2.7])
    fig.subplots_adjust(bottom=0.15, hspace=0.3, right=0.98,
                        top=0.96, left=0.28)

    num_rows = 2
    num_cols = len(param_mod_list)
    scale_clrs = ["blue", "k", "lightblue"]

    ax_I = fig.add_axes([0.075, 0.17, 0.11, 0.37],
                            xscale="log", xlim=[8, 10000],
                            yticks=[1e-3, 1e-2, 1e-1, 1e0],
                            xlabel="distance (µm)", 
                            ylabel=r"$V_{\rm e}$ (µV)")
    
    ax_I.loglog(dist, ve_amp[:, f_idx_numeric_0], c='green', lw=1)
    ax_I.loglog([5, 50], [1.1 / 2 * 1e2, 0.11 / 2 * 1e2],
                c='gray', lw=1.5, ls=':')
    ax_I.loglog([1000, 10000], [0.18e-2 / 2 * 1e2, 0.18e-4 / 2 * 1e2],
                c='gray', lw=1.5, ls=':')
    ax_I.text(20, 0.2 * 1e2, "1/$r$", c='gray')
    ax_I.text(1400, 0.6e-3 * 1e2, "1/$r^2$", c='gray')
    # ax_p.loglog(dist, 1/dist**2)

    for p_idx, param_name in enumerate(param_mod_list):

        ax1 = fig.add_subplot(num_rows, num_cols, p_idx + 1,
                              xlim=[1, 1e4], ylim=[2e-1, 2e2], #yticks=[0.1, 1.0],
                              xscale="log")

        ax3 = fig.add_subplot(num_rows, num_cols, p_idx + 1 + num_cols,
                              xlim=[1, 1e4], ylim=[2e-6, 2e-3],
                              xscale="log")

        if p_idx == 0:
            ax1.set_ylabel(r"$V_{\rm e}^{\rm near}$ (µV)")
            ax3.set_ylabel(r"$V_{\rm e}^{\rm far}$ (µV)")
        else:
            ax1.set_yticklabels([])
            ax3.set_yticklabels([])
        ax1.set_xlabel("Hz")
        ax3.set_xlabel("Hz")

        lines = []
        line_names = []

        results_close = {}
        results_far = {}
        scaling_factors = [2.0, 1.0, 0.5]

        for s_idx, scaling_factor in enumerate(scaling_factors):
            param_dict = default_param_dict.copy()
            param_dict[param_name] = default_param_dict[param_name] * scaling_factor

            tau_m = param_dict['Cm'] * param_dict['Rm']   # m
            lambd = np.sqrt(param_dict['d'] * param_dict['Rm'] / (4 * param_dict['Ra']))
            L = param_dict['l'] / lambd

            W = omega * tau_m
            q = np.sqrt(1 + 1j * W)

            Y = q * param_dict['d_s'] ** 2 / param_dict['d'] / lambd

            T_s_I = - np.sinh(q * L) / (Y * np.cosh(q*L) + np.sinh(q*L))
            T_s_p = lambd / q * (np.cosh(q * L) - 1) / (Y * np.cosh(q * L) + np.sinh(q*L))

            V_e_close = 1e-9 * T_s_I / (4 * np.pi * sigma_T * d_close) * 1e6
            V_e_far = 1e-9 * T_s_p / (4 * np.pi * sigma_T * d_far**2) * 1e6

            results_close[scaling_factor] = V_e_close
            results_far[scaling_factor] = V_e_far

        for s_idx, scaling_factor in enumerate(scaling_factors):
            V_e_close = results_close[scaling_factor]
            V_e_far = results_far[scaling_factor]


            l, = ax1.loglog(freqs, np.abs(V_e_close), lw=1, c=scale_clrs[s_idx])
            ax3.loglog(freqs, np.abs(V_e_far), lw=1, c=scale_clrs[s_idx])

            if scaling_factor == 1.0:
                ax1.loglog([1e3, 1e4], 
                           [np.abs(V_e_close[-1]) * np.sqrt(10), np.abs(V_e_close[-1])],
                           c='gray', lw=1.2, ls=':', zorder=10)
                ax3.loglog([1e3, 1e4], [np.abs(V_e_far[-1]) * 10, np.abs(V_e_far[-1])],
                           c='gray', lw=1.2, ls=':', zorder=10)
                if p_idx == 0:
                    ax1.text(0.3e4, np.abs(V_e_close[-1]) * 2, r"1/$\sqrt{f}$", c='gray')
                    ax3.text(1e4, np.abs(V_e_far[-1]), r"1/$f$", c='gray')

            lines.append(l)
            if scaling_factor == 0.5:
                line_name = f'${param_name}$/2'
            elif scaling_factor == 1.0:
                line_name = f'${param_name}$'
            elif scaling_factor == 2.0:
                line_name = rf'${param_name}$$\cdot$2'
            else:
                raise ValueError(f'')
            line_name = line_name.replace("m", r"_{\rm m}")
            line_name = line_name.replace("a", r"_{\rm a}")
            line_name = line_name.replace("_s", r"_{\rm s}")

            line_names.append(line_name)

        plt.rcParams.update({"text.usetex": True})
        ax1.legend(lines, line_names, frameon=False, ncol=1, loc=(0.0, -0.03),
                   labelspacing=0.2, handlelength=1)
        plt.rcParams.update({"text.usetex": False})

        mark_subplots(ax1, "BCDEFG"[p_idx], ypos=1.05, xpos=-0.05)

    simplify_axes(fig.axes)

    for ax in fig.axes:
        ax.set_xticks([1, 100, 10000])

    for ax in fig.axes[3:]:
        ax.set_yticklabels([])

    ax_I.set_yticks([1e-2, 1e0, 1e2])
    ax_I.set_xlim([3, 10000])
    ax_I.set_xticks([10, 100, 1000, 10000])

    ax_m = fig.add_axes([0.03, 0.62, 0.17, 0.35], aspect=1, xticks=[], yticks=[],
                        frameon=False, xlim=[-300, 700], ylim=[-400, 1100])

    ax_m.plot(cell.x.T, cell.z.T, c='k')
    ax_m.plot(cell.x[0].mean(), cell.z[0].mean(), 'o', c='k', ms=5)
    ax_m.plot(electrode.x, electrode.z, ls='--', lw=0.5, c='green')

    ax_m.plot(cell.x[0].mean(), cell.z[0].mean(), 'o',
                        c='orange', ms=3, mec='k', mew=0.5)

    mark_subplots(ax_m, "A", ypos=1)

    fig.savefig(join(fig_folder, f"analytic_ext_pot_combos.png"))
    fig.savefig(join(fig_folder, f"analytic_ext_pot_combos.pdf"))


def neural_elements_fig(results_folder, fig_folder):

    case_names = ["PC", "IN", "axon terminal", "passing axon"]
    input_idxs = [0, 0, 0, "midpoint"]
    cell_names = ["hay", "allen_491623973", "myelinated_axon", "myelinated_axon"]

    case_clrs = {
        case_name: plt.cm.tab10(n / 3)
            for n, case_name in enumerate(case_names)}

    plot_freqs = np.array([1, 100, 1000, 10000])
    plot_dists = np.array([10, 1000, 10000])
    make_passive = True
    do_neural_simulations = True
    if do_neural_simulations:
        white_noise_neural_sims(1.0, make_passive, case_names,
                                cell_names, input_idxs, plot_dists, plot_freqs,
                                results_folder, fig_folder)

    fig_title = f"neural_elements_passive:{make_passive}"

    num_cols = len(cell_names)
    plt.close("all")
    fig = plt.figure(figsize=[6, 3.2])
    fig.subplots_adjust(top=0.91, bottom=0.01, right=0.57, left=0.01,
                        hspace=0.12, wspace=0.1)
    cax = fig.add_axes([0.572, 0.15, 0.008, 0.7],
                       frameon=False, zorder=1000)

    ax_dist = fig.add_axes([0.70, 0.57, 0.27, 0.32], xlabel="distance (µm)",
                            ylabel="amplitude (µV)",
                            xticks=[10, 100, 1000, 1000],
                            #xlim=[10, 10000],
                            ylim=[1e-2, 0.5e2])

    ax_freq = fig.add_axes([0.70, 0.1, 0.27, 0.32], xlabel="frequency (Hz)",
                            ylabel="amplitude (norm.)",
                            #xticks=[10, 100, 1000],
                            #xlim=[1, 1000],
                            ylim=[2e-2, 1.5e0])

    ax_dist.loglog([0.7e1, 0.7e2], [5e1, 5e0], c='gray', lw=1, ls=':', zorder=1000)
    ax_dist.text(70, 0.8e1, "1/$r$", va="center", ha="left", c='gray')
    ax_dist.loglog([2.9e2, 2.9e3], [0.4e0, 0.4e-2], c='gray', lw=1, ls=':', zorder=1000)
    ax_dist.text(1.2e3, 0.3e-1, "1/$r$²", va="center", ha="left", c='gray')
    
    lines = []
    line_names = []

    scale_max = 50
    for c_idx, case_name in enumerate(case_names):
        sim_name = f'white_noise_{case_names[c_idx]}_passive:{make_passive}'
        data_dict = np.load(join(results_folder, f"{sim_name}.npz"), 
                            allow_pickle=True)
        print("SIM NAME: ", data_dict["sim_name"])
        print("CELL NAME: ", data_dict["cell_name"])

        grid_elec_params = data_dict["grid_elec_params"][()].copy()
        grid_shape = data_dict["grid_LFPs_freqs"][0].shape
        lateral_elec_params = data_dict["lateral_elec_params"][()].copy()
        wm_stim_amp = data_dict["wn_stim_amp"]
        
        scaling_factor = 1 / wm_stim_amp

        grid_x = grid_elec_params['x'].reshape(grid_shape)
        grid_z = grid_elec_params['z'].reshape(grid_shape)
        cell_x = data_dict["cell.x"]
        cell_z = data_dict["cell.z"]

        lateral_LFP_dists = data_dict["lateral_LFPs_dists"] * scaling_factor * 1000
        lateral_LFP_freqs = data_dict["lateral_LFPs_freqs"] * scaling_factor * 1000
        grid_LFP_freqs = data_dict["grid_LFPs_freqs"]  * scaling_factor * 1000
        lateral_dists = data_dict["dists"]
        freqs = data_dict["f_ve"]

        print("MAX: ", np.max(np.abs(grid_LFP_freqs)))

        plot_freqs = [1, 1000]
        plot_dists = [10, 10000]

        for idx, freq in enumerate(plot_freqs):
            pf_idx = np.argmin(np.abs(data_dict["plot_freqs"] - freq))
            if not data_dict["plot_freqs"][pf_idx] == freq:
                raise RuntimeError("Something wrong with freq!")
            if idx == 0:
                ax_title = f'{case_names[c_idx]}\n{freq} Hz'
            else: 
                ax_title = f'{freq} Hz'

            stim_idx = data_dict["input_idx"]#input_idxs[c_idx]

            grid_LFP = grid_LFP_freqs[pf_idx]
            lateral_LFP = lateral_LFP_freqs[pf_idx]

            if idx == 0:
                ls = '-'
            elif idx == 1:
                ls = '--'
            elif idx == 2:
                ls = '--'
            else:
                raise RuntimeError('')

            num = 16
            levels = np.logspace(-2.5, 0, num=num)
            
            levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
            colors_from_map = [cmap_v_e(i / (len(levels_norm) - 2))
                                for i in range(len(levels_norm) - 1)]
            colors_from_map[num - 1] = (1.0, 1.0, 1.0, 0.0)

            title = f'{freq} Hz'
            ax1 = fig.add_subplot(2, num_cols, c_idx + num_cols * idx + 1,
                                    aspect=1,
                                    ylim=[np.min(grid_z), np.max(grid_z)],
                                    xlim=[np.min(grid_x), np.max(grid_x)],
                                    frameon=False, xticks=[], yticks=[])
            ax1.set_title(ax_title, pad=-5)

            if idx == 0:
                mark_subplots(ax1, "ABCD"[c_idx], xpos=0.06, ypos=1.03)
                lines.append(Line2D([0], [0], color=case_clrs[case_name], lw=4))
                line_names.append(f'{case_names[c_idx]}')

            l, = ax_dist.loglog(lateral_dists, np.abs(lateral_LFP), ls=ls,
                                    c=case_clrs[case_name], zorder=-c_idx)

            if (c_idx == 0) and (idx == 0):
                ax1.plot([250, 250], [300, 500], c='k', lw=0.7,
                         clip_on=False)
                ax1.text(260, 400, "200\nµm", va='center', ha='left')

            ax1.plot(cell_x.T, cell_z.T, c='k', lw=0.1,
                     solid_capstyle="butt", zorder=-1, alpha=0.25)

            ax1.plot(cell_x[stim_idx].mean(), cell_z[stim_idx].mean(), 'o',
                        c='orange', ms=3, mec='k', mew=0.5)

            ax1.contour(grid_x, grid_z, grid_LFP, colors=colors_from_map,
                        linewidths=(0.1), zorder=-2,
                            levels=levels_norm, alpha=1)
            ep_intervals = ax1.contourf(grid_x, grid_z, grid_LFP,
                                            zorder=-2, colors=colors_from_map,
                                            levels=levels_norm, extend='both',
                                        alpha=1)

            ax1.plot(lateral_elec_params["x"], lateral_elec_params["z"],
                        c='k', ls=':', lw=0.3)


        for idx, dist in enumerate([10, 10000]):
            pd_idx = np.argmin(np.abs(data_dict["plot_dists"] - dist))
            lateral_LFP = lateral_LFP_dists[pd_idx] * scaling_factor

            if idx == 0:
                ls = '-'
            elif idx == 1:
                ls = '--'
            elif idx == 2:
                ls = '--'
            l, = ax_freq.loglog(freqs, lateral_LFP / np.max(lateral_LFP), ls=ls,
                                    c=case_clrs[case_name], zorder=-c_idx)

    sim_name = f'white_noise_{case_names[0]}_passive:{make_passive}'
    data_dict = np.load(join(results_folder, f"{sim_name}.npz"),
                        allow_pickle=True)
    print("SIM NAME: ", data_dict["sim_name"])
    print("CELL NAME: ", data_dict["cell_name"])

    lateral_elec_params = data_dict["lateral_elec_params"][()].copy()

    lateral_dists = data_dict["dists"]
    wm_stim_amp = data_dict["wn_stim_amp"]

    dist_1_lat = np.sqrt((lateral_elec_params['x'] - 0)**2 +
                    (lateral_elec_params['y'] - 0)**2 +
                    (lateral_elec_params['z'] - 0)**2)

    monopole_ecp_lateral = -wm_stim_amp / (4 * np.pi * lateral_elec_params['sigma'] * dist_1_lat) * 1000

    l, = ax_dist.loglog(lateral_dists, np.abs(monopole_ecp_lateral), ls='-',
                        c='gray', zorder=-c_idx)
    ax_freq.loglog(freqs, np.ones(len(freqs)), ls='-',
                        c='gray', zorder=-1000)

    cbar = fig.colorbar(ep_intervals, cax=cax)

    cbar.set_label('µV', labelpad=-15)
    cbar_ticks = np.array([-1, -0.1, -0.01,  0.01, 0.1, 1]) * scale_max
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([int(c_) if np.round(c_) == c_ else c_ for c_ in cbar_ticks])

    ax_dist.legend([Line2D([0], [0], color='k', lw=1.5), 
                    Line2D([0], [0], color='k', lw=1.5, ls='--')],
                    [f"{f_} Hz" for f_ in plot_freqs], frameon=False)
    ax_freq.legend([Line2D([0], [0], color='k', lw=1.5), 
                    Line2D([0], [0], color='k', lw=1.5, ls='--')],
                    [f"{d_} µm" for d_ in plot_dists], frameon=False)

    first_legend = fig.legend([Line2D([0], [0], color='gray', lw=4)],
                              [r"PS"], frameon=False,
                              loc=(0.64, 0.945))
    fig.add_artist(first_legend)  # Add the first legend manually

    fig.legend(lines, line_names,
                   frameon=False, loc=(0.73, 0.9), ncol=2)

    simplify_axes([ax_dist, ax_freq])
    mark_subplots([ax_dist, ax_freq], "EF",
                  ypos=1.02, xpos=-0.17)

    ax_dist.set_yticks([1e-2, 1e-1, 1e0, 1e1])
    ax_dist.set_xticks([10, 100, 1000])
    ax_freq.set_xticks([1, 10, 100, 1000, 10000])
    ax_dist.set_xticklabels(["10", "100", "1000"])
    ax_freq.set_xticklabels(["1", "10", "100", "1000", "10000"])
    ax_dist.set_xlim(8, 6000)
    ax_freq.set_xlim(1, 10000)
    plt.savefig(join(fig_folder, f"{fig_title}.pdf"))


def white_noise_neural_sims(amp_i, make_passive, case_names,
                            cell_names, input_idxs, plot_dists, plot_freqs, 
                            results_folder, fig_folder):

    cell_clrs = {
        cell_name: plt.cm.tab20c(0.1 + n / 3)
            for n, cell_name in enumerate(cell_names)}
    dt = 2 ** -7
    tstop = 1000. - dt
    if make_passive:
        cutoff = 100.
    else:
        cutoff = 6000
    num_tsteps = round((tstop) / dt + 1)
    sample_freq = ff.fftfreq(num_tsteps, d=dt / 1000)
    pidxs = np.where(sample_freq >= 0)
    freqs = sample_freq[pidxs]
    freqs_idx0 = np.argmin(np.abs(freqs - plot_freqs[0]))
    freqs_idx1 = np.argmin(np.abs(freqs - plot_freqs[-1] - 100))

    stim_freqs = freqs[freqs_idx0:freqs_idx1]
    # print(stim_freqs)
    for c_idx, cell_name in enumerate(cell_names):
        c_clr = cell_clrs[cell_name]
        if amp_i is None:
            wn_stim_amp = cell_input_subthreshold_amp[cell_name] / 100
        else: 
            wn_stim_amp = amp_i

        pid = os.fork()
        if pid == 0:
            sim_name = f'white_noise_{case_names[c_idx]}_passive:{make_passive}'
            # delete old sections from NEURON namespace
            neuron.h("forall delete_section()")

            cell = return_cell_model(tstop, dt, cutoff, make_passive, cell_name)

            if input_idxs[c_idx] == "midpoint":
                try:
                    stim_idx = cell.get_closest_idx(z=np.max(cell.z) / 2, section="node")
                except ValueError:
                    stim_idx = cell.get_closest_idx(z=np.max(cell.z) / 2)
                print(cell.z[stim_idx].mean(), np.max(cell.z))
            else:
                stim_idx = input_idxs[c_idx]
            z_offset = 0 if stim_idx == 0 else 500
            cell.set_pos(z=-cell.z[stim_idx].mean() + z_offset)

            # # Create a grid of measurement locations, in (um)
            grid_x, grid_z = np.mgrid[-457:451:20, -557:1300:20]
            grid_y = np.ones(grid_x.shape) * 0
            grid_shape = grid_x.shape

            grid_elec_params = {
                'sigma': sigma_T,  # extracellular conductivity
                'x': grid_x.flatten(),  # electrode positions
                'y': grid_y.flatten(),
                'z': grid_z.flatten(),
                'method': 'linesource'
            }

            dists = np.logspace(np.log10(cell.d[stim_idx]/2), 4, 1000)
            lateral_elec_params = {
                'sigma': sigma_T,  # extracellular conductivity
                'x': np.cos(np.deg2rad(-30)) * dists,  # electrode positions
                'y': np.zeros(len(dists)),
                'z': np.sin(np.deg2rad(-30)) * dists + cell.z[stim_idx].mean(),
                'method': 'pointsource'
            }

            cell, syn1, pulse, b1 = make_white_noise_stimuli(cell, stim_idx,
                                                            -wn_stim_amp, stim_freqs)

            cell.simulate(rec_vmem=True, rec_imem=True)

            cutoff_idx = np.argmin(np.abs(cell.tvec - cutoff))

            cell.tvec = cell.tvec[cutoff_idx:] - cell.tvec[cutoff_idx]
            pulse = pulse[cutoff_idx:]
            cell.imem = cell.imem[:, cutoff_idx:]
            cell.imem = cell.imem - cell.imem.mean(axis=1)[:, None]

            cell.vmem = cell.vmem[:, cutoff_idx:]
            print("Vm std: ", np.std(cell.vmem[stim_idx, :]))

            lateral_electrode = LFPy.RecExtElectrode(cell, **lateral_elec_params)
            grid_electrode = LFPy.RecExtElectrode(cell, **grid_elec_params)

            # lateral_LFP =  lateral_electrode.get_transformation_matrix() @ cell.imem
            # lateral_LFP =  1000 * lateral_LFP

            f_pulse, stim_amp, stim_phase = return_amp_and_phase(-pulse, len(pulse), dt)
            f_im, im_amp, im_phase = return_amp_and_phase(cell.imem, cell.imem.shape[1], dt)
            
            grid_LFPs_freqs = []
            lateral_LFPs_freqs = []
            im_phase -= stim_phase

            for freq in plot_freqs:
                f_idx = np.argmin(np.abs(f_im - freq))

                imem_freq1 = np.real(im_amp[:, f_idx] * np.exp(
                    1j * np.deg2rad(im_phase[:, f_idx])))

                grid_LFP =  grid_electrode.get_transformation_matrix() @ imem_freq1
                grid_LFP =  grid_LFP.reshape(grid_shape)
                grid_LFPs_freqs.append(grid_LFP)

                lateral_LFP =  lateral_electrode.get_transformation_matrix() @ imem_freq1
                lateral_LFP =  lateral_LFP
                lateral_LFPs_freqs.append(lateral_LFP)

            lateral_LFPs_dists = []
            for plot_dist in plot_dists:
                d_idx = np.argmin(np.abs(dists - plot_dist))
                lateral_elec_params_ = lateral_elec_params.copy()
                lateral_elec_params_['x'] = lateral_elec_params['x'][d_idx]
                lateral_elec_params_['y'] = lateral_elec_params['y'][d_idx]
                lateral_elec_params_['z'] = lateral_elec_params['z'][d_idx]
                lateral_electrode_ = LFPy.RecExtElectrode(cell, **lateral_elec_params_)
                lateral_LFP_ =  lateral_electrode_.get_transformation_matrix() @ cell.imem
                f_ve, ve_amp, ve_phase = return_amp_and_phase(lateral_LFP_, 
                                                              cell.imem.shape[1], 
                                                              dt)
                f_idxs = (f_ve >= plot_freqs[0]) & (f_ve <= plot_freqs[-1])
                f_ve = f_ve[f_idxs]
                ve_amp = ve_amp[0, f_idxs]
                lateral_LFPs_dists.append(ve_amp)

            if len(f_pulse) != len(f_im):
                raise RuntimeError("Frequency arrays should be of the same length")

            data_dict = {
                'sim_name': sim_name,
                'cell.x': cell.x,
                'cell.y': cell.y,
                'cell.z': cell.z,
                'case_name': case_names[c_idx],
                'input_type': 'white_noise',
                'cell_name': cell_name,
                'input_idx': stim_idx,
                'imem': cell.imem,
                'vmem': cell.vmem,
                'freqs_imem': f_im,
                'imem_amps': im_amp,
                'imem_phases': im_phase,
                'pulse': pulse,
                'tvec': cell.tvec,
                'stim_freqs': stim_freqs,
                'wn_stim_amp': wn_stim_amp,
                'plot_dists': plot_dists,
                'lateral_LFPs_dists': lateral_LFPs_dists,
                'f_ve': f_ve,
                'dists': dists,
                'lateral_elec_params': lateral_elec_params,
                'grid_elec_params': grid_elec_params,
                'plot_freqs': plot_freqs,
                'grid_LFPs_freqs': grid_LFPs_freqs,
                'lateral_LFPs_freqs': lateral_LFPs_freqs,
                # 'lateral_LFP': lateral_LFP,
                # 'lateral_elec_params': lateral_elec_params,
                # 'distances': distances
                }
            np.savez(os.path.join(results_folder, f"{sim_name}.npz"), **data_dict)

            plt.close("all")
            fig = plt.figure(figsize=[10, 5])
            plt.subplots_adjust(wspace=0.7, hspace=0.4, left=0.01, right=0.99,
                                bottom=0.07, top=0.95)

            ax6 = fig.add_subplot(247, xlabel="distance (µm)",
                        ylabel="$V_e$ (mV)")
            cax = fig.add_axes([0.15, 0.25, 0.005, 0.5],
                    frameon=False, zorder=1000)

            num = 11
            levels = np.logspace(-2.3, 0, num=num)
            scale_max = np.max(np.abs(grid_LFPs_freqs))

            levels_norm = scale_max * np.concatenate((-levels[::-1], levels))
            colors_from_map = [cmap_v_e(i / (len(levels_norm) - 2))
                            for i in range(len(levels_norm) - 1)]
            colors_from_map[num - 1] = (1.0, 1.0, 1.0, 0.0)

            num_cols = 4

            lines = []
            line_names = []
            line_styles = ['-','--', ':', '-.']
            for f_idx, freq in enumerate(plot_freqs):
                ax_m = fig.add_subplot(len(plot_freqs), num_cols, 
                                       num_cols * f_idx + 1, aspect=1,
                            xticks=[], yticks=[], frameon=False, 
                            xlim=[-450, 450], ylim=[-550, 1250])

                ax_m.plot(cell.x.T, cell.z.T, 'k', lw=.5)
                plt.plot(lateral_electrode.x, lateral_electrode.z, ls=':', c='gray')

                ax_m.contour(grid_x, grid_z, grid_LFPs_freqs[f_idx], colors=colors_from_map,
                        linewidths=(0.1), zorder=-2,
                            levels=levels_norm, alpha=1)
                ep_intervals = ax_m.contourf(grid_x, grid_z, grid_LFPs_freqs[f_idx],
                                            zorder=-2, colors=colors_from_map,
                                            levels=levels_norm, extend='both',
                                            alpha=1)
                l_, = ax6.loglog(dists, np.abs(lateral_LFPs_freqs[f_idx]), c=c_clr, 
                                 ls=line_styles[f_idx])
                lines.append(l_)
                line_names.append(f"{freq} Hz")

            cbar = fig.colorbar(ep_intervals, cax=cax)
            cbar.set_label('mV', labelpad=0)
            cbar.set_ticks(np.array([-1, -0.1, -0.01,  0.01, 0.1, 1]) * scale_max)
            ax6.legend(lines, line_names, frameon=False)
            ax2 = fig.add_subplot(2, 4, 2, ylabel="$I_{0}$ (nA)", xlabel="time (ms)")

            ax3 = fig.add_subplot(243, xlabel="time (ms)",
                        ylabel="$V_m$ at\nstim site (mV)")
            ax4 = fig.add_subplot(244, xlabel="time (ms)",
                        ylabel="$I_m$ at\nstim site (nA)")

            ax5 = fig.add_subplot(246, xlabel="frequency (Hz)",
                         ylabel="$I_0$ (nA)",
                         ylim=[wn_stim_amp * 1e-1, wn_stim_amp * 1e1])
            ax8 = fig.add_subplot(248, xlabel="frequency (Hz)",
                         ylabel="$V_e$ (norm.)")

            ax2.plot(cell.tvec, pulse, lw=1, c='k')
            ax3.plot(cell.tvec, cell.vmem[stim_idx], lw=1, c='k')
            ax4.plot(cell.tvec, cell.imem[stim_idx], lw=1, c='k')

            ax5.loglog(f_pulse[freqs_idx0:freqs_idx1],
                           stim_amp[0,freqs_idx0:freqs_idx1], lw=1, c=c_clr)

            lines = []
            line_names = []
            line_styles = ['-', '--', ':']
            for d_idx, plot_dist in enumerate(plot_dists):
                l_, = ax8.loglog(f_ve, 
                                 lateral_LFPs_dists[d_idx] / lateral_LFPs_dists[d_idx][0],
                                 c=c_clr, ls=line_styles[d_idx])
                lines.append(l_)
                line_names.append(f"{plot_dist} µm")
            ax8.legend(lines, line_names, frameon=False)
            # l1, = ax6.loglog(f_pulse[freqs_idx0:freqs_idx1], 
                    #    ve_amp[0, freqs_idx0:freqs_idx1] / ve_amp[0, freqs_idx0], 
                        # lw=1, c='k')
            # l2, = ax6.loglog(f_[freqs_idx0:freqs_idx1], 
                        # ve_amp[-1, freqs_idx0:freqs_idx1] / ve_amp[-1, freqs_idx0], 
                        # lw=1, c='r')
                
            # plt.legend([l1, l2], [f"{dists[0]} µm", f"{dists[-1]} µm"])
            plt.savefig(os.path.join(fig_folder, f"control_{sim_name}.png"))

            cell  = None
            syn1 = None
            #del cell, lateral_electrode, syn1, pulse, b1
            os._exit(0)
            
        else:
            os.waitpid(pid, 0)


def return_cell_model(tstop, dt, cutoff, make_passive, cell_name,
                      my_length=None, axon_diameter_scaling=None):
    neuron.h("forall delete_section()")

    if cell_name == "hay":
        cell = return_hay_cell(tstop + cutoff, dt,
                               make_passive=make_passive)
    elif cell_name == "stick":
        cell = return_stick_cell(tstop + cutoff, dt,
                                 make_passive=make_passive)
    elif cell_name == "ball_and_stick":
        cell = return_ball_and_stick_cell(tstop + cutoff, dt, )
    elif cell_name == "2comp":
        cell = return_2comp_cell(tstop + cutoff, dt, make_passive)
    elif cell_name == "axon":
        cell = return_axon_cell(tstop + cutoff, dt,
                                make_passive=make_passive)
    elif cell_name == "myelinated_axon":
        if my_length is None:
            my_length = 50
        if axon_diameter_scaling is None:
            axon_diameter_scaling = 1
        cell = ha.return_constructed_myelinated_axon(tstop + cutoff, dt,
                                                     axon_diam_scaling=axon_diameter_scaling,
                                                     my_length=my_length, axon_length=10000,
                                                     make_passive=make_passive)
    elif cell_name == "unmyelinated_axon":
        cell = ha.return_constructed_unmyelinated_axon(tstop + cutoff, dt,
                                                       axon_diam_scaling=1.0,
                                                       axon_length=1000,
                                                       make_passive=make_passive)
    elif cell_name == "MRG_axon":
        cell = return_MRG_axon_cell(tstop + cutoff, dt,
                                make_passive=make_passive)
    elif "allen" in cell_name:
        cell_id = cell_name.split("_")[1]
        cell = return_allen_cell_model(cell_id, dt, tstop + cutoff,
                                       make_passive=make_passive)
    elif cell_name.startswith("L"):
        cell = return_BBP_neuron(cell_name, tstop + cutoff, dt,
                                       make_passive=make_passive)
    else:
        raise ValueError(f"Did not recognize cell name: {cell_name}")
    #cell = remove_active_mechanisms(["ih"], cell)
    return cell


def rotation_matrix(d):
    """
    Calculates a rotation matrix given a vector d. The direction of d
    corresponds to the rotation axis. The length of d corresponds to
    the sin of the angle of rotation.

    Variant of: http://mail.scipy.org/pipermail/numpy-discussion/2009-March/040806.html
    """
    sin_angle = np.linalg.norm(d)

    if sin_angle == 0:
        return np.identity(3)

    d /= sin_angle

    eye = np.eye(3)
    ddt = np.outer(d, d)
    skew = np.array([[    0,  d[2],  -d[1]],
                  [-d[2],     0,  d[0]],
                  [d[1], -d[0],    0]], dtype=np.float64)

    M = ddt + np.sqrt(1 - sin_angle**2) * (eye - ddt) + sin_angle * skew
    return M


def pathpatch_2d_to_3d(pathpatch, z=0, normal='z'):
    """
    Transforms a 2D Patch to a 3D patch using the given normal vector.

    The patch is projected into they XY plane, rotated about the origin
    and finally translated by z.
    """
    if type(normal) is str: #Translate strings to normal vectors
        index = "xyz".index(normal)
        normal = np.roll((1.0,0,0), index)

    normal /= np.linalg.norm(normal) #Make sure the vector is normalised

    path = pathpatch.get_path() #Get the path and the associated transform
    trans = pathpatch.get_patch_transform()

    path = trans.transform_path(path) #Apply the transform

    pathpatch.__class__ = art3d.PathPatch3D #Change the class
    pathpatch._code3d = path.codes #Copy the codes
    pathpatch._facecolor3d = pathpatch.get_facecolor #Get the face color

    verts = path.vertices #Get the vertices in 2D

    d = np.cross(normal, (0, 0, 1)) #Obtain the rotation vector
    M = rotation_matrix(d) #Get the rotation matrix

    pathpatch._segment3d = np.array([np.dot(M, (x, y, 0)) + (0, 0, z) for x, y in verts])


def pathpatch_translate(pathpatch, delta):
    """
    Translates the 3D pathpatch by the amount delta.
    """
    pathpatch._segment3d += delta



def run_TES_sim_detailed_head_model(cell_name, amp_TES, freq, local_E_field,
                                    tstop, dt, cutoff, make_passive, save_loc):

    # Assuming a uniform electric field we can find extracellular potential
    local_ext_pot = np.vectorize(lambda x, y, z: local_E_field * z / 1000)
    cell = return_cell_model(tstop, dt, cutoff, make_passive, cell_name)
    n_tsteps_ = int((tstop + cutoff) / dt + 1)
    t_ = np.arange(n_tsteps_) * dt

    # Lead field from NY HEAD is from a 1 mA input
    # Our amp is in nA, and must therefore be scaled to mA
    pulse = amp_TES * 1e-6 * np.sin(2 * np.pi * freq * t_ / 1000)

    v_cell_ext = np.zeros((cell.totnsegs, n_tsteps_))
    v_cell_ext[:, :] = local_ext_pot(cell.x.mean(axis=-1),
                                 cell.y.mean(axis=-1),
                                 cell.z.mean(axis=-1)
                                 ).reshape(cell.totnsegs, 1) * pulse.reshape(1, n_tsteps_)
    cell.insert_v_ext(v_cell_ext, t_)
    cell.simulate(rec_imem=True, rec_vmem=True)
    t0 = np.argmin(np.abs(cell.tvec - cutoff))
    pulse = pulse[t0:]
    cell.imem = cell.imem[:, t0:]
    cell.vmem = cell.vmem[:, t0:]
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
    dV_TES = 1000 * (cell.vmem[0] - np.mean(cell.vmem[0]))
    np.save(save_loc, dV_TES)


def detailed_head_model_neuron_fig(results_folder, fig_folder):

    make_passive = True

    target_loc = "soma"
    # target_loc = "apic"
    # stim_idx = 0
    amp_i = 1  # nA
    amp_TES = 1e6  # nA

    cell_names = ["hay", "allen_491623973", "myelinated_axon", "ball_and_stick"]

    cell_legend_names = {'hay': 'rat PC',
                         "allen_626170736": "human PC",
                         "allen_626170610": "human PC",
                         "allen_491623973": 'mouse IN',
                         "myelinated_axon": "axon terminal",
                         "unmyelinated_axon": "unmyelinated axon",
                         "MRG_axon": "axon terminal",
                         "ball_and_stick": "ball and stick",
                         }
    cell_clrs = {
        cell_name: plt.cm.tab10(n / 3)
        for n, cell_name in enumerate(cell_names)}

    freqs = np.logspace(0, 4, 21)
    plot_freq = freqs[np.argmin(np.abs(freqs - 10))]
    eeg_elec_idx = 140

    nyhead = NYHeadModel()
    cortex = nyhead.cortex # Locations of every vertex in cortex
    elecs = np.array(nyhead.elecs) # 3D locations of electrodes
    elec_loc = elecs[:3, eeg_elec_idx]
    vertex_idx = np.argmax(np.abs(nyhead.lead_field_normal[:, eeg_elec_idx]))
    dipole_loc = nyhead.cortex[:, vertex_idx]
    nyhead.set_dipole_pos(dipole_loc)

    head_tri = np.asarray(nyhead.head_data["head"]["tri"]).T - 1 # For 3D plotting

    head_vc = np.asarray(nyhead.head_data["head"]["vc"])
    cortex_tri = np.asarray(nyhead.head_data["cortex75K"]["tri"]).T - 1 # For 3D plotting
    x_ctx, y_ctx, z_ctx = cortex
    x_h, y_h, z_h = head_vc[0, :], head_vc[1, :], head_vc[2, :]

    # For a 1 mA TES, the lead field is in V/m
    lead_field = nyhead.lead_field_normal
    local_E_field = lead_field[vertex_idx, eeg_elec_idx]

    def return_simname(cell_name, target_loc, eeg_elec_idx, make_passive, freq):
        return f"sim_cdm_{cell_name}_{target_loc}_{eeg_elec_idx}_passive:{make_passive}_{freq}"

    if make_passive:
        cutoff = 100
    else:
        cutoff = 2000

    target_num_tsteps = 1000
    def return_dt(tstop):
        dt_ = tstop / target_num_tsteps
        dts = 1/2**np.arange(3, 10)
        return dts[np.argmin(np.abs(dt_ - dts))]

    run_neural_sim = True
    run_TES_sim = True
    if run_neural_sim:
        for cell_name in cell_names:
            for freq in freqs:
                pid = os.fork()
                if pid == 0:
                    tstop = 50 * 100 / freq
                    dt = return_dt(tstop)
                    tstop -= dt
                    cell = return_cell_model(tstop, dt, cutoff, make_passive, cell_name)
                    print(cell_name, freq, dt)
                    if "allen" in cell_name:
                        cell.set_rotation(y=np.pi)

                    if target_loc == "soma":
                        stim_idx = 0
                    elif target_loc == "apic":
                        stim_idx = cell.get_closest_idx(z=1200)

                    cell, syn1, pulse, b1 = make_sinusoidal_stimuli(cell, stim_idx,
                                                                    -amp_i, freq)

                    cell.simulate(rec_imem=True)

                    t0 = np.argmin(np.abs(cell.tvec - cutoff))
                    pulse = pulse[t0:]
                    cell.imem = cell.imem[:, t0:]
                    #cell.somav = cell.somav[t0:]
                    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]
                    cdm = CurrentDipoleMoment(cell).get_transformation_matrix() @ cell.imem
                    cdm[0, :] -= np.mean(cdm[0, :])
                    cdm[1, :] -= np.mean(cdm[1, :])
                    cdm[2, :] -= np.mean(cdm[2, :])

                    sim_name = return_simname(cell_name, target_loc,
                                              eeg_elec_idx, make_passive, freq)
                    np.save(join(results_folder, f"{sim_name}.npy"), cdm)

                    plt.close("all")
                    plt.plot(cell.tvec, cdm[0, :])
                    plt.plot(cell.tvec, cdm[1, :])
                    plt.plot(cell.tvec, cdm[2, :])
                    plt.axhline(np.max(np.abs(cdm)))
                    plt.axhline(-np.max(np.abs(cdm)))
                    plt.savefig(join(fig_folder, f"control_{sim_name}.png"))
                    os._exit(0)
                else:
                    os.waitpid(pid, 0)

    cell_name_TES = "hay"
    sim_name_TES = return_simname(cell_name_TES, target_loc,
                              eeg_elec_idx, make_passive, plot_freq)
    save_loc_TES = join(results_folder, f"TES_{sim_name_TES}.npy")
    if run_TES_sim:

        pid = os.fork()
        if pid == 0:
            tstop = 50 * 100 / plot_freq
            dt = return_dt(tstop)
            tstop -= dt
            run_TES_sim_detailed_head_model(cell_name_TES, amp_TES, plot_freq,
                                            local_E_field, tstop, dt, cutoff,
                                            make_passive, save_loc_TES)
            os._exit(0)
        else:
            os.waitpid(pid, 0)

    dV_TES = np.load(save_loc_TES)

    cdm_amps = {cell_name: np.zeros(len(freqs)) for cell_name in cell_names}
    for cell_name in cell_names:
        for f_idx, freq in enumerate(freqs):
            sim_name = return_simname(cell_name, target_loc,
                                      eeg_elec_idx, make_passive, freq)
            cdm_ = np.load(join(results_folder, f"{sim_name}.npy"))
            cdm_norm = np.linalg.norm(cdm_, axis=0)

            cdm_amps[cell_name][f_idx] = np.max(np.abs(cdm_norm))#np.max(np.abs(cdm_[2, :]))

            if (f_idx == 0) or (f_idx == len(freqs) - 1):
                print(cell_name, freq, cdm_amps[cell_name][f_idx])

    if np.round(plot_freq) == plot_freq:
        plot_freq = int(plot_freq)

    freq = plot_freq
    tstop = 50 * 100 / plot_freq
    dt = return_dt(tstop)
    tstop -= dt
    cell_name = cell_names[0]

    sim_name = return_simname(cell_name, target_loc,
                              eeg_elec_idx, make_passive, float(freq))

    cell = return_cell_model(tstop, dt, cutoff, make_passive, cell_name)

    if "allen" in cell_name:
        cell.set_rotation(y=np.pi)
    if target_loc == "soma":
        stim_idx = 0
    elif target_loc == "apic":
        stim_idx = cell.get_closest_idx(z=1200)

    cell, syn1, pulse, b1 = make_sinusoidal_stimuli(cell, stim_idx,
                                                    -amp_i, plot_freq)
    cell.simulate(rec_imem=True)

    t0 = np.argmin(np.abs(cell.tvec - cutoff))
    pulse = pulse[t0:]

    pulse = - pulse
    f_, pulse_amp, pulse_phase = return_amp_and_phase(pulse, len(pulse), dt)
    f_idx = np.argmin(np.abs(plot_freq - f_))

    cell.imem = cell.imem[:, t0:]
    # cell.somav = cell.somav[t0:]
    cell.tvec = cell.tvec[t0:] - cell.tvec[t0]

    cdm = CurrentDipoleMoment(cell).get_transformation_matrix() @ cell.imem
    cdm[0, :] -= np.mean(cdm[0, :])
    cdm[1, :] -= np.mean(cdm[1, :])
    cdm[2, :] -= np.mean(cdm[2, :])
    cdm_loaded = np.load(join(results_folder, f"{sim_name}.npy"))

    if not np.sum(np.abs(cdm - cdm_loaded)) == 0:
        raise RuntimeError

    f_, cdm_amp, cdm_phase = return_amp_and_phase(cdm[2, :], len(cdm[2, :]), dt)

    f_idx = np.argmin(np.abs(freq - f_))

    stim_amp_2_cdm_factor = cdm_amp[0, f_idx] / amp_i
    stim_phase_2_cdm_factor = np.deg2rad(cdm_phase[0, f_idx] - pulse_phase[0, f_idx])

    print("CDM: ", stim_amp_2_cdm_factor)

    p = nyhead.rotate_dipole_to_surface_normal(cdm)

    eeg = nyhead.get_transformation_matrix() @ p * 1e9  # pV
    vm_rt = eeg[eeg_elec_idx] / amp_i * amp_TES * 1e-6  # µV

    cdm_amp = np.max(np.abs(cdm))

    max_time_idx = np.argmax(np.abs(eeg[eeg_elec_idx]))
    eeg_amp_max = np.max(np.abs(eeg))

    cmap_v_e = plt.get_cmap('PRGn')
    vmax = np.max(np.abs(eeg[:, max_time_idx]))

    vmap = lambda v: cmap_v_e((v + vmax) / (2*vmax))

    print("Plotting")
    plt.close("all")
    fig = plt.figure(figsize=[6, 5])
    ax1 = fig.add_axes([0.0, 0.51, 0.16, 0.49],
                       xlim=[np.min(cell.x) - 10, np.max(cell.x) + 10],
                       ylim=[np.min(cell.z) - 10, np.max(cell.z) + 10],
                       frameon=False, aspect=1, xticks=[], yticks=[])

    ax_stim = fig.add_axes([0.2, 0.89, 0.2, 0.05], frameon=False,
                           xticks=[], yticks=[],
                           title=f"input current, {freq} Hz")
    ax_p = fig.add_axes([0.2, 0.77, 0.2, 0.05], frameon=False,
                        xticks=[], yticks=[],
                        title="current dipole moment, $P_z$")
    ax_f = fig.add_axes([0.2, 0.58, 0.2, 0.13], ylabel="max$(|P|)$ (nAµm)",
                        xlabel="frequency (Hz)",)
    ax_eeg = fig.add_axes([0.5, 0.55, 0.2, 0.05], frameon=False,
                          xticks=[], yticks=[],
                          title="EEG (max. amp. electrode)")
    ax_ctx_xz = fig.add_axes([0.47, 0.65, 0.28, 0.30], aspect=1,
                             frameon=False,
                             xticks=[], yticks=[])

    # Plot 3D head
    ax_head = fig.add_axes([.74, 0.65, 0.28, 0.35], projection='3d',
                           xlim=[-70, 70], facecolor="none",
                           ylim=[-70, 70], zlim=[-70, 70], rasterized=True,
                           computed_zorder=False,)

    mark_subplots([ax1], "A", ypos=0.95, xpos=0.1)
    mark_subplots([ax_stim, ax_p, ax_f], "BCD")
    mark_subplots([ax_ctx_xz], "E", ypos=0.98, xpos=0.1)
    mark_subplots([ax_eeg], "F")
    fig.text(0.77, 0.95, "G",
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='demibold',
             fontsize=8)

    ax_head.axis('off')
    cax = fig.add_axes([0.77, 0.6, 0.2, 0.01]) # This axis is the colorbar
    mpl_amp = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmax=vmax,
                                         vmin=-vmax), cmap=cmap_v_e)
    cbar_amp = plt.colorbar(mpl_amp, cax=cax, label='EEG (pV)',
                            orientation="horizontal")

    if 1:
        ax_head.plot_trisurf(x_ctx, y_ctx, z_ctx, triangles=cortex_tri,
                             color="pink", zorder=0, rasterized=True)
        ax_head.plot_trisurf(x_h, y_h, z_h, triangles=head_tri,
                             color="#c87137", zorder=0, alpha=0.2)
    all_patches = []
    for elec_idx in range(len(elecs[0, :])):
        elec_normal = elecs[3:, elec_idx]
        elec_xyz = elecs[:3, elec_idx]
        clr = vmap(eeg[elec_idx, max_time_idx])
        p_ = Circle((0, 0), 6, facecolor=clr, zorder=elec_xyz[2],
                    ) #Add a circle in the xy plane
        all_patches.append(p_)
        ax_head.add_patch(p_)
        pathpatch_2d_to_3d(p_, z=0, normal=elec_normal)
        pathpatch_translate(p_, elec_xyz)

    ax_head.view_init(elev=90., azim=-90)
    ax_head.plot(cortex[0, nyhead.vertex_idx],
                 cortex[1, nyhead.vertex_idx],
                 cortex[2, nyhead.vertex_idx], 'o', c='orange', ms=5, mec='k', mew=0.5)

    ax1.plot(cell.x.T, cell.z.T, 'k', lw=1)
    ax1.plot(cell.x[stim_idx].mean(), cell.z[stim_idx].mean(), 'o',
             c='orange', ms=5, mec='k', mew=0.5)

    # axis cross
    ax1.arrow(-150, -250, 100, 0, head_width=8, fc='k', clip_on=False)
    ax1.arrow(-150, -250, 0, 100, head_width=8, fc='k', clip_on=False)
    ax1.plot(-150, -250, 'k.', ms=0.5)
    ax1.text(-150 + 50, -255, "$x$", va="top", ha="center")
    ax1.text(-155, -250 + 50, "$z$", va="center", ha="right")


    lines = []
    line_names = []
    for cell_name in cell_names:
        l, = ax_f.loglog(freqs, cdm_amps[cell_name],
                         c=cell_clrs[cell_name], lw=1, marker='o', ms=2)
        lines.append(l)
        line_names.append(cell_legend_names[cell_name])

    ax_f.legend(lines, line_names, frameon=False, loc=(0., -1.), ncol=2)
    ax_stim.plot(cell.tvec, pulse, c='orange')
    ax_p.plot(cell.tvec, cdm[2], 'k')
    ax_eeg.plot(cell.tvec, eeg[eeg_elec_idx], 'k')
    ax_stim.axvline(cell.tvec[max_time_idx], c='gray', lw=0.5, ls='--')
    ax_p.axvline(cell.tvec[max_time_idx], c='gray', lw=0.5, ls='--')
    ax_eeg.axvline(cell.tvec[max_time_idx], c='gray', lw=0.5, ls='--')

    ax_stim.plot([cell.tvec[-1]*1.05, cell.tvec[-1]*1.05],
                 [0, amp_i], clip_on=False, c='k', lw=0.5)
    ax_stim.text(cell.tvec[-1]*1.06, amp_i / 2, f"{amp_i:0.0f} nA")

    ax_stim.plot([cell.tvec[-1], cell.tvec[-1] - 100],
                 [-amp_i*1.2, -amp_i*1.2], lw=0.5, c='k', clip_on=False)
    ax_stim.text(cell.tvec[-1] - 50, -amp_i * 1.3, "100 ms", va="top", ha="center")

    ax_p.plot([cell.tvec[-1]*1.05, cell.tvec[-1]*1.05],
              [0, cdm_amp], clip_on=False, c='k', lw=0.5)
    ax_p.text(cell.tvec[-1]*1.06, cdm_amp / 2, f"{cdm_amp:0.0f} nAµm")

    ax_eeg.plot([cell.tvec[-1]*1.05, cell.tvec[-1]*1.05],
                [0, eeg_amp_max], clip_on=False, c='k', lw=0.5)
    ax_eeg.text(cell.tvec[-1]*1.06, eeg_amp_max / 2, f"{eeg_amp_max:0.0f} pV")

    ax_f.plot([1e3, 1e4], [5e1, 5e0], c='gray', ls=':', lw=1.7)
    ax_f.text(12e3, 5e0, "1/f", ha='left', c='gray')

    # Plotting crossection of cortex around active region center
    threshold = 2  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] -
                                    dipole_loc[1]) < threshold)[0]

    ax_ctx_xz.scatter(cortex[0, xz_plane_idxs],
                      cortex[2, xz_plane_idxs], s=1, c='0.9')

    ax_ctx_xz.plot([-30, -40], [-30, -30], c='k', lw=1)
    ax_ctx_xz.text(-35, -35, "20 mm", ha='center', va="top")
    ax_ctx_xz.arrow(cortex[0, nyhead.vertex_idx],
                    cortex[2, nyhead.vertex_idx],
                    np.std(p[0])*.1, np.std(p[2])*0.1,
                    color='k', head_width=2)
    ax_ctx_xz.plot(cortex[0, nyhead.vertex_idx],
                    cortex[2, nyhead.vertex_idx], 'o',
                    color='orange', ms=5, mec='k', mew=0.5)


    # ax_ctx_xz.plot(elecs[0, eeg_elec_idx], elecs[2, eeg_elec_idx], 'o',
    #                c='green', ms=5, mec='k', mew=0.5)
    p_ = Ellipse((elecs[0, eeg_elec_idx], elecs[2, eeg_elec_idx]), 6,
                 height=2, facecolor='green', zorder=elec_xyz[2],
                edgecolor='k', lw=0.5, angle=-55) #Add a circle in the xy plane
    ax_ctx_xz.add_patch(p_)

    simplify_axes(ax_f)
    ax_f.set_xticks([1, 10, 100, 1000, 10000])

    ### tES part:
    lead_field_orig = nyhead.lead_field_normal[:, eeg_elec_idx]
    lead_field = lead_field_orig * stim_amp_2_cdm_factor

    lead_field_max = np.max(np.abs(lead_field))

    cmap_v_e = plt.get_cmap('PRGn')
    vmax = np.max(np.abs(lead_field))
    vmap = lambda v: cmap_v_e((v + vmax) / (2*vmax))

    # Plot 3D head
    ax_head = fig.add_axes([-.02, 0.07, 0.28, 0.4], projection='3d',
                           xlim=[-70, 70], facecolor="none",
                           ylim=[-70, 70], zlim=[-70, 70], rasterized=True,
                           computed_zorder=False,
                           )

    ax_stim_rt = fig.add_axes([0.03, 0.01, 0.2, 0.05], frameon=False,
                           xticks=[], yticks=[],
                           title=f"tES current, {freq} Hz")

    ax_ctx_xz = fig.add_axes([0.25, 0.02, 0.4, 0.45], aspect=1,
                             frameon=False,
                             xticks=[], yticks=[], rasterized=True)

    ax_vm = fig.add_axes([0.69, 0.24, 0.26, 0.2], frameon=False,
                          xticks=[], yticks=[],
                          title=r"somatic $V_\mathrm{m}$")

    mark_subplots([ax_stim_rt, ax_vm], "IK", xpos=-.05, ypos=0.95)
    mark_subplots([ax_ctx_xz], "J", xpos=.2, ypos=0.95)
    fig.text(0.05, 0.4, "H",
             horizontalalignment='center',
             verticalalignment='center',
             fontweight='demibold',
             fontsize=8)

    ax_head.axis('off')
    cax = fig.add_axes([0.63, 0.08, 0.2, 0.01]) # This axis is the colorbar
    mpl_amp = matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(vmax=vmax,
                                         vmin=-vmax), cmap=cmap_v_e)
    cbar_amp = plt.colorbar(mpl_amp, cax=cax,
                            label=r'somatic $V_{\rm m}$ (µV)',
                            orientation="horizontal")
    cbar_amp.set_ticks([-int(lead_field_max), 0, int(lead_field_max)])
    # Add secondary scale to the colorbar

    cax_2 = cax.twiny()  # Create a twin axis
    cax_2.set_xlim(cax.get_xlim())  # Match colorbar limits

    cax_2.set_xticks([-int(lead_field_max), 0, int(lead_field_max)])  # Match primary scale positions
    cax_2.set_xticklabels([f"{val:.2f}" for val in [-lead_field_max / stim_amp_2_cdm_factor,
                                                    0, lead_field_max / stim_amp_2_cdm_factor]])  # Secondary scale labels
    cax_2.set_xlabel(r'$E_{\rm cn}$ (mV/mm)', labelpad=3)

    if 1:
        ax_head.plot_trisurf(x_ctx, y_ctx, z_ctx, triangles=cortex_tri,
                             color="pink", zorder=0, rasterized=True)

        ax_head.plot_trisurf(x_h, y_h, z_h, triangles=head_tri,
                             color="#c87137", zorder=0, alpha=0.2)
    all_patches = []
    for elec_idx in [eeg_elec_idx]:
        elec_normal = elecs[3:, elec_idx]
        elec_xyz = elecs[:3, elec_idx]
        clr = 'orange'
        p_ = Circle((0, 0), 5, facecolor=clr, zorder=elec_xyz[2],
                    edgecolor='k', lw=0.5,
                    ) #Add a circle in the xy plane
        all_patches.append(p_)
        ax_head.add_patch(p_)
        pathpatch_2d_to_3d(p_, z=0, normal=elec_normal)
        pathpatch_translate(p_, elec_xyz)

    ax_head.view_init(elev=90., azim=-90)
    ax_head.plot(cortex[0, nyhead.vertex_idx],
                 cortex[1, nyhead.vertex_idx],
                 cortex[2, nyhead.vertex_idx], 'o', c='green', ms=5, mew=0.5, mec='k')

    ax_stim_rt.plot(cell.tvec, pulse / amp_i * amp_TES * 1e-6, c='orange')
    ax_stim_rt.plot([cell.tvec[-1]*1.05, cell.tvec[-1]*1.05],
                 [0, amp_TES * 1e-6], clip_on=False, c='k', lw=0.5)
    ax_stim_rt.text(cell.tvec[-1]*1.06, amp_TES * 1e-6/2, f"{amp_TES * 1e-6:0.0f} mA")
    cell_clr = plt.cm.tab10(0 / 3)

    print("RT error: ", np.std(dV_TES - vm_rt) / np.std(dV_TES))

    l1, = ax_vm.plot(cell.tvec, dV_TES, c=cell_clr)
    l2, = ax_vm.plot(cell.tvec, vm_rt, 'k--')
    ax_vm.legend([l1, l2], [r"tES", "reciprocity-based"],
                 frameon=False, loc=(0.25, -0.35))

    ax_vm.plot([cell.tvec[-1]*1.05, cell.tvec[-1]*1.05],
                [0, eeg_amp_max], clip_on=False, c='k', lw=0.5)
    ax_vm.text(cell.tvec[-1]*1.06, eeg_amp_max/2, f"{eeg_amp_max:0.0f} µV")

    ax_stim_rt.axvline(cell.tvec[max_time_idx], c='gray', lw=0.5, ls='--')
    ax_vm.axvline(cell.tvec[max_time_idx], c='gray', lw=0.5, ls='--')

    # Plotting crossection of cortex around active region center
    threshold = 1  # threshold in mm for including points in plot
    xz_plane_idxs = np.where(np.abs(cortex[1, :] -
                                    elec_loc[1]) < threshold)[0]

    ax_ctx_xz.scatter(cortex[0, xz_plane_idxs],
                      cortex[2, xz_plane_idxs], s=1,
                      c=vmap(lead_field[xz_plane_idxs]))

    ax_ctx_xz.plot([-30, -40], [-30, -30], c='k', lw=1)
    ax_ctx_xz.text(-35, -32, "20 mm", ha='center', va="top")

    p_ = Ellipse((elecs[0, eeg_elec_idx], elecs[2, eeg_elec_idx]), 6,
                 height=2, facecolor='orange', zorder=elec_xyz[2],
                edgecolor='k', lw=0.5, angle=-55) #Add a circle in the xy plane
    ax_ctx_xz.add_patch(p_)

    fig.savefig(join(fig_folder, f"EEG_{sim_name}.png"))
    fig.savefig(join(fig_folder, f"EEG_{sim_name}.pdf"))



if __name__ == '__main__':
    fig_folder = join(root_dir, 'figs')
    results_folder = join(root_dir, 'sim_results')
    os.makedirs(fig_folder, exist_ok=True)
    os.makedirs(results_folder, exist_ok=True)

    # Figure 2:
    # make_RT_validation_plot(results_folder, fig_folder)

    # Figure 3:
    # neural_elements_fig(results_folder, fig_folder)

    # Figure 4:
    analytic_ext_pot(fig_folder)

    # Figure 5:
    # detailed_head_model_neuron_fig(results_folder, fig_folder)