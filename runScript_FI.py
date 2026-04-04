#!/usr/bin/env python

import argparse
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys.multipoint import Multipoint
from dafoam.mphys import DAFoamBuilder
from mphys.scenario_aerodynamic import ScenarioAerodynamic

parser = argparse.ArgumentParser()
parser.add_argument("-optimizer", help="optimizer to use", type=str, default="IPOPT")
parser.add_argument("-task", help="type of run to do", type=str, default="run_driver")
args = parser.parse_args()

U0 = 1.0
nCells = 12225
np.random.seed(0)

daOptions = {
    "solverName": "DASimpleFoam",
    "primalMinResTol": 1.0e-8,
    "primalMinResTolDiff": 1e3,
    "primalBC": {
        "U0": {"variable": "U", "patches": ["inlet"], "value": [U0, 0, 0]},
        "useWallFunction": True,
    },
    "primalVarBounds": {
        "omegaMin": -1e16,
    },
    "regressionModel": {
        "active": True,
        "model": {
            "modelType": "neuralNetwork",
            "inputNames": ["PoD", "VoS", "PSoSS", "KoU2"],
            "outputName": "dummy",
            "hiddenLayerNeurons": [20, 20],
            "inputShift": [0.0, 0.0, 0.0, 0.0],
            "inputScale": [1.0, 1.0, 1.0, 1.0],
            "outputShift": 1.0,
            "outputScale": 1.0,
            "activationFunction": "tanh",
            "printInputInfo": True,
            "defaultOutputValue": 1.0,
            "outputUpperBound": 1e1,
            "outputLowerBound": -1e1,
            "writeFeatures": True,
        },
    },
    "function": {
        "pVar": {
            "type": "variance",
            "source": "patchToFace",
            "patches": ["lowerWall"],
            "scale": 1.0,
            "mode": "surface",
            "varName": "p",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "UFieldVar": {
            "type": "variance",
            "source": "boxToCell",
            "min": [-10.0, -10.0, -10.0],
            "max": [40.0, 10.0, 10.0],
            "scale": 0.1,
            "mode": "field",
            "varName": "U",
            "varType": "vector",
            "indices": [0, 1],
            "timeDependentRefData": False,
        },
        "CDVar": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["lowerWall"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
            "calcRefVar": True,
            "ref": [3.2285619e-06],
        },
        "betaVar": {
            "type": "variance",
            "source": "allCells",
            "scale": 0.01,
            "mode": "field",
            "varName": "betaFIOmega",
            "varType": "scalar",
            "timeDependentRefData": False,
        },
        "CD": {
            "type": "force",
            "source": "patchToFace",
            "patches": ["lowerWall"],
            "directionMode": "fixedDirection",
            "direction": [1.0, 0.0, 0.0],
            "scale": 1.0,
        },
    },
    "adjStateOrdering": "cell",
    "adjEqnOption": {
        "gmresRelTol": 1.0e-6,
        "pcFillLevel": 1,
        "jacMatReOrdering": "natural",
    },
    "normalizeStates": {
        "U": U0,
        "p": U0 * U0 / 2.0,
        "nuTilda": 1e-3,
        "phi": 1.0,
    },
    "inputInfo": {
        "beta": {
            "type": "field",
            "fieldName": "betaFIOmega",
            "fieldType": "scalar",
            "distributed": False,
            "components": ["solver", "function"],
        },
    },
}


class Top(Multipoint):
    def setup(self):
        dafoam_builder = DAFoamBuilder(
            options=daOptions,
            mesh_options=None,
            scenario="aerodynamic",
            run_directory=".",
        )
        dafoam_builder.initialize(self.comm)

        self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
        self.mphys_add_scenario(
            "scenario1", ScenarioAerodynamic(aero_builder=dafoam_builder)
        )
        self.add_subsystem("obj", om.ExecComp("val=error1+error2+error3+regulation"))

    def configure(self):
        self.dvs.add_output("beta", val=np.ones(nCells), distributed=False)
        self.connect("beta", "scenario1.beta")

        self.add_design_var("beta", lower=-5.0, upper=10.0, scaler=1.0)

        self.connect("scenario1.aero_post.pVar", "obj.error1")
        self.connect("scenario1.aero_post.UFieldVar", "obj.error2")
        self.connect("scenario1.aero_post.CDVar", "obj.error3")
        self.connect("scenario1.aero_post.betaVar", "obj.regulation")
        self.add_objective("obj.val", scaler=1.0)


# === Εδώ είναι ΕΞΩ από την κλάση ===
prob = om.Problem()
prob.model = Top()
prob.setup(mode="rev")
om.n2(prob, show_browser=False, outfile="mphys.html")

prob.driver = om.pyOptSparseDriver()
prob.driver.options["optimizer"] = args.optimizer

if args.optimizer == "IPOPT":
    prob.driver.opt_settings = {
        "tol": 1.0e-5,
        "constr_viol_tol": 1.0e-5,
        "max_iter": 50,
        "print_level": 5,
        "output_file": "opt_IPOPT.txt",
        "mu_strategy": "adaptive",
        "limited_memory_max_history": 10,
        "nlp_scaling_method": "none",
        "alpha_for_y": "full",
        "recalc_y": "yes",
    }
elif args.optimizer == "SLSQP":
    prob.driver.opt_settings = {
        "ACC": 1.0e-5,
        "MAXIT": 100,
        "IFILE": "opt_SLSQP.txt",
    }
else:
    print("optimizer arg not valid!")
    exit(1)

prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
prob.driver.options["print_opt_prob"] = True
prob.driver.hist_file = "OptView.hst"

if args.task == "run_driver":
    prob.run_driver()
elif args.task == "run_model":
    prob.run_model()
elif args.task == "compute_totals":
    prob.run_model()
    totals = prob.compute_totals()
    if MPI.COMM_WORLD.rank == 0:
        print(totals)
elif args.task == "check_totals":
    prob.run_model()
    prob.check_totals(compact_print=False, step=1e-3, form="central", step_calc="abs")
else:
    print("task arg not found!")
    exit(1)