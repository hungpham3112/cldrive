import io
from math import e
import pathlib
import subprocess
import tempfile
import typing
import pandas as pd
import logging
import os
from tqdm import tqdm
import multiprocessing
import random
import hashlib
from retry.api import retry_call
from loguru import logger
from sklearn.linear_model import LinearRegression
import numpy as np
import json


CLDRIVE = "bazel-bin/gpu/clmem/clmem"
KERNEL_DIR = "kernels-modified"
BACKUP_DIR = "backup"
TIMEOUT = 5
NUM_GPU = 1
NUM_PROCESS = 16
verbose_cldrive = False
device_num_sm = 82  # {"GPU|NVIDIA|NVIDIA_GeForce_RTX_3090|535.86.05|3.0": 82}
n_sample_local = 4
n_sample_wg = 50


random.seed(2610)

logger.remove(0)
logger.add("cldrive.log", level="DEBUG")
rng = np.random.default_rng()



def getOpenCLPlatforms() -> None:
    """
    Identify compatible OpenCL platforms for current system.
    """
    try:
        cmd = subprocess.Popen(
            f"{CLDRIVE} --clinfo".split(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        stdout, stderr = cmd.communicate()
        if stderr:
            raise ValueError(stderr)
    except Exception as e:
        logger.error(cmd)
        logger.error(e)
    return [platform for platform in stdout.split("\n") if len(platform) > 0]


def RunCLDrive(
    src: str,
    header_file: str = None,
    num_runs: int = 1000,
    gsize: int = 4096,
    lsize: int = 1024,
    extra_args: typing.List[str] = [],
    timeout: int = 0,
    cl_platform: str = None,
) -> str:
    """
    If CLDrive executable exists, run it over provided source code.
    """
    if not CLDRIVE:
        logger.warn(
            "CLDrive executable has not been found. Skipping CLDrive execution."
        )
        return ""

    tdir = tempfile.mkdtemp()

    with tempfile.NamedTemporaryFile(
            "w", prefix="benchpress_opencl_cldrive", suffix=".cl", dir=tdir
        ) as f:
        if header_file:
            with tempfile.NamedTemporaryFile(
                            "w", prefix="benchpress_opencl_clheader", suffix=".h", dir=tdir
                        ) as hf:
                f.write(f'#include "{pathlib.Path(hf.name).resolve().name}"\n{src}')
                f.flush()
                hf.write(header_file)
                hf.flush()
                cmd = f"""{f"timeout -s9 {timeout}" if timeout > 0 else ""} {CLDRIVE} --srcs={f.name} --cl_build_opt="-I{pathlib.Path(hf.name).resolve().parent}{f',{",".join(extra_args)}' if len(extra_args) > 0 else ""}" --num_runs={num_runs} --gsize={gsize} --lsize={lsize} --envs={cl_platform}"""
                if verbose_cldrive:
                    print(cmd)
                    print(src)
                proc = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                stdout, stderr = proc.communicate()
        else:
            f.write(src)
            f.flush()
            cmd = "{} {} --srcs={} {} --num_runs={} --gsize={} --lsize={} --envs={} --output_format=null".format(
                f"timeout -s9 {timeout}" if timeout > 0 else "",
                CLDRIVE,
                f.name,
                f'--cl_build_opt={",".join(extra_args)}'
                if len(extra_args) > 0
                else "",
                num_runs,
                gsize,
                lsize,
                cl_platform,
            )
            if verbose_cldrive:
                print(cmd)
                print(src)
            proc = subprocess.Popen(
                cmd.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )
            try:
                stdout, stderr = proc.communicate()
            except UnicodeDecodeError:
                return "", ""
        if proc.returncode == 9:
            stderr = "TIMEOUT"
    return stdout, stderr


def GetCLDriveStdout(
    src: str,
    header_file: str = None,
    num_runs: int = 5,
    gsize: int = 4096,
    lsize: int = 1024,
    extra_args: typing.List[str] = [],
    timeout: int = 0,
    cl_platform: str = None,
) -> pd.DataFrame:
    """
    Run CLDrive with given configuration and return pandas dataframe.
    """
    stdout, stderr = RunCLDrive(
        src,
        header_file=header_file,
        num_runs=num_runs,
        gsize=gsize,
        lsize=lsize,
        extra_args=extra_args,
        timeout=timeout,
        cl_platform=cl_platform,
    )

    return stdout, stderr


def get_config():
    # including both simple case (multiple of 32) and complex case (not multiple of 32
    local_sizes = [4, 16, 24, 32]
    small_wg_sizes = list(range(1, 80))
    MAX_GSIZE = int(1e5) - 1

    def gen_launch_configs():
        launch_configs = []
        l_samples = random.sample(local_sizes, n_sample_local)

        wg_samples = []
        wg_samples.extend(random.sample(small_wg_sizes, n_sample_wg))

        for lsize in l_samples:
            for wg_size in wg_samples:
                gsize = lsize * wg_size
                if gsize > MAX_GSIZE:
                    gsize = lsize * int(MAX_GSIZE / lsize)
                launch_configs.append((gsize, lsize))
        return launch_configs
    launch_configs = gen_launch_configs()
    return launch_configs

class KernelRunInstance:
    def __init__(self, kernel_code, gsize, lsize) -> None:
        self.kernel_code = kernel_code
        self.gsize = gsize
        self.lsize = lsize
        self.cl_platform = getOpenCLPlatforms()[0]
    
    def run(self):
        stdout, stderr = GetCLDriveStdout(
            self.kernel_code,
            num_runs=1,
            lsize=self.lsize,
            gsize=self.gsize,
            cl_platform=self.cl_platform,
        )

        return stdout, stderr
  
class KernelMemoryAnalyzer:
    def __init__(self, kernel_path) -> None:
        self.kernel_path = kernel_path
        with open(kernel_path, "r", encoding="utf-8") as f:
            self.kernel_code = f.read()
        self.argument_list = self.get_argument_list()
        self.id2arg = {i: arg for (arg, i) in self.argument_list}
    
    def get_argument_list(self):
        # get argument list from the first line of kernel code
        argument_json = self.kernel_code.split("\n")[0]
        argument_json = argument_json[2:] # ignore the first "//"
        argument_dict_obj = json.loads(argument_json)
        return list(argument_dict_obj.items())
        
    
    def get_array_bound_relation(self):
        try:
            X, ys = self.get_mem_access_dataset()
        
            result = {}
            model = LinearRegression()
            for arg_name, arg_id in self.argument_list:
                if arg_name not in ys.keys(): continue
                y = ys[arg_name]
                # fit datapoint into a linear function
                # array_bound = a_2*gsize + a_1*lsize + a_0
                model.fit(X, y)
                
                result[arg_name] = {"coef": np.concatenate((model.coef_, [model.intercept_])).tolist(), "arg_id": arg_id}
            return result
        except Exception as e:
            logger.error(f"ERROR: Get memory access dataset failed with error:\n{e}")
            return {}
    
    def get_mem_access_dataset(self):
        launch_configs = list(get_config())
        max_ids, min_ids, launch_configs = self.run_multiple_configs(launch_configs)
        X = launch_configs
        ys = pd.DataFrame(max_ids).to_dict("list")
        return X, ys
    
    def run_multiple_configs(self, launch_configs):
        # # Create a pool of 4 processes
        with multiprocessing.Pool(
            processes=NUM_PROCESS, initializer=set_cuda_visible
        ) as pool:
            # Use the pool to map the process_element function to the elements
            results = tqdm(pool.imap(self.wrapping_func, launch_configs), total=len(launch_configs))
            max_list, min_list, launch_config = zip(*results)
            return max_list, min_list, launch_config
            
    def wrapping_func(self, config):
        return retry_call(
            self.get_max_min_of_run,
            fargs=[config],
            tries=5,
            delay=0.5,
            jitter=0.2,
            logger=logger,
        )
    
    def get_max_min_of_run(self, launch_config):
        gsize, lsize = launch_config
        kernel_instance = KernelRunInstance(self.kernel_code, gsize, lsize)
        stdout, stderr = kernel_instance.run()
        if len(stdout) == 0:
            raise Exception(f"ERROR: Run kernel (`{self.kernel_path}`, gsize={gsize}, lsize={lsize}) failed with error:\n{stderr}")
        # for each memory access pair (arg_id, id_access), get the max and min of the run
        # return a list of (arg_id, max, min)
        max_list, min_list = self.parse_stdout(stdout)
        max_dict = {self.id2arg[x["arg_id"]]: x["id_access"] for x in max_list}
        min_dict = {self.id2arg[x["arg_id"]]: x["id_access"] for x in min_list}
        return max_dict, min_dict, launch_config
        
    def parse_stdout(self, stdout):
        # parse the stdout of CLDrive
        # return a list of (arg_id, id_access)
        stdout = "arg_id,id_access\n" + stdout
        df = pd.read_csv(io.StringIO(stdout))
        df_group = df.groupby("arg_id")
        max_list = df_group.agg({"id_access":'max'}).reset_index().to_dict("records")
        min_list = df_group.agg({"id_access":'min'}).reset_index().to_dict("records")
        return max_list, min_list


def set_cuda_visible():
    process_number = rng.integers(0, NUM_GPU)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_number)


if __name__ == "__main__":
    BACKUPED_LIST = set()
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR, exist_ok=True)
    else:
        BACKUPED_LIST = {
            os.path.splitext(kernel)[0] for kernel in os.listdir(BACKUP_DIR)
        }

    need_calculate = []
    for kernel in os.listdir(KERNEL_DIR):
        kernel_path = os.path.join(KERNEL_DIR, kernel)
        if os.path.splitext(kernel_path)[1] != ".cl" or os.path.splitext(kernel)[0] in BACKUPED_LIST:
            continue
        need_calculate.append(kernel)

    for kernel in tqdm(need_calculate):
        kernel_path = os.path.join(KERNEL_DIR, kernel)
        logger.info(f"Analyzing {kernel_path}")

        analyzer = KernelMemoryAnalyzer(kernel_path)
        res_dict = analyzer.get_array_bound_relation()
        res_path = os.path.join(BACKUP_DIR, os.path.splitext(kernel)[0] + ".json")
        with open(res_path, "w", encoding="utf-8") as f:
            json.dump(res_dict, f)
