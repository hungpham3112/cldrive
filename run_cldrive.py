import io
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

CLDRIVE = "bazel-bin/gpu/cldrive/cldrive"
KERNEL_DIR = "kernels"
BACKUP_DIR = "cldrive-backup"
MEM_ANALYSIS_DIR = "mem-analysis"
os.makedirs(BACKUP_DIR, exist_ok=True)
TIMEOUT = 100
NRUN = 10
NUM_GPU = 1
verbose_cldrive = False
device_num_sm = 72  # {"GPU|NVIDIA|NVIDIA_GeForce_RTX_3090|535.86.05|3.0": 82}
BACKUPED_LIST = [os.path.splitext(path)[0] for path in os.listdir(BACKUP_DIR)]

random.seed(2610)

logger.remove(0)
logger.add("cldrive.log", level="DEBUG")



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
        logging.error(cmd)
        logging.error(e)
    return [platform for platform in stdout.split("\n") if len(platform) > 0]


def RunCLDrive(
    src_file: str,
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
        logging.warn(
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
                cmd = f"""{f"timeout -s9 {timeout}" if timeout > 0 else ""} {CLDRIVE} --srcs={src_file} --cl_build_opt="-I{pathlib.Path(hf.name).resolve().parent}{f',{",".join(extra_args)}' if len(extra_args) > 0 else ""}" --num_runs={num_runs} --gsize={gsize} --lsize={lsize} --envs={cl_platform}"""
                if verbose_cldrive:
                    print(cmd)
                    # print(src)
                proc = subprocess.Popen(
                    cmd.split(),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                stdout, stderr = proc.communicate()
        else:
            # f.write(src)
            f.flush()
            cmd = "{} {} --srcs={} {} --num_runs={} --gsize={} --lsize={} --envs={} --mem_analysis_dir={}".format(
                f"timeout -s9 {timeout}" if timeout > 0 else "",
                CLDRIVE,
                src_file,
                f'--cl_build_opt={",".join(extra_args)}'
                if len(extra_args) > 0
                else "",
                num_runs,
                gsize,
                lsize,
                cl_platform,
                MEM_ANALYSIS_DIR,
            )
            if verbose_cldrive:
                print(cmd)
                # print(src)
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


def GetCLDriveDataFrame(
    src_file: str,
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
        src_file,
        header_file=header_file,
        num_runs=num_runs,
        gsize=gsize,
        lsize=lsize,
        extra_args=extra_args,
        timeout=timeout,
        cl_platform=cl_platform,
    )
    try:
        df = pd.read_csv(io.StringIO(stdout), sep=",")
    except Exception as e:
        df = None

    return df, stderr


def get_config():
    # including both simple case (multiple of 32) and complex case (not multiple of 32)
    n_sample_local = 4
    n_sample_wg = 50
    n_sample_small_wg = int(n_sample_wg * 0.25)
    n_sample_medium_wg = int(n_sample_wg * 0.6)
    n_sample_large_wg = int(n_sample_wg * 0.15)
    local_sizes = [32 * i for i in range(1, 32)]
    small_wg_sizes = list(range(1, device_num_sm))
    medium_wg_sizes = list(range(device_num_sm, device_num_sm * 20))
    large_wg_sizes = list(range(device_num_sm * 20, device_num_sm * 1000))
    MAX_GSIZE = int(1e7) - 1

    def gen_launch_configs():
        launch_configs = []
        l_samples = random.sample(local_sizes, n_sample_local)

        wg_samples = []
        wg_samples.extend(random.sample(small_wg_sizes, n_sample_small_wg))
        wg_samples.extend(random.sample(medium_wg_sizes, n_sample_medium_wg))
        wg_samples.extend(random.sample(large_wg_sizes, n_sample_large_wg))

        for lsize in l_samples:
            for wg_size in wg_samples:
                gsize = lsize * wg_size
                if gsize > MAX_GSIZE:
                    gsize = lsize * int(MAX_GSIZE / lsize)
                launch_configs.append((gsize, lsize))
        return launch_configs

    def gen_kernel_path_configs():
        # Get a list of all files and subdirectories in the specified directory
        all_files_and_directories = os.listdir(KERNEL_DIR)

        # Filter the list to include only files (not directories)
        kernel_paths = [
            os.path.join(KERNEL_DIR, item)
            for item in all_files_and_directories
            if os.path.isfile(os.path.join(KERNEL_DIR, item))
        ]
        return kernel_paths

    kernel_path_configs = gen_kernel_path_configs()
    launch_configs = gen_launch_configs()
    nrun = NRUN
    for kernel in kernel_path_configs:
        launch_configs = gen_launch_configs()
        for launch_config in launch_configs:
            yield {
                "kernel_path": kernel,
                "num_runs": nrun,
                "gsize": launch_config[0],
                "lsize": launch_config[1],
            }


def get_hash_kernel_instance(kernel_path, gsize, lsize, device_name):
    return hashlib.sha256(
        kernel_path.encode("utf-8")
        + str(gsize).encode("utf-8")
        + str(lsize).encode("utf-8")
        + str(device_name).encode("utf-8")
    ).hexdigest()


def get_kernel_cldrive_df(config):
    with open(config["kernel_path"], "r", encoding="utf-8") as f:
        src = f.read()

    cl_platform = getOpenCLPlatforms()[0]
    h = get_hash_kernel_instance(
        config["kernel_path"], config["gsize"], config["lsize"], cl_platform
    )

    if str(h) in BACKUPED_LIST:
        logger.debug(
            f"Skip {h} with kernel {config['kernel_path']}, gsize {config['gsize']}, lsize {config['lsize']}"
        )
        return pd.read_csv(os.path.join(BACKUP_DIR, f"{h}.csv")).to_dict("records")[0]

    logger.debug(
        f"Running {h} with kernel {config['kernel_path']}, gsize {config['gsize']}, lsize {config['lsize']}"
    )
    df, stderr = GetCLDriveDataFrame(
        src_file=config["kernel_path"],
        num_runs=config["num_runs"],
        lsize=config["lsize"],
        gsize=config["gsize"],
        cl_platform=cl_platform,
    )

    result = {
        "kernel_path": config["kernel_path"],
        "num_runs": config["num_runs"],
        "gsize": config["gsize"],
        "lsize": config["lsize"],
        "kernel_name": df["kernel"][0]
        if df is not None
        else "",  # assume one kernel per file
        "outcome": df["outcome"][0] if df is not None else "FAILED",
        "device_name": df["device"][0] if df is not None else "FAILED",
        "work_item_local_mem_size": df["work_item_local_mem_size"][0]
        if df is not None
        else 0,
        "work_item_private_mem_size": df["work_item_private_mem_size"][0]
        if df is not None
        else 0,
        "transferred_bytes": df["transferred_bytes"].to_list()
        if df is not None
        else [],
        "transfer_time_ns": df["transfer_time_ns"].to_list() if df is not None else [],
        "kernel_time_ns": df["kernel_time_ns"].to_list() if df is not None else [],
        "stderr": stderr,
    }
    pd.DataFrame([result]).to_csv(os.path.join(BACKUP_DIR, f"{h}.csv"), index=None)
    return result


def wrapping_func(config):
    return retry_call(
        get_kernel_cldrive_df,
        fargs=[config],
        tries=2,
        delay=0.1,
        jitter=0.2,
        logger=logger,
    )


def set_cuda_visible():
    process_number = multiprocessing.current_process()._identity[0] - 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(process_number)


if __name__ == "__main__":
    # Define the list of elements you want to process

    configs = pd.read_json("mem_analysis_pilot.jsonl", orient="records", lines=True).to_dict("records")
    for config in configs:
        config.__delitem__("outcome")
        config["num_runs"] = NRUN

    # # Create a pool of 4 processes
    num_processes = NUM_GPU
    with multiprocessing.Pool(
        processes=num_processes, initializer=set_cuda_visible
    ) as pool:
        # Use the pool to map the process_element function to the elements
        results = list(tqdm(pool.imap(wrapping_func, configs), total=len(configs)))

    # # for config in tqdm(get_config()):
    # #     results.append(get_kernel_cldrive_df(config))
    pd.DataFrame(results).to_csv("cldrive_results.csv", index=None)
