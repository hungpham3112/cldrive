// Copyright (c) 2016-2020 Chris Cummins.
// This file is part of cldrive.
//
// cldrive is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// cldrive is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
#include "gpu/cldrive/kernel_driver.h"

#include "gpu/cldrive/logger.h"
#include "gpu/cldrive/opencl_util.h"
#include "gpu/clinfo/libclinfo.h"
#include "gpu/cldrive/mem_analysis_util.h"

#include "labm8/cpp/logging.h"
#include "labm8/cpp/status_macros.h"

namespace gpu {
namespace cldrive {

KernelDriver::KernelDriver(const cl::Context& context,
                           const cl::CommandQueue& queue,
                           const cl::Kernel& kernel, CldriveInstance* instance,
                           int instance_num)
    : context_(context),
      queue_(queue),
      device_(context.getInfo<CL_CONTEXT_DEVICES>()[0]),
      kernel_(kernel),
      instance_(*instance),
      instance_num_(instance_num),
      kernel_instance_(instance->add_kernel()),
      name_(util::GetOpenClKernelName(kernel)),
      args_set_(&kernel_) {}

void KernelDriver::RunOrDie(Logger& logger) {
  kernel_instance_->set_name(name_);
  kernel_instance_->set_work_item_local_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_LOCAL_MEM_SIZE>(device_));
  kernel_instance_->set_work_item_private_mem_size_in_bytes(
      kernel_.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(device_));

  kernel_instance_->set_outcome(args_set_.Init());
  if (kernel_instance_->outcome() != CldriveKernelInstance::PASS) {
    LOG(WARNING) << "Skipping kernel with unsupported arguments: '" << name_
                 << "'";
    logger.RecordLog(&instance_, kernel_instance_, /*run=*/nullptr,
                     /*log=*/nullptr);
    return;
  }

  for (int i = 0; i < instance_.dynamic_params_size(); ++i) {
    auto run = RunDynamicParams(instance_.dynamic_params(i), logger);
    if (run.ok()) {
      *kernel_instance_->add_run() = run.ValueOrDie();
    } else {
      kernel_instance_->clear_run();
      kernel_instance_->set_outcome(
          CldriveKernelInstance::UNSUPPORTED_ARGUMENTS);
    }
  }
}

labm8::StatusOr<CldriveKernelRun> KernelDriver::RunDynamicParams(
    const DynamicParams& dynamic_params, Logger& logger) {
  CldriveKernelRun run;

  try {
    RunDynamicParams(dynamic_params, logger, &run);
  } catch (cl::Error error) {
    LOG(WARNING) << "Error code " << error.err() << " ("
                 << labm8::gpu::clinfo::OpenClErrorString(error.err()) << ") "
                 << "raised by " << error.what() << "() while driving kernel: '"
                 << name_ << "'";
    run.set_outcome(CldriveKernelRun::CL_ERROR);
    logger.RecordLog(&instance_, kernel_instance_, &run, /*log=*/nullptr);
  }

  return run;
}

namespace {

gpu::libcecl::OpenClKernelInvocation DynamicParamsToLog(
    const DynamicParams& dynamic_params) {
  gpu::libcecl::OpenClKernelInvocation invocation;
  invocation.set_global_size(dynamic_params.global_size_x());
  invocation.set_local_size(dynamic_params.local_size_x());
  // Negative values indicate null.
  invocation.set_kernel_time_ns(-1);
  invocation.set_transfer_time_ns(-1);
  invocation.set_transferred_bytes(-1);
  return invocation;
}

}  // anonymous namespace

labm8::Status KernelDriver::RunDynamicParams(
    const DynamicParams& dynamic_params, Logger& logger,
    CldriveKernelRun* run) {
  // Create a log message with just the dynamic params so that we can log the
  // global and local sizes on error.
  gpu::libcecl::OpenClKernelInvocation log = DynamicParamsToLog(dynamic_params);

  // Check that the dynamic params are within legal range.
  auto max_work_group_size = device_.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
  if (static_cast<int>(max_work_group_size) < dynamic_params.local_size_x()) {
    run->set_outcome(CldriveKernelRun::INVALID_DYNAMIC_PARAMS);
    LOG(WARNING) << "Unsupported dynamic params to kernel '" << name_
                 << "' (local size " << dynamic_params.local_size_x()
                 << " exceeds maximum device work group size "
                 << max_work_group_size << ")";
    logger.RecordLog(&instance_, kernel_instance_, run, &log);
    return labm8::Status(labm8::error::Code::INVALID_ARGUMENT,
                         "Unsupported dynamic params");
  }

  // generate array bound based on dynamic params
  std::map<int,int> memAnalysisInfo = gpu::cldrive::mem_analysis::getMemAnalysisInfo(instance_.mem_filepath(), 
                                                                                      dynamic_params.global_size_x(), 
                                                                                      dynamic_params.local_size_x());
  std::vector<long long> arrayBoundArgs;
  int nArgs = kernel_.getInfo<CL_KERNEL_NUM_ARGS>();
  for (int i = 0; i < nArgs; ++i) {
    // if mem analysis info is not found, then use global size as array bound
    if (memAnalysisInfo.find(i) != memAnalysisInfo.end()) {
      kernel_instance_->add_arg_array_bounds(dynamic_params.global_size_x());
    }
    else {
      kernel_instance_->add_arg_array_bounds(memAnalysisInfo[i]);
    }
  }

  // 2 warmup run
  KernelArgValuesSet inputs;
  KernelArgValuesSet output_a;
  CHECK(args_set_.SetRandom(context_, dynamic_params, &inputs).ok());
  inputs.SetAsArgs(&kernel_);
  RunOnceOrDie(dynamic_params, inputs, &output_a);
  RunOnceOrDie(dynamic_params, inputs, &output_a);
  // We've passed the point of rejecting the kernel. Flush the buffered logs
  // from the preliminary runs.
  logger.PrintAndClearBuffer();

  for (int i = 0; i < instance_.min_runs_per_kernel(); ++i) {
    *run->add_log() =
        RunOnceOrDie(dynamic_params, inputs, &output_a, run, logger);
  }

  run->set_outcome(CldriveKernelRun::PASS);
  return labm8::Status::OK;
}

gpu::libcecl::OpenClKernelInvocation KernelDriver::RunOnceOrDie(
    const DynamicParams& dynamic_params, KernelArgValuesSet& inputs,
    KernelArgValuesSet* outputs, const CldriveKernelRun* const run,
    Logger& logger, bool flush) {
  gpu::libcecl::OpenClKernelInvocation log;
  ProfilingData profiling;
  cl::Event event;

  size_t global_size = dynamic_params.global_size_x();
  size_t local_size = dynamic_params.local_size_x();

  log.set_global_size(global_size);
  log.set_local_size(local_size);
  log.set_kernel_name(name_);

  inputs.CopyToDevice(queue_, &profiling);
  inputs.SetAsArgs(&kernel_);

  queue_.enqueueNDRangeKernel(kernel_, /*offset=*/cl::NullRange,
                              /*global=*/cl::NDRange(global_size),
                              /*local=*/cl::NDRange(local_size),
                              /*events=*/nullptr, /*event=*/&event);
  profiling.kernel_nanoseconds += GetElapsedNanoseconds(event);

  inputs.CopyFromDeviceToNewValueSet(queue_, outputs, &profiling);

  // Set run proto fields.
  log.set_kernel_time_ns(profiling.kernel_nanoseconds);
  log.set_transfer_time_ns(profiling.transfer_nanoseconds);
  log.set_transferred_bytes(profiling.transferred_bytes);
  log.set_args_info(args_set_.ToStringWithValue(inputs));

  logger.RecordLog(&instance_, kernel_instance_, run, &log, flush);

  return log;
}

gpu::libcecl::OpenClKernelInvocation KernelDriver::RunOnceOrDie(
    const DynamicParams& dynamic_params, KernelArgValuesSet& inputs,
    KernelArgValuesSet* outputs) {
  gpu::libcecl::OpenClKernelInvocation log;
  ProfilingData profiling;
  cl::Event event;

  size_t global_size = dynamic_params.global_size_x();
  size_t local_size = dynamic_params.local_size_x();

  log.set_global_size(global_size);
  log.set_local_size(local_size);
  log.set_kernel_name(name_);

  inputs.CopyToDevice(queue_, &profiling);
  inputs.SetAsArgs(&kernel_);

  queue_.enqueueNDRangeKernel(kernel_, /*offset=*/cl::NullRange,
                              /*global=*/cl::NDRange(global_size),
                              /*local=*/cl::NDRange(local_size),
                              /*events=*/nullptr, /*event=*/&event);
  profiling.kernel_nanoseconds += GetElapsedNanoseconds(event);

  inputs.CopyFromDeviceToNewValueSet(queue_, outputs, &profiling);

  // Set run proto fields.
  log.set_kernel_time_ns(profiling.kernel_nanoseconds);
  log.set_transfer_time_ns(profiling.transfer_nanoseconds);
  log.set_transferred_bytes(profiling.transferred_bytes);

  return log;
}

}  // namespace cldrive
}  // namespace gpu
