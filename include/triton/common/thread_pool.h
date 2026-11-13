// Copyright 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace triton { namespace common {

// Generic fixed-size Thread Pool to execute tasks asynchronously

class ThreadPool {
 public:
  explicit ThreadPool(std::size_t thread_count);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  using Task = std::function<void(void)>;
  // Assigns "task" to the task queue for a worker thread to execute when
  // available. This will not track the return value of the task.
  void Enqueue(Task&& task);

  // Returns the number of tasks waiting in the queue
  size_t TaskQueueSize() const
  {
    return queue_size_.load(std::memory_order_relaxed);
  }

  // Returns the number of threads in thread pool
  size_t Size() const { return workers_.size(); }

 private:
  struct alignas(64) Shard {
    std::mutex mtx;
    std::condition_variable cv;
    std::deque<Task> tasks;
    bool waiting = false;
    bool stop = false;
    std::atomic<bool> idle{false};
  };

  void WorkerLoop(size_t index);
  bool Steal(size_t index, Task& task);

  size_t shard_count_;
  std::unique_ptr<Shard[]> shards_;
  std::atomic<size_t> next_shard_{0};
  std::atomic<size_t> queue_size_{0};
  std::vector<std::thread> workers_;
};

}}  // namespace triton::common
