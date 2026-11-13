// Copyright 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "triton/common/thread_pool.h"

#include <stdexcept>

namespace triton { namespace common {

ThreadPool::ThreadPool(size_t thread_count)
    : shard_count_(thread_count)
{
  if (!thread_count) {
    throw std::invalid_argument("Thread count must be greater than zero.");
  }

  shards_.reset(new Shard[thread_count]);
  workers_.reserve(thread_count);
  for (size_t i = 0; i < thread_count; ++i) {
    workers_.emplace_back([this, i]() { WorkerLoop(i); });
  }
}

ThreadPool::~ThreadPool()
{
  for (size_t i = 0; i < shard_count_; ++i) {
    std::lock_guard<std::mutex> lk(shards_[i].mtx);
    shards_[i].stop = true;
  }
  for (size_t i = 0; i < shard_count_; ++i) {
    shards_[i].cv.notify_all();
  }
  for (auto& t : workers_) {
    t.join();
  }
}

bool
ThreadPool::Steal(size_t index, Task& task)
{
  bool contended = false;
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t k = 1; k < shard_count_; ++k) {
      Shard& victim = shards_[(index + k) % shard_count_];
      std::unique_lock<std::mutex> lk(victim.mtx, std::defer_lock);
      if (pass == 0) {
        if (!lk.try_lock()) {
          contended = true;
          continue;
        }
      } else {
        lk.lock();
      }
      if (!victim.tasks.empty()) {
        task = std::move(victim.tasks.front());
        victim.tasks.pop_front();
        queue_size_.fetch_sub(1, std::memory_order_relaxed);
        return true;
      }
    }
    if (!contended) {
      break;
    }
  }
  return false;
}

void
ThreadPool::WorkerLoop(size_t index)
{
  Shard& own = shards_[index];
  while (true) {
    Task task;
    bool stolen = false;
    {
      std::unique_lock<std::mutex> lk(own.mtx);
      if (own.tasks.empty() && !own.stop) {
        lk.unlock();
        stolen = Steal(index, task);
        if (!stolen) {
          lk.lock();
          while (own.tasks.empty() && !own.stop) {
            own.waiting = true;
            own.idle.store(true, std::memory_order_relaxed);
            own.cv.wait(lk);
            own.waiting = false;
            own.idle.store(false, std::memory_order_relaxed);
          }
        }
      }
      if (!stolen) {
        if (own.tasks.empty()) {
          break;
        }
        task = std::move(own.tasks.front());
        own.tasks.pop_front();
        queue_size_.fetch_sub(1, std::memory_order_relaxed);
      }
    }

    if (task) {
      task();
    }
  }
}

void
ThreadPool::Enqueue(Task&& task)
{
  const size_t start =
      next_shard_.fetch_add(1, std::memory_order_relaxed) % shard_count_;
  size_t target = start;
  for (size_t k = 0; k < shard_count_; ++k) {
    const size_t candidate = (start + k) % shard_count_;
    if (shards_[candidate].idle.load(std::memory_order_relaxed)) {
      target = candidate;
      break;
    }
  }

  Shard& shard = shards_[target];
  bool wake = false;
  {
    std::lock_guard<std::mutex> lk(shard.mtx);
    // Don't accept more work if pool is shutting down
    if (shard.stop) {
      return;
    }
    shard.tasks.push_back(std::move(task));
    queue_size_.fetch_add(1, std::memory_order_relaxed);
    wake = shard.waiting;
  }
  if (wake) {
    shard.cv.notify_one();
  }
}

}}  // namespace triton::common
