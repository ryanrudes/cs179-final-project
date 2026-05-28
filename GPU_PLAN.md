# Introduction and Brainstorming

I will be doing motion retargeting over a full time sequence. Typical robot arm and humanoid URDFs are less than 32 DOFs, and sequence lengths are many frames. The CPU version normally warm-starts the program solver with the results of frame t for the program at frame t + 1, and the very first frame is warm started from either the nominal pose or IK (as done here). Because of this sequential nature, motion retargeting is generally a very CPU-friendly task, but it can be decently parallelized with some tricks and tradeoffs.

The main issue is coordinating smoothness. This requires velocity/acceleration terms, which tie together the state at multiple adjacent frames. For GPU, we could use short temporal windows, which each solve a small consecutive chunk of frames. This would still leave the problem of coordinating smoothness at window boundaries, which would be addressed with halos. However, I will soon show that this method is non-ideal for our purposes. There are two immediately clear options: you can solve the frames inside each window sequentially, or in parallel. Sequentially is more natural from the CPU implementations, but I will attempt to parallelize this part to the maximum.

To reduce project scope, I will use robot arms, which have simpler FK and Jacobian formulations. But since this project idea is ideally targeted for humanoid robots, and since humanoid robots tend to have in the 20s DOFs (e.g. Booster T1 with 23 and 29 DOF variants), it seems natural to use one warp per frame and one lane per DOF. The closer the DOFs are to 32, the better the warp usage.

The parallelization idea is to use double-buffered parallel window updates. We maintain buffers `q_curr[dof][frame]` and `q_next[dof][frame]` (dof-major so that for each thread, frames are contiguous). The column size will be the number of frames, rounded up to the nearest multiple of 32 for padding/alignment purposes (or possibly 64, depending on the datatype used).

First, imagine we want to use multiple blocks to retarget a single trajectory. The issue is that any coupling between frame windows requires communication every outer iteration, which would have to occur over global memory. The loop would involve repeated outer iterations to propagate continuity. After each outer iteration, we would swap the buffers. In each inner iteration, we would:
1. read halo values from `q_curr`
2. retarget owned frames
3. write owned frames to `q_next`
However, since trajectories are at most a few hundred frames, and a single block can support up to 32 warps (i.e. 32 frames at once), it could better for our cases to have each block retarget a single trajectory, and use multiple blocks to simultaneously retarget different trajectories in the dataset. We need to see if a trajectory can fit entirely within shared memory for this to work. Assuming a shared memory size of 48 KiB per block, a maximum trajectory length of 500 frames, a 32 DOF robot:
We have 2 buffers (read and write buffer). shared memory = num_buffers * dof_pad * time_pad * bytes_per_value. Float is necessary to maintain good enough precision for our purposes, so it is 4 bytes per value. For typical robot arms and humanoids with 3 to 29 DOFs, common DOF padding choices might be 4, 8, 16, or 32. Taking the worst-case of 32, and a typical time alignment of 32, that gives 2 * 32 * T_pad * 4 = 256 * T_pad bytes. This would mean we can fit up to T_pad = 192, which is exactly the 48 KiB limit. For robot arms like the UR3e with 6 joints, we would use a DOF padding of 8, resulting in 2 * 8 * T_pad * 4 = 64 * T_pad, which means up to T_pad = 768. Humanoid demo motions are often around 30 Hz, so this could support 6 second humanoid demos. DROID arm demos are 15 Hz, which means we could support up to 51.2 second long demos. This is fairly restrictive for the humanoid demos, but for short behavioes, it frankly isn't too bad. I think it is a necessary compromise and well worth the efficiency of avoiding excessive global memory writes, so we will go with one trajectory = one block, and retarget multiple trajectories simultaneously using the many blocks in the grid. So, the double buffers represent a full trajectory and are entirely stored in shared memory.

# Main Idea

Each block will handle one trajectory, i.e. `blockIdx.x` = trajectory index.

Each block:
1. loads the full trajectory into shared memory
2. runs all retargeting and smoothing iterations locally
3. writes the final trajectory back to global memory

Within one block per trajectory:
- all frames in the trajectory are visible inside the same block
- all smoothing terms are handled through shared memory
- `__syncthreads()` is enough for synchronization

So, there is no need for halos or global memory communication every outer iteration. Since each warp manages one frame, and the block can have up to 32 warps, we divide trajectories longer than 32 frames into 32-frame tiles. So, we still have to consider tile boundaries, but they are much less problematic than before, since everything is now in shared memory and we can coordinate across tiles with `__syncthreads()`.

# Memory Layout

We use structure-of-arrays in feature (dof)-major order, i.e. `q[dof][frame]`, or flattened, `q[d * T_pad + t]`.

In global memory, we store a batch of trajectories as `q_global[traj_id * D_pad * T_pad + d * T_pad + t]`

`D_pad` = 4, 8, 16, or 32 depending on the number of DOFs. `T_pad` equals the length of the trajectory, rounded up to the nearest multiple of 32. For global memory storage, we could either use the max length among all trajectories, or do per-trajectory padding and separately maintain an offset table.

For shared memory allocation, we would launch with the maximum, making room for two shared memory buffers: `q_curr[D_pad][T_max_pad]` and `q_next[D_pad][T_max_pad]`. This supports Jacobi updates of the form read, write, swap.

Registers will store per-thread scalar temporaries.

Constant memory will store robot model constants, like the following:
- parent indices
- joint axes
- joint limits
- link offsets
- link inertial/geometry constants, if needed
- retargeting correspondences
- task weights
- joint type metadata
- kinematic tree level info

The important condition for constant memory is warp-uniform access, i.e. all lanes in a warp read the same constant address at the same time, since constant memory is best when it is broadcasting one value to a warp. We would prefer SoA here. If a constant is read many times inside a block, we could load it once into a register. Since each warp handles a frame, and each lane is responsible for one DOF, a naive approach to constant memory might result in extremely divergent accesses. Instead, we should iterate over tasks, and load task-specific constant memory into registers, that way we broadcast constant memory across the warp (where a task is just one term in the retargeting objective). As an example purely for understanding:
```cpp
for (int task = 0; task < num_tasks; ++task) {
    int link_id = c_task_link[task];
    int type    = c_task_type[task];
    float w     = c_task_weight[task];

    // Every lane reads the same task metadata.
    // Lane d computes how DOF d affects this task.
    grad_d += compute_task_gradient_for_dof(
        d,
        task,
        link_id,
        type,
        w,
        q_frame,
        target_for_this_frame
    );
}
```

# Shared Memory Calculation

We could determine the amount of shared memory supported per block with:
```cpp
int device = 0;
cudaDeviceProp prop{};
cudaError_t err = cudaGetDeviceProperties(&prop, device);
```

`prop.sharedMemPerBlock` gives the legacy amount, and `prop.sharedMemPerBlockOptin` gives the higher limit available on newer architectures if you configure the kernel attributes accordingly.

The formula for shared memory allocation size is
```
shmem_q = 2 * D_pad * T_max_pad * sizeof(float)
```

We may need to leave a little room for scratch space in shared memory, which could reduce the maximum supported trajectory length. It is hard to know how much, if any, scratch space we would need just yet.

# Thread/Block Mapping

A block owns one trajectory. Inside the block, one warp = one active frame and one lane = one DOF. The mapping is
```cpp
int traj = blockIdx.x;
int lane = threadIdx.x & 31;
int warp = threadIdx.x >> 5;
```
A CUDA block can have up to 1024 threads, which is 32 warps, so a block can actively process up to 32 frames at a time. If the trajectory has more than 32 frames, we process it in 32-frame tiles, but the full trajectory still lives in shared memory.

# Outer Iteration Structure
All outer retargeting/smoothing iterations happen inside the block. For Jacobi updates, we have the following:
```
for each outer iteration
    read all frames from q_curr
    write all updated frames to q_next
    after all frame tiles are processed, swap q_curr and q_next
```
The rule is that, during one iteration, all reads come from `q_curr` and all writes go to `q_next`, i.e. we cannot allow later frame tiles to read newly written earlier frame tiles.

# Temporal Smoothing

Since the full trajectory is stored in shared memory, smoothing is much simpler. We will use velocity and acceleration smoothing with the same formulas from the CPU demo. In the CPU version, we processed frames from start to end in time, using each frame's result as a warm start for the following. For the very first frame, we warm started with IK and omitted both velocity and acceleration smoothing terms. On the final frame, we used one-sided backward finite differences. We couldn't have used one-sided forward finite differences on the first frame because when processing sequentially, we didn't have access to the next frame yet. However, since we know process frames jointly, with each iteration updating each frame across all time, we can use one-sided finite difference smoothing on both ends.

# Retargeting Update

Each active frame warp computes a retargeting update for one frame. A typical update is the sum of a retargeting gradient, a temporal smoothing gradient, and a constraint penalty gradient. Then, a gradient step would usually look like `q_solve = q_0 - step_size * total_grad;`. However, using relaxation with `alpha` around 0.5 to 0.8 could help prevent oscillation in iterative smoothing, i.e. `q_next[d * T_max_pad + t] = (1.0f - alpha) * q_0 + alpha * q_solved;`

# Constraint Projection
After each outer iteration, or after several iterations, we apply constraints inside shared memory. These could potentially include some of the following or others:
- joint limits
- velocity limits
- acceleration limits
- contact consistency
- robot-specific bounds

# Global Memory

Global memory is used only for the input trajectory, the final output trajectory, and source model features. Target model constants go in constant memory as discussed previously. The main $q$ trajectory is loaded once into shared memory, then all iterations happen in shared memory. At the end, the final $q$ trajectory is copied from shared memory back to global memory.

# Handling More Than 32 Frames

A block has at most 32 warps, so can process at most 32 active frames at a time. `FRAMES_PER_TILE = blockDim.x / 32`. The tradeoff is that with more threads per block, we could have more active frames per tile, but possibly more occupancy pressure. With fewer threads per block, we have less parallelism per trajectory, but maybe easier scheduling. We could try different block sizes, and it is likely that different block sizes will be ideal for different robot models. We could start with 256 or 512.

# Occupancy Considerations

One block per trajectory will use significant shared memory, which means only one block resident per SM usually. That makes this implementation ideal for cases where we have many trajectories, each block does enough work, and the kernel is compute-heavy. It is not as good if we only process a couple trajectories, shared memory limits block residency too much, or the retargeting math is too light. However, if we have to retarget only a few trajectories, we could retarget the same trajectory multiple times using different hyperparameter samples or candidate retargeting initializations, allowing us to better utilize the GPU.

If we really do need to end up supporting longer trajectories, we could later consider the original chunked/windowed trajectory mode with global memory communication between outer iterations, but that will be left as an optional addon if time permits.

# Registers
We have up to 255 32-bit registers per thread, but we don't want to be using anywhere near that because for one, it could lead to register spilling excess data into local memory, and also so that we can use more threads and support higher occupancy.

Adam needs extra persistent state for the velocity and momentum buffers, which triples state. L-BFGS needs multiple past vectors, which is even worse. Gauss-Newton / Levenberg-Marquardt could be okay if the system per frame is small and we don't store full dense Jacobians. But using projected gradient descent with damped updates, register usage is definitely best.

Since we need to be as efficient as possible with register usage, I have already separately written code that compiles FK and Jacobian kernels with symbolic reduction and matrix sparsity, which can optimize for the minimum possible register usage. It also only emits the parts of the result that are dependent on $q$, so if some link's position or orientation or the derivative thereof happens to be constant w.r.t $q$, it doesn't emit it, allowing us to minimize register usage in FK and Jacobian computation and handle the constant components separately.

---

# Implementation status (cs179)

| Plan item | Status |
|-----------|--------|
| One block per trajectory, batch over demos | Done (`cuda/src/retarget_gpu.cu`; mixed lengths: max `T_pad` + `lengths[]`, one launch) |
| dof-major `q[d,t]` in shared memory | Done |
| Jacobi `q_curr` / `q_next` double buffer + swap | Done |
| Warp = frame, lane = DOF within 32-frame tiles | Done (`frames_per_tile` default 256 = 8 warps/tile) |
| fastfk `spatial_local` 6×6 `J` + 6D DLS | Done (`kernels/ur3e_link_task_spatial_local_best/tool0/`) |
| Retarget params in `__constant__` memory | Done (`c_rt_params`, `retarget_gpu_set_params`) |
| Temporal vel/acc + neutral-pose gradient | Done (boundary one-sided velocity; interior acc) |
| Joint-limit projection | Done |
| Initial ``q`` = neutral; Jacobi pose/temporal on device | Done (no per-frame host IK; `pack_initial_gpu_q`) |
| Optional elbow refine (Pinocchio) | Done when `elbow_branch` > 0 |
| Long trajectories (> SMEM) | Skipped (`trajectory_fits_gpu_shmem`; no host windowing) |
| Full NLopt cost stack on GPU | Not planned (DLS + temporal grad per Registers section) |
| Constant-memory full kinematic tree | Not needed (symbolic FK in device codegen) |
| Multi-robot / humanoid kernels | Future (UR3e tool0 only) |

Regenerate device FK after kernel updates: `uv run python scripts/generate_device_fk.py` then `./scripts/build_native.sh`.