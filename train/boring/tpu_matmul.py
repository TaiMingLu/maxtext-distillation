"""Simple TPU matrix multiplication to fill utilization on all chips."""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import time

def main():
    # Get all devices
    devices = jax.devices()
    num_devices = len(devices)
    print(f"Found {num_devices} devices: {devices}")

    # Large matrices for good TPU utilization (per device)
    size = 8192

    # Create mesh with all devices
    mesh = Mesh(jax.numpy.array(devices), axis_names=('devices',))

    # Initialize random matrices - one per device
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)

    # Shape: (num_devices, size, size) - each device gets its own matrix
    a = jax.random.normal(key1, (num_devices, size, size), dtype=jnp.bfloat16)
    b = jax.random.normal(key2, (num_devices, size, size), dtype=jnp.bfloat16)

    # Shard across devices (first dimension sharded)
    sharding = NamedSharding(mesh, PartitionSpec('devices', None, None))
    a = jax.device_put(a, sharding)
    b = jax.device_put(b, sharding)

    # JIT compile the matmul - runs on all devices in parallel
    @jax.jit
    def matmul(x, y):
        # vmap over the device dimension, each device does its own matmul
        return jax.vmap(jnp.dot)(x, y)

    # Warm up
    c = matmul(a, b)
    c.block_until_ready()

    print(f"Running matrix multiplication ({size}x{size}) on {num_devices} TPU chips...")
    print("Press Ctrl+C to stop")

    iteration = 0
    try:
        while True:
            c = matmul(a, b)
            c.block_until_ready()
            iteration += 1

            if iteration % 100 == 0:
                print(f"Iteration {iteration}")

            # Very short sleep to allow interrupts
            time.sleep(0.001)
    except KeyboardInterrupt:
        print(f"\nStopped after {iteration} iterations")

if __name__ == "__main__":
    main()
