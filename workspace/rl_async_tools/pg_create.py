from xtuner.v1.ray.base import AcceleratorResourcesConfig, AutoAcceleratorWorkers
import ray

if __name__ == "__main__":
    ray.init(num_cpus=120, ignore_reinit_error=True)

    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=4,
        num_cpus_per_worker=12,
        cpu_memory_per_worker=16 * 1024 ** 3,  # 16 GB
    )
    rollout_pg = AutoAcceleratorWorkers.build_placement_group(resources, name="rollout")
    rollout_pg.wait()
   
    resources = AcceleratorResourcesConfig(
        accelerator="GPU",
        num_workers=4,
        num_cpus_per_worker=12,
        cpu_memory_per_worker=16 * 1024 ** 3,  # 16 GB
    )
    train_pg = AutoAcceleratorWorkers.build_placement_group(resources, name="train")
    train_pg.wait()

    print("Placement group created successfully.")
    print(ray.util.placement_group_table())
    ray.shutdown()



