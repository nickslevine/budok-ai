"""Save samplers for Run 2 iter 5/10/15 checkpoints for eval."""
import asyncio
import tinker

BASE = "tinker://95f6bba2-6013-5394-97ac-5278b26fbbfd:train:0/weights/rl-selfplay-iter"
ITERS = [5, 10, 15]


async def main() -> None:
    service = tinker.ServiceClient()
    for iter_num in ITERS:
        ckpt = f"{BASE}{iter_num}"
        print(f"Loading iter {iter_num}: {ckpt}")
        tc = await service.create_training_client_from_state_async(path=ckpt)
        fut = await tc.save_weights_for_sampler_async(
            name=f"rl-selfplay-run2-iter{iter_num}-eval"
        )
        res = await fut.result_async()
        print(f"SAMPLER iter{iter_num}: {res.path}")


if __name__ == "__main__":
    asyncio.run(main())
