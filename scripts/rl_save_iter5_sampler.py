"""Load the iter5 persistent checkpoint and save a sampler for eval."""
import asyncio
import tinker

CKPT = "tinker://de755919-e29d-5043-80ba-5e85044d6646:train:0/weights/rl-selfplay-iter5"


async def main() -> None:
    service = tinker.ServiceClient()
    tc = await service.create_training_client_from_state_async(path=CKPT)
    fut = await tc.save_weights_for_sampler_async(name="rl-selfplay-run3b-iter5-eval")
    res = await fut.result_async()
    print(f"SAMPLER: {res.path}")


if __name__ == "__main__":
    asyncio.run(main())
