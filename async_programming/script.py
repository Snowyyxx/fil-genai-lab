import asyncio
import time

# Define an asynchronous coroutine
async def fetch_data(source_id: int, delay: int):
    print(f"-> Starting fetch for source {source_id}...")
    # Simulate a network request using non-blocking sleep
    await asyncio.sleep(delay)
    print(f"<- Finished fetch for source {source_id}!")
    return {"source": source_id, "data": "success"}

async def main():
    start_time = time.time()
    
    # Schedule multiple coroutines to run concurrently
    task1 = asyncio.create_task(fetch_data(1, 2))
    task2 = asyncio.create_task(fetch_data(2, 3))
    task3 = asyncio.create_task(fetch_data(3, 1))
    
    # Wait for all tasks to complete and gather results
    results = await asyncio.gather(task1, task2, task3)
    
    end_time = time.time()
    print(f"\nAll tasks completed in {end_time - start_time:.2f} seconds.")
    print(f"Results: {results}")

# Run the event loop
if __name__ == "__main__":
    asyncio.run(main())
