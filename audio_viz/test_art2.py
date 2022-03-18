import asyncio
from pyartnet import ArtNetNode

async def run():
# # Run this code in your async function
    # node = ArtNetNode('168.254.42.40')
    # node = ArtNetNode('192.168.1.110')
    node = ArtNetNode('172.0.0.1')
    await node.start()

    # Create universe 0
    universe = node.add_universe(1)

    # Add a channel to the universe which consists of 3 values
    # Default size of a value is 8Bit (0..255) so this would fill
    # the DMX values 1..3 of the universe
    # channel = universe.add_channel(start=1, width=100)
    channel = universe.add_channel(start=1, width=100)

    # Fade channel to 255,0,0 in 5s
    # The fade will automatically run in the background
    UP = [255 for _ in range(100)]
    channel.add_fade(UP, 1000)

    # this can be used to wait till the fade is complete
    await channel.wait_till_fade_complete()

asyncio.run(run())
# await run()